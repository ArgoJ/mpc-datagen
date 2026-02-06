import numpy as np

from numpy.typing import NDArray
from acados_template import AcadosOcpSolver
from tqdm import tqdm
from dataclasses import replace
from typing import Literal

from .mpc_solve import solve_mpc_closed_loop, BreakOn, EpsBandConfig
from ..extractor import MPCConfigExtractor
from ..mpc_data import MPCDataset
from ..package_logger import PackageLogger, DEFAULT_MODULE_NAME

class MPCDataGenerator:
    """
    Generator for MPC closed-loop datasets.
    """
    def __init__(
        self,
        solver: AcadosOcpSolver,
        x0_bounds: NDArray,
        T_sim: int,
        break_on: BreakOn = BreakOn.INFEASIBLE,
        xeps_cfg: EpsBandConfig | None = None,
        seed: int | None = None,
        bound_type: Literal["absolute", "percentage"] = "absolute",
        reset_solver: bool = False,
    ):
        """
        Initializes the MPC Data Generator.

        Parameters
        ----------
        solver : AcadosOcpSolver
            The initialized Acados OCP solver instance.
        x0_bounds : NDArray
            Bounds for initial state sampling (shape: (2, nx)) (lower_bounds, upper_bounds).
            In 'percentage' mode this is the single (shape: (nx,))  percentage array (0-1) used to shrink the solver's
            state bounds toward their midpoint.
        T_sim : int
            Number of simulation steps per trajectory.
        break_on : BreakOn
            Condition to stop simulation if the solver fails.
        xeps_cfg : EpsBandConfig, optional
            Configuration for epsilon band checks used when `break_on` is `BreakOn.IN_EPS` or `BreakOn.ALL`.
        seed : int, optional
            Random seed for reproducibility.
        bound_type : Literal["absolute", "percentage"]
            Type of bounds: 'absolute' (default) or 'percentage'.
            'percentage' shrinks the solver's lbx/ubx around the midpoint with a single percentage array.
        reset_solver : bool
            If True, resets the solver states to zero before each simulation.
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.solver = solver
        self.break_on = break_on
        self.xeps_cfg = xeps_cfg
        self.reset_solver = reset_solver
        self.mpc_config = MPCConfigExtractor.get_cfg(self.solver)
        self.mpc_config.T_sim = T_sim
        
        self.sample_lb = None
        self.sample_ub = None
        self.calc_sample_bounds(bound_type, x0_bounds)
        
    def calc_sample_bounds(self, bound_type: str, x0_bounds: NDArray) -> None:
        nx = self.mpc_config.nx
        
        if bound_type == "percentage":
            percentages = x0_bounds

            # Basic validation
            if percentages.shape[0] != nx:
                raise ValueError(f"Percentage array must have shape ({nx},). Got {percentages.shape}.")
            if np.any(percentages <= 0) or np.any(percentages > 1):
                raise ValueError("Percentages must be in the interval (0, 1].")

            if np.any(~np.isfinite(self.mpc_config.constraints.lbx)) or np.any(~np.isfinite(self.mpc_config.constraints.ubx)):
                raise ValueError("Percentage mode requires finite lbx/ubx for all states.")

            self.sample_lb, self.sample_ub = self._calculate_percentage_bounds(
                self.mpc_config.constraints.lbx, self.mpc_config.constraints.ubx, percentages)
        elif bound_type == "absolute":
            if x0_bounds.shape != (2, nx):
                raise ValueError(f"Bounds must have shape (2, {nx}) for absolute mode. Got {x0_bounds.shape}.")
            self.sample_lb = x0_bounds[0]
            self.sample_ub = x0_bounds[1]
        else:
            raise ValueError(f"Unknown bound_type: {bound_type}. Use 'absolute' or 'percentage'.")
        
            
    def generate(self, n_samples: int) -> MPCDataset:
        """
        Generates a dataset of MPC closed-loop trajectories starting from random initial states.

        Parameters
        ----------
        n_samples : int
            Number of trajectories to generate.

        Returns
        -------
        dataset : MPCDataset
            A dataset containing the generated trajectories.
        """
        dataset = MPCDataset()
        
        restored_handlers = []
        tqdm_handler, restored_handlers = PackageLogger.add_tqdm_handler()

        try:
            for _ in tqdm(range(n_samples), desc="Generating Trajectories"):
                x0 = np.random.uniform(self.sample_lb, self.sample_ub)
                temp_cfg = replace(
                    self.mpc_config, 
                    constraints=replace(self.mpc_config.constraints, x0=x0))

                if self.reset_solver:
                    self.solver.reset()

                mpc_data = solve_mpc_closed_loop(
                    solver=self.solver,
                    cfg=temp_cfg,
                    break_on=self.break_on,
                    xeps_cfg=self.xeps_cfg,
                )

                dataset.add(mpc_data)
        finally:
            if tqdm_handler:
                PackageLogger.restore_handlers(DEFAULT_MODULE_NAME, tqdm_handler, restored_handlers)

        return dataset

    @staticmethod
    def _calculate_percentage_bounds(lbx: NDArray, ubx: NDArray, percentages: NDArray) -> tuple[NDArray, NDArray]:
        """Shrink bounds symmetrically around the midpoint using the provided percentages"""
        mid = 0.5 * (lbx + ubx)
        half_range = 0.5 * (ubx - lbx)
        shrink = (1.0 - percentages) * half_range

        sample_lb = mid - (half_range - shrink)
        sample_ub = mid + (half_range - shrink)

        if np.any(sample_lb >= sample_ub):
            raise ValueError("Computed sampling bounds are invalid (lower >= upper). Check percentages and solver bounds.")
        
        return sample_lb, sample_ub