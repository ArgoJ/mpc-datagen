import numpy as np

from numpy.typing import NDArray
from acados_template import AcadosOcpSolver
from dataclasses import replace

from .mpc_solve import solve_mpc_closed_loop, EpsBandConfig
from .sampler import UniqueBoundedSampler, SamplerBase
from ..extractor import MPCConfigExtractor
from ..mpc_data import MPCDataset
from pkg_logger import get_package_logger

__logger__ = get_package_logger(__name__)


class MPCDataGenerator:
    """
    Generator for MPC closed-loop datasets.
    """
    def __init__(
        self,
        solver: AcadosOcpSolver,
        T_sim: int,
        sampler: SamplerBase | None = None,
        xeps_cfg: EpsBandConfig | None = None,
        reset_solver: bool = False,
    ):
        """
        Initializes the MPC Data Generator.

        Parameters
        ----------
        solver : AcadosOcpSolver
            The initialized Acados OCP solver instance.
        T_sim : int
            Number of simulation steps per trajectory.
        sampler : Sampler, optional
            Configuration for initial state sampling. If None, defaults to uniform sampling in [-1, 1]^nx with no uniqueness filtering.
        xeps_cfg : EpsBandConfig, optional
            Configuration for epsilon band checks used when `break_on` is `BreakOn.IN_EPS` or `BreakOn.ALL`.
        reset_solver : bool
            If True, resets the solver states to zero before each simulation.
        """
        self.solver = solver
        self.xeps_cfg = xeps_cfg
        self.reset_solver = reset_solver

        self.mpc_config = MPCConfigExtractor.get_cfg(self.solver)
        self.mpc_config.T_sim = T_sim

        if sampler is None:
            default_bounds = np.stack(
                (self.mpc_config.constraints.lbx, self.mpc_config.constraints.ubx),
                axis=0)
            sampler = UniqueBoundedSampler(bounds=default_bounds)
        self.sampler = sampler
        self._validate_sampler()
    
    def _validate_sampler(self) -> None:
        if not isinstance(self.sampler, SamplerBase):
            raise ValueError("Sampler must be an instance of SamplerBase.")
        if self.sampler.bounds.shape[1] != self.mpc_config.nx:
            raise ValueError(f"Sampler bounds dimension {self.sampler.bounds.shape[1]} does not match MPC state dimension {self.mpc_config.nx}.")

    def generate(self, n_samples: int, only_feasible: bool = False) -> MPCDataset:
        """
        Generates a dataset of MPC closed-loop trajectories starting from random initial states.

        Parameters
        ----------
        n_samples : int
            Number of trajectories to generate.
        only_feasible : bool, optional
            If True, only trajectories that are feasible are saved.

        Returns
        -------
        dataset : MPCDataset
            A dataset containing the generated trajectories.
        """
        dataset = MPCDataset()

        feasible_count = 0
        iter_count = 0
        with __logger__.tqdm(total=n_samples, desc="Generating Trajectories") as pbar:
            while len(dataset) < n_samples:
                try:
                    x0 = self.sampler.sample_x0()
                except RuntimeError as e:
                    __logger__.error(f"Sampling failed: {e} \n STOPPING GENERATION")
                    break

                temp_cfg = replace(
                    self.mpc_config, 
                    constraints=replace(self.mpc_config.constraints, x0=x0))

                if self.reset_solver:
                    self.solver.reset()

                mpc_data = solve_mpc_closed_loop(
                    solver=self.solver,
                    cfg=temp_cfg,
                    xeps_cfg=self.xeps_cfg,
                )

                iter_count += 1
                if not only_feasible or mpc_data.is_feasible():
                    dataset.add(mpc_data)
                    
                    feasible_count += 1
                    feasible_percentage = feasible_count / iter_count * 100
                    pbar.update(1)
                    pbar.set_postfix_str(f"feasible: {feasible_percentage:.1f}%")

        return dataset
