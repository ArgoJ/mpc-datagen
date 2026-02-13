from numpy.typing import NDArray
from acados_template import AcadosOcpSolver
from dataclasses import replace

from .mpc_solve import solve_mpc_closed_loop, EpsBandConfig
from .sampler import Sampler, SamplerBase
from ..extractor import MPCConfigExtractor
from ..mpc_data import MPCDataset
from ..package_logger import get_package_logger

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
            sampler = Sampler()
        self.sampler = sampler
        self.sampler.post_init_cfg(self.mpc_config)

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
        accepted_x0: list[NDArray] = []

        with __logger__.tqdm(range(n_samples), desc="Generating Trajectories") as pbar:
            for _ in pbar:
                try:
                    x0 = self.sampler.sample_x0(accepted_x0)
                except RuntimeError as e:
                    __logger__.error(f"Sampling failed: {e} \n - Stopping generation.")
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

                dataset.add(mpc_data)
                accepted_x0.append(x0)

        return dataset