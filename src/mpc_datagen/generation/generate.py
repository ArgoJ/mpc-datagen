import numpy as np

from numpy.typing import NDArray
from acados_template import AcadosOcpSolver
from tqdm import tqdm
from dataclasses import replace

from .mpc_solve import solve_mpc_closed_loop, EpsBandConfig
from .sampler import Sampler
from ..extractor import MPCConfigExtractor
from ..mpc_data import MPCDataset
from ..package_logger import PackageLogger, DEFAULT_MODULE_NAME

__logger__ = PackageLogger.get_logger(__name__)


class MPCDataGenerator:
    """
    Generator for MPC closed-loop datasets.
    """
    def __init__(
        self,
        solver: AcadosOcpSolver,
        T_sim: int,
        sampler: Sampler | None = None,
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
        self.sampler.cfg_post_init(self.mpc_config)

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
        restored_handlers = []
        tqdm_handler, restored_handlers = PackageLogger.add_tqdm_handler()

        try:
            for _ in tqdm(range(n_samples), desc="Generating Trajectories"):
                x0 = self.sampler.sample_unique_x0(accepted_x0)
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
        finally:
            if tqdm_handler:
                PackageLogger.restore_handlers(DEFAULT_MODULE_NAME, tqdm_handler, restored_handlers)

        return dataset