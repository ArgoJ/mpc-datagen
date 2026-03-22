import os
import shutil
import tempfile
import numpy as np

from numpy.typing import NDArray
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosOcpBatchSolver
from dataclasses import replace, dataclass

from .mpc_solve import solve_mpc_closed_loop
from .sampler import SamplerBase
from ..extractor import get_primary_solver, resolve_solver, extract_cfg, is_batch_solver
from ..mpc_data import MPCDataset, MPCConfig, MPCData, MPCMeta, MPCTrajectory
from pkg_logger import get_package_logger, suppress_native_output

__logger__ = get_package_logger(__name__)

@dataclass
class EpsBandConfig:
    """
    Configuration for epsilon band checks in closed-loop simulation.
    
    Parameters
    ----------
    eps_band : float | NDArray
        Epsilon band around the reference output `yref` used. 
        Can be a scalar (same band for all states) or a vector of shape (nx,) for per-state bands.  
        Default is 1e-2.
    eps_consecutive : int
        Number of consecutive steps within the eps_band required to trigger a break. Must be >= 1.  
        Default is 5.
    """
    eps_band: float | NDArray = 1e-2
    eps_consecutive: int = 5

    def __post_init__(self):
        if self.eps_consecutive < 1:
            raise ValueError("eps_consecutive must be >= 1.")

    def resolve_eps_band(self, nx: int) -> None:
        """Normalize `eps_band` to a per-state vector of shape (nx,)."""
        if np.isscalar(self.eps_band):
            eps_vec = np.full(int(nx), float(self.eps_band), dtype=float)
        else:
            eps_vec = np.asarray(self.eps_band, dtype=float).reshape(-1)
            if eps_vec.shape != (int(nx),):
                raise ValueError(f"eps_band must be a scalar or shape ({int(nx)},), got {eps_vec.shape}")

        if not np.all(np.isfinite(eps_vec)):
            raise ValueError("eps_band must contain only finite values")
        if np.any(eps_vec < 0.0):
            raise ValueError("eps_band must be >= 0 component-wise")
        self.eps_band = eps_vec

    def in_state_band(self, x: NDArray, cfg: MPCConfig) -> NDArray:
        """Return True if |x - x_ref| <= eps_band component-wise.

        `eps_band` may be a scalar or a vector of shape (nx,) to account for different state scales.
        """
        x = np.asarray(x, dtype=float)
        x_ref = cfg.cost.yref @ cfg.cost.Vx
        x_ref = np.asarray(x_ref, dtype=float).reshape(-1)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != cfg.nx:
            raise ValueError(f"x must have shape (B, {cfg.nx}), got {x.shape}")
        return np.all(np.abs(x - x_ref[None, :]) <= self.eps_band[None, :], axis=1)

def _set_initial_state_constraints(
    solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    current_x: NDArray,
) -> None:
    if is_batch_solver(solver):
        current_x = np.asarray(current_x, dtype=float).reshape(1, -1)
    solver.constraints_set(0, "lbx", current_x)
    solver.constraints_set(0, "ubx", current_x)

def _set_x_guess(
    solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    x_guess: NDArray,
) -> None:
    if is_batch_solver(solver):
        solver.set_flat("x", np.asarray(x_guess, dtype=float).reshape(1, -1))
    else:
        solver.set_flat("x", x_guess)

def _is_sqp_solver(solver: AcadosOcpSolver | AcadosOcpBatchSolver) -> bool:
    primary_solver = get_primary_solver(solver)
    return primary_solver.acados_ocp.solver_options.nlp_solver_type.lower() == "sqp"

def _solve_once(
    solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    x0: NDArray,
    x_guess: NDArray | None = None,
) -> int | list[int]:
    _set_initial_state_constraints(solver, x0)
    if x_guess is not None:
        _set_x_guess(solver, x_guess)
    with suppress_native_output(suppress_stdout=True, suppress_stderr=True):
        if is_batch_solver(solver):
            solver.solve(n_batch=1)
            return solver.status
        return solver.solve()



class MPCDataGenerator:
    """
    Generator for MPC closed-loop datasets.
    """
    def __init__(
        self,
        solver: AcadosOcpSolver | AcadosOcpBatchSolver,
        T_sim: int,
        sampler: SamplerBase | None = None,
        xeps_cfg: EpsBandConfig | None = None,
        reset_solver: bool = False,
        solver_regen_interval: int | None = None,
    ):
        """
        Initializes the MPC Data Generator.

        Parameters
        ----------
        solver : AcadosOcpSolver | AcadosOcpBatchSolver
            The initialized Acados OCP solver instance.
        T_sim : int
            Number of simulation steps per trajectory.
        sampler : Sampler, optional
            Configuration for initial state sampling. If None, defaults to uniform sampling in [-1, 1]^nx with no uniqueness filtering.
        xeps_cfg : EpsBandConfig, optional
            Configuration for epsilon band checks used when `break_on` is `BreakOn.IN_EPS` or `BreakOn.ALL`.
        reset_solver : bool
            If True, resets the solver states to zero before each simulation.
        solver_regen_interval : int | None
            If not None, regenerates the solver every N iterations to reset internal state.
        """
        self.solver = solver
        self.reset_solver = reset_solver
        self.solver_regen_interval = solver_regen_interval

        self.mpc_cfg = extract_cfg(resolve_solver(self.solver))
        self.T_sim = T_sim

        self.xeps_cfg = xeps_cfg
        self.sampler = sampler
        self._resolve()
    
    def _resolve(self) -> None:
        temp_solver = resolve_solver(self.solver)
        nx = temp_solver.acados_ocp.dims.nx
        self._validate_sampler(nx)
        self._resolve_eps_cfg(nx)
    
    def _validate_sampler(self, nx: int) -> None:
        if not isinstance(self.sampler, SamplerBase):
            raise ValueError("Sampler must be an instance of SamplerBase.")
        if self.sampler.bounds.shape[1] != nx:
            raise ValueError((
                f"Sampler bounds dimension {self.sampler.bounds.shape[1]} "
                f"does not match MPC state dimension {nx}."
            ))
    
    def _resolve_eps_cfg(self, nx: int) -> None:
        if self.xeps_cfg is not None and not isinstance(self.xeps_cfg, EpsBandConfig):
            raise ValueError("xeps_cfg must be an instance of EpsBandConfig or None.")
        if self.xeps_cfg is not None:
            self.xeps_cfg.resolve_eps_band(nx)
        
    @staticmethod
    def _generate_empty_dataset(cfg: MPCConfig, n_samples: int) -> MPCDataset:
        dataset = MPCDataset()
        for _ in range(n_samples):
            dataset.add(MPCData(
                config=cfg,
                trajectory=MPCTrajectory.empty_from_cfg(cfg),
                meta=MPCMeta()
            ))
        return dataset

    # @staticmethod
    # def _regenerate_solver(
    #     solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    # ) -> AcadosOcpSolver | AcadosOcpBatchSolver:
    #     """
    #     Regenerates the solver instance to reset its internal state.
    #     """
    #     with suppress_native_output(suppress_stdout=True, suppress_stderr=True):
    #         if is_batch_solver(solver):
    #             return AcadosOcpBatchSolver(
    #                 ocp=solver.ocp_solvers[0].acados_ocp,
    #                 N_batch_init=solver.n_batch_current,
    #                 build=False,
    #                 generate=False,
    #                 verbose=False,
    #                 check_code_reuse_possible=True,
    #             )

    #         return AcadosOcpSolver(
    #             solver.acados_ocp,
    #             build=False,
    #             check_reuse_possible=True,
    #         )

    # def cleanup(self) -> None:
    #     """Cleans up temporary files and resources associated with the solver."""
    #     if self.solver is None:
    #         return

    #     solver = self.solver
    #     cleanup_solver = get_primary_solver(solver)

    #     tmp_dir = getattr(solver, "_mpc_datagen_temp_dir", None)
    #     if isinstance(tmp_dir, str) and tmp_dir:
    #         try:
    #             shutil.rmtree(tmp_dir, ignore_errors=True)
    #         except Exception as err:
    #             __logger__.debug(f"Temporary solver directory cleanup failed: {err}")
    #     else:
    #         json_file = getattr(cleanup_solver.acados_ocp, "json_file", None)
    #         code_export_directory = getattr(cleanup_solver.acados_ocp, "code_export_directory", None)

    #         if isinstance(json_file, str) and json_file:
    #             try:
    #                 if os.path.isfile(json_file):
    #                     os.remove(json_file)
    #             except Exception as err:
    #                 __logger__.debug(f"Failed to remove solver json file '{json_file}': {err}")

    #         if isinstance(code_export_directory, str) and code_export_directory:
    #             try:
    #                 if os.path.isdir(code_export_directory):
    #                     shutil.rmtree(code_export_directory, ignore_errors=True)
    #             except Exception as err:
    #                 __logger__.debug(
    #                     f"Failed to remove solver code export directory '{code_export_directory}': {err}"
    #                 )

    #     self.solver = None


    # def __del__(self) -> None:
    #     try:
    #         self.cleanup()
    #     except Exception:
    #         pass
        

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
        if self.solver is None:
            raise RuntimeError("Solver is not available. The generator may have been cleaned up.")

        feasible_count = 0
        iter_count = 0
        consequtive_infeasible = 0
        batch_size = self.solver.n_batch_current if is_batch_solver(self.solver) else 1

        dataset = self._generate_empty_dataset(self.mpc_cfg, n_samples)
        with __logger__.tqdm(total=n_samples, desc="Generating Trajectories") as pbar:
            while len(dataset) < n_samples:
                n_remaining = n_samples - len(dataset)
                n_draw = min(batch_size, n_remaining)

                if self.reset_solver:
                    self.solver.reset()

                try:
                    mpc_data = solve_mpc_closed_loop(
                        solver=self.solver,
                        cfg=temp_cfg,
                        xeps_cfg=self.xeps_cfg,
                    )
                except Exception:
                    self.cleanup()
                    raise

                mpc_data_list = mpc_data if isinstance(mpc_data, list) else [mpc_data]

                for sample in mpc_data_list:
                    iter_count += 1
                    if not only_feasible or sample.is_feasible():
                        dataset.add(sample)
                        feasible_count += 1
                        consequtive_infeasible = 0
                        pbar.update(1)
                    else:
                        consequtive_infeasible += 1

                    if self.solver_regen_interval is not None and consequtive_infeasible >= self.solver_regen_interval:
                        __logger__.debug((
                            f"{consequtive_infeasible} consecutive infeasible trajectories. "
                            "Regenerating solver to reset internal state."
                        ))
                        self.solver = self._regenerate_solver(self.solver)
                        consequtive_infeasible = 0

                    if len(dataset) >= n_samples:
                        break

                feasible_percentage = feasible_count / iter_count * 100 if iter_count > 0 else 0.0
                pbar.set_postfix_str(f"feasible: {feasible_percentage:.1f}%")

        return dataset


def get_temp_solver(
    ocp: AcadosOcp,
    *args,
    **kwargs,
) -> AcadosOcpSolver:
    json_file = f"{ocp.model.name}_ocp.json"
    temp_dir = tempfile.mkdtemp(prefix=f"acados_{ocp.model.name}_")
    __logger__.info(f"Created temporary directory for solver: {temp_dir}")
    ocp.code_export_directory = os.path.join(temp_dir, "code_export")
    json_file = os.path.join(temp_dir, f"{ocp.model.name}_ocp.json")

    solver = AcadosOcpSolver(ocp, json_file=json_file, *args, **kwargs)
    setattr(solver, "_mpc_datagen_temp_dir", temp_dir)

    return solver