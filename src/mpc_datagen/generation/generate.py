import os
import shutil
import tempfile
import numpy as np

from numpy.typing import NDArray
from acados_template import AcadosOcpSolver, AcadosOcp
from dataclasses import replace

from .mpc_solve import solve_mpc_closed_loop, EpsBandConfig
from .sampler import UniqueBoundedSampler, SamplerBase
from ..extractor import MPCConfigExtractor
from ..mpc_data import MPCDataset
from pkg_logger import get_package_logger, suppress_native_output

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
        solver_regen_interval: int | None = None,
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
        solver_regen_interval : int | None
            If not None, regenerates the solver every N iterations to reset internal state.
        """
        self.solver = solver
        self.xeps_cfg = xeps_cfg
        self.reset_solver = reset_solver
        self.solver_regen_interval = solver_regen_interval

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
            raise ValueError((
                f"Sampler bounds dimension {self.sampler.bounds.shape[1]} "
                f"does not match MPC state dimension {self.mpc_config.nx}."
            ))

    @staticmethod
    def _regenerate_solver(solver: AcadosOcpSolver) -> AcadosOcpSolver:
        """
        Regenerates the solver instance to reset its internal state.
        """
        with suppress_native_output(suppress_stdout=True, suppress_stderr=True):
            return AcadosOcpSolver(
                solver.acados_ocp,
                build=False,
                check_reuse_possible=True,
            )

    def cleanup(self) -> None:
        """Cleans up temporary files and resources associated with the solver."""
        if self.solver is None:
            return

        solver = self.solver

        tmp_dir = getattr(solver, "_mpc_datagen_temp_dir", None)
        if isinstance(tmp_dir, str) and tmp_dir:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as err:
                __logger__.debug(f"Temporary solver directory cleanup failed: {err}")
        else:
            json_file = getattr(solver.acados_ocp, "json_file", None)
            code_export_directory = getattr(solver.acados_ocp, "code_export_directory", None)

            if isinstance(json_file, str) and json_file:
                try:
                    if os.path.isfile(json_file):
                        os.remove(json_file)
                except Exception as err:
                    __logger__.debug(f"Failed to remove solver json file '{json_file}': {err}")

            if isinstance(code_export_directory, str) and code_export_directory:
                try:
                    if os.path.isdir(code_export_directory):
                        shutil.rmtree(code_export_directory, ignore_errors=True)
                except Exception as err:
                    __logger__.debug(
                        f"Failed to remove solver code export directory '{code_export_directory}': {err}"
                    )

        self.solver = None


    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

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

        dataset = MPCDataset()

        feasible_count = 0
        iter_count = 0
        consequtive_infeasible = 0
        with __logger__.tqdm(total=n_samples, desc="Generating Trajectories") as pbar:
            while len(dataset) < n_samples:
                try:
                    x0 = self.sampler.sample_x0()
                except RuntimeError as e:
                    __logger__.warning(f"Sampling failed: {e} \n STOPPING GENERATION")
                    self.cleanup()
                    break

                temp_cfg = replace(
                    self.mpc_config, 
                    constraints=replace(self.mpc_config.constraints, x0=x0))

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

                iter_count += 1
                if not only_feasible or mpc_data.is_feasible():
                    dataset.add(mpc_data)
                    
                    feasible_count += 1
                    consequtive_infeasible = 0
                    feasible_percentage = feasible_count / iter_count * 100
                    pbar.update(1)
                    pbar.set_postfix_str(f"feasible: {feasible_percentage:.1f}%")
                else:
                    consequtive_infeasible += 1
                    if (self.solver_regen_interval is not None and \
                        consequtive_infeasible >= self.solver_regen_interval):
                        __logger__.debug((
                            f"{consequtive_infeasible} consecutive infeasible trajectories. "
                            f"Regenerating solver to reset internal state."
                        ))
                        self.solver = self._regenerate_solver(self.solver)
                        consequtive_infeasible = 0

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