import os
import shutil
import tempfile
import copy
import numpy as np
from datetime import datetime, timezone

from numpy.typing import NDArray
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosOcpBatchSolver
from dataclasses import replace, dataclass

from .sampler import SamplerBase
from .solver_adapter import SolverAdapter, AcadosBatchSolverAdapter, AcadosSolverAdapter
from ..extractor import get_primary_solver, resolve_solver, extract_cfg, is_batch_solver
from ..mpc_data import MPCDataset, MPCConfig, MPCData, MPCMeta, MPCTrajectory
from pkg_logger import get_package_logger, suppress_native_output

__logger__ = get_package_logger(__name__)


def create_solver_adapter(solver: AcadosOcpSolver | AcadosOcpBatchSolver) -> SolverAdapter:
    if isinstance(solver, AcadosOcpBatchSolver):
        return AcadosBatchSolverAdapter(solver)
    return AcadosSolverAdapter(solver)


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
        reset_solver: bool = True,
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
        self.solver_adapter = create_solver_adapter(solver)
        self.reset_solver = reset_solver
        self.solver_regen_interval = solver_regen_interval

        self.mpc_cfg = extract_cfg(resolve_solver(solver))
        self.mpc_cfg.T_sim = T_sim

        self.xeps_cfg = xeps_cfg
        self.sampler = sampler
        self._resolve()

        self.iter_count = 0
        self.generated_count = 0
        self.feasible_count = 0

        self.batch_size = self.solver_adapter.batch_size
        self.is_sqp = self.solver_adapter.is_sqp()
    
    @property
    def feasibility_percentage(self) -> float:
        return self.feasible_count / self.iter_count * 100 if self.iter_count > 0 else 0.0
    
    def _resolve(self) -> None:
        nx = self.mpc_cfg.nx
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
            cfg_copy = copy.deepcopy(cfg)
            dataset.add(MPCData(
                config=cfg_copy,
                trajectory=MPCTrajectory.empty_from_cfg(cfg_copy),
                meta=MPCMeta()
            ))
        return dataset

    def _new_empty_entry(self, data_idx: int) -> MPCData:
        cfg_copy = copy.deepcopy(self.mpc_cfg)
        entry = MPCData(
            config=cfg_copy,
            trajectory=MPCTrajectory.empty_from_cfg(cfg_copy),
            meta=MPCMeta(id=data_idx),
        )
        return entry

    def _get_x_guess(self, x0: NDArray) -> NDArray:
        return np.repeat(np.atleast_2d(x0)[:, np.newaxis, :], self.mpc_cfg.N + 1, axis=1)

    def _solve_once(self, n: int) -> NDArray:
        with suppress_native_output(suppress_stdout=True, suppress_stderr=True):
            self.solver_adapter.solve(n)
        return self.solver_adapter.get_status(n)

    @staticmethod
    def _is_feasible_status(status: int) -> bool:
        return status in (0, 5)

    def _in_eps_band(self, x: NDArray) -> bool:
        if self.xeps_cfg is None:
            return False
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        return bool(self.xeps_cfg.in_state_band(x_arr, self.mpc_cfg)[0])

    def _in_eps_band_batch(self, x: NDArray) -> NDArray:
        if self.xeps_cfg is None:
            return np.zeros((x.shape[0],), dtype=bool)
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        return self.xeps_cfg.in_state_band(x_arr, self.mpc_cfg)

    def _fill_dataset(self, dataset: MPCDataset, slot_data_idxs: list[int], n_active: int) -> None:
        if n_active == 0:
            return

        x_batch = self.solver_adapter.get_x(n_active, self.mpc_cfg.N, self.mpc_cfg.nx)
        u_batch = self.solver_adapter.get_u(n_active, self.mpc_cfg.N, self.mpc_cfg.nu)
        status_batch = np.asarray(self.solver_adapter.get_status(n_active), dtype=int).reshape(-1)
        cost_batch = self.solver_adapter.get_cost(n_active)
        time_batch = self.solver_adapter.get_time(n_active)

        now_iso = datetime.now(timezone.utc).isoformat()

        for slot_idx in range(n_active):
            data_idx = slot_data_idxs[slot_idx]
            if data_idx == -1:
                continue # Dummy-Slot

            entry = dataset[data_idx]
            traj = entry.trajectory
            meta = entry.meta
            cfg = entry.config

            if meta.id != int(data_idx):
                __logger__.warning(f"Wrong idx in dataset entry meta: {meta.id} vs slot idx {slot_idx} and data idx {data_idx}. Overwriting meta id to match data idx.")
                meta.id = int(data_idx)

            k = int(meta.steps_simulated)
            if k >= int(cfg.T_sim):
                continue

            x_pred = x_batch[slot_idx]
            u_pred = u_batch[slot_idx]
            status = int(status_batch[slot_idx])
            solve_time = float(time_batch[slot_idx])
            cost = float(cost_batch[slot_idx])

            traj.states[k, :] = x_pred[0, :]
            if k + 1 < traj.states.shape[0]:
                traj.states[k + 1, :] = x_pred[1, :]
            traj.inputs[k, :] = u_pred[0, :]
            traj.V_solver[k] = cost

            if traj.predicted_states is not None and k < traj.predicted_states.shape[0]:
                traj.predicted_states[k, :, :] = x_pred
            if traj.predicted_inputs is not None and k < traj.predicted_inputs.shape[0]:
                traj.predicted_inputs[k, :, :] = u_pred

            if not meta.timestamp:
                meta.timestamp = now_iso
            meta.steps_simulated = k + 1
            meta.status_codes.append(status)
            meta.solve_time_total += solve_time
            meta.solve_time_max = max(meta.solve_time_max, solve_time)
            meta.solve_time_mean = meta.solve_time_total / max(meta.steps_simulated, 1)
            
            step_feasible = self._is_feasible_status(status)
            meta.feasible = step_feasible if len(meta.status_codes) == 1 else (meta.feasible and step_feasible)

    def _shift(self, arr: NDArray, n_active: int) -> None:
        if self.is_sqp:
            arr[:n_active, :-1, :] = arr[:n_active, 1:, :]
        else:
            arr[:n_active, 0, :] = arr[:n_active, 1, :]

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
        dataset = self._generate_empty_dataset(self.mpc_cfg, n_samples)
        
        n_active = min(self.batch_size, n_samples)

        slot_data_idxs = np.arange(n_active, dtype=int)
        next_data_idx = n_active

        current_x = np.full((n_active, self.mpc_cfg.N + 1, self.mpc_cfg.nx), np.nan)
        current_u = np.zeros((n_active, self.mpc_cfg.N, self.mpc_cfg.nu))
        eps_hits = np.zeros(n_active, dtype=int)
        fail_hits = np.zeros(n_active, dtype=int)

        x0_init = self.sampler.sample_x0(n_active)
        current_x[:] = self._get_x_guess(x0_init)

        with __logger__.tqdm(total=n_samples, desc="Generating Trajectories") as pbar:
            while self.generated_count < n_samples:

                # --- Solve OCP ---
                self.solver_adapter.set_x0(current_x[:n_active, 0, :])
                if self.is_sqp:
                    self.solver_adapter.set_x_guess(current_x[:n_active], self.mpc_cfg.N, self.mpc_cfg.nx)
                    self.solver_adapter.set_u_guess(current_u[:n_active], self.mpc_cfg.N, self.mpc_cfg.nu)

                self._solve_once(n_active)
                self._fill_dataset(dataset, slot_data_idxs.tolist(), n_active)
                
                # --- Extract Data & Update States ---
                current_x[:n_active, :] = self.solver_adapter.get_x(n_active, self.mpc_cfg.N, self.mpc_cfg.nx)
                current_u[:n_active, :] = self.solver_adapter.get_u(n_active, self.mpc_cfg.N, self.mpc_cfg.nu)
                solved_status = np.asarray(self.solver_adapter.get_status(n_active), dtype=int).reshape(-1)

                # --- State Shift ---
                self._shift(current_x, n_active)
                self._shift(current_u, n_active)

                # --- Evaluation ---
                # Feasibility Check
                is_feasible = np.isin(solved_status, (0, 5))
                fail_hits[:n_active] = np.where(is_feasible, 0, fail_hits[:n_active] + 1)
                
                # Epsilon Band Check
                in_eps = self._in_eps_band_batch(current_x[:n_active, 0, :])
                eps_hits[:n_active] = np.where(is_feasible & in_eps, eps_hits[:n_active] + 1, 0)
                reached_eps = (self.xeps_cfg is not None) and (eps_hits[:n_active] >= int(self.xeps_cfg.eps_consecutive))
                
                # Simulation Step Check
                steps = np.array([dataset[int(idx)].meta.steps_simulated if idx != -1 else 0 for idx in slot_data_idxs])
                reached_tsim = steps >= int(self.mpc_cfg.T_sim)

                # Combined Termination Mask
                is_done = (~is_feasible) | reached_tsim | reached_eps
                done_slots = np.flatnonzero(is_done)

                # --- Batch Update ---
                if done_slots.size > 0:
                    if self.reset_solver:
                        self.solver_adapter.reset_solvers(done_slots)

                    if self.solver_regen_interval is not None:
                        regen_mask = fail_hits[done_slots] >= self.solver_regen_interval
                        if np.any(regen_mask):
                            regen_slots = done_slots[regen_mask]
                            self.solver_adapter.regenerate(regen_slots)
                            fail_hits[regen_slots] = 0

                    valid_mask = slot_data_idxs[done_slots] != -1
                    valid_slots = done_slots[valid_mask]
                    valid_data_idxs = slot_data_idxs[valid_slots]

                    if valid_slots.size > 0:
                        traj_feasible_arr = is_feasible[valid_slots]

                        if only_feasible:
                            keep_mask = traj_feasible_arr
                        else:
                            keep_mask = np.ones(valid_slots.size, dtype=bool)

                        n_kept = int(np.sum(keep_mask))

                        self.iter_count += valid_slots.size
                        self.feasible_count += int(np.sum(traj_feasible_arr))

                        if n_kept > 0:
                            self.generated_count += n_kept
                            pbar.update(n_kept)

                        kept_slots = valid_slots[keep_mask]
                        rejected_data_idxs = valid_data_idxs[~keep_mask]

                        n_new_alloc = min(n_kept, n_samples - next_data_idx)
                        if n_new_alloc > 0:
                            new_data_idxs = np.arange(next_data_idx, next_data_idx + n_new_alloc)
                            slot_data_idxs[kept_slots[:n_new_alloc]] = new_data_idxs
                            next_data_idx += n_new_alloc
                        else:
                            new_data_idxs = np.array([], dtype=int)

                        if n_kept > n_new_alloc:
                            slot_data_idxs[kept_slots[n_new_alloc:]] = -1

                        idxs_to_reset = np.concatenate([rejected_data_idxs, new_data_idxs]).astype(int)
                        for idx in idxs_to_reset:
                            dataset.memory_buffer[idx] = self._new_empty_entry(idx)

                    # Resample initial states for done slots
                    new_x0 = self.sampler.sample_x0(done_slots.size)
                    current_x[done_slots, :] = self._get_x_guess(new_x0)
                    current_u[done_slots, :] = 0
                    eps_hits[done_slots] = 0

                pbar.set_postfix_str(f"feasible: {self.feasibility_percentage:.1f}%")

        dataset.finalize(recalculate_costs=True, truncate=True)
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