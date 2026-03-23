import os
import shutil
import tempfile
import numpy as np
from datetime import datetime, timezone

from numpy.typing import NDArray
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosOcpBatchSolver
from dataclasses import replace, dataclass

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

        self.iter_count = 0
        self.generated_count = 0
        self.feasible_count = 0
        self.batch_size = self.solver.n_batch_current if is_batch_solver(self.solver) else 1
        self.is_sqp = self._is_sqp_solver()
    
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

    @property
    def feasibility_percentage(self) -> float:
        return self.feasible_count / self.iter_count * 100 if self.iter_count > 0 else 0.0
    
    def _set_x0(self, x0: NDArray) -> None:
        if not is_batch_solver(self.solver):
            x0 = np.asarray(x0, dtype=float).flatten()
        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)

    def _set_x_guess(self, x_guess: NDArray) -> None:
        if not is_batch_solver(self.solver):
            x_guess = np.asarray(x_guess, dtype=float).flatten()
        self.solver.set_flat("x", x_guess)
    
    def _get_x_guess(self, x0: NDArray, x: NDArray) -> NDArray:
        primary_solver = get_primary_solver(self.solver)
        return np.tile(x0, primary_solver.acados_ocp.dims.N + 1)

    def _is_sqp_solver(self) -> bool:
        primary_solver = get_primary_solver(self.solver)
        return primary_solver.acados_ocp.solver_options.nlp_solver_type.lower() == "sqp"

    def _solve_once(self, n: int) -> None:
        with suppress_native_output(suppress_stdout=True, suppress_stderr=True):
            if is_batch_solver(self.solver):
                self.solver.solve(n_batch=n)
            else:
                self.solver.solve()
        return self._get_status(n)
        
    def _get_status(self, n: int) -> int | list[int]:
        if is_batch_solver(self.solver):
            return self.solver.status[:n]
        return self.solver.status
    
    def _get_x(self, n: int) -> NDArray:
        return self.solver.get_flat("x", n_batch=n).reshape(n, self.mpc_cfg.N + 1, self.mpc_cfg.nx)
    
    def _get_u(self, n: int) -> NDArray:
        return self.solver.get_flat("u", n_batch=n).reshape(n, self.mpc_cfg.N, self.mpc_cfg.nu)

    def _get_cost_batch(self, n: int) -> NDArray:
        if is_batch_solver(self.solver):
            return np.asarray([
                self.solver.ocp_solvers[i].get_cost()
                for i in range(n)
            ], dtype=float)
        return np.asarray([float(self.solver.get_cost())], dtype=float)

    def _get_time_batch(self, n: int) -> NDArray:
        if is_batch_solver(self.solver):
            return np.asarray([
                self.solver.ocp_solvers[i].get_stats("time_tot")
                for i in range(n)
            ], dtype=float)
        return np.asarray([float(self.solver.get_stats("time_tot"))], dtype=float)

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

    def _new_empty_entry(self, data_idx: int) -> MPCData:
        entry = MPCData(
            config=self.mpc_cfg,
            trajectory=MPCTrajectory.empty_from_cfg(self.mpc_cfg),
            meta=MPCMeta(id=data_idx),
        )
        return entry

    def _fill_dataset(self, dataset: MPCDataset, idxs: list[int]) -> None:
        if not idxs:
            return

        n = len(idxs)
        x_batch = self._get_x(n)
        u_batch = self._get_u(n)
        status_batch = np.asarray(self._get_status(n), dtype=int).reshape(-1)
        cost_batch = self._get_cost_batch(n)
        time_batch = self._get_time_batch(n)

        if status_batch.shape[0] != n:
            raise ValueError(f"Status batch length mismatch: expected {n}, got {status_batch.shape[0]}")

        now_iso = datetime.now(timezone.utc).isoformat()

        for batch_idx, data_idx in enumerate(idxs):
            entry = dataset[data_idx]
            traj = entry.trajectory
            meta = entry.meta
            cfg = entry.config

            k = int(meta.steps_simulated)
            if k >= int(cfg.T_sim):
                continue

            x_pred = x_batch[batch_idx]
            u_pred = u_batch[batch_idx]
            status = int(status_batch[batch_idx])
            solve_time = float(time_batch[batch_idx])
            cost = float(cost_batch[batch_idx])

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

        dataset = self._generate_empty_dataset(self.mpc_cfg, n_samples)
        active_slots = min(self.batch_size, n_samples)
        active_idxs = np.arange(active_slots, dtype=int)
        next_data_idx = active_slots

        current_x0 = np.full((self.batch_size, self.mpc_cfg.nx), np.nan)
        current_x_flat = np.full((self.batch_size, self.mpc_cfg.nx * (self.mpc_cfg.N + 1)), np.nan)
        eps_hits = np.zeros(self.batch_size, dtype=int)

        x0_init = self.sampler.sample_x0(active_slots)
        current_x0[:active_slots, :] = x0_init
        current_x_flat[:active_slots, :] = np.tile(x0_init, (1, self.mpc_cfg.N + 1))

        with __logger__.tqdm(total=n_samples, desc="Generating Trajectories") as pbar:
            while self.generated_count < n_samples:
                n_active = int(active_idxs.shape[0])
                if n_active == 0:
                    break

                self._set_x0(current_x0)
                if self.is_sqp:
                    self._set_x_guess(current_x_flat)

                self._solve_once(n_active)
                self._fill_dataset(dataset, active_idxs.tolist())

                x_pred_batch = self._get_x(n_active)
                current_x_flat[:n_active, :] = x_pred_batch.reshape(n_active, -1)
                current_x0[:n_active, :] = x_pred_batch[:, 1, :]

                solved_status = np.asarray(self._get_status(n_active), dtype=int).reshape(-1)

                keep_mask = np.ones(n_active, dtype=bool)
                refill_slots: list[int] = []

                step_feasible = np.isin(solved_status, (0, 5))
                in_eps = self._in_eps_band_batch(current_x0[:n_active, :])
                eps_hits[:n_active] = np.where(step_feasible & in_eps, eps_hits[:n_active] + 1, 0)

                steps = np.asarray([dataset[int(idx)].meta.steps_simulated for idx in active_idxs], dtype=int)
                reached_tsim = steps >= int(self.mpc_cfg.T_sim)
                reached_eps = (
                    self.xeps_cfg is not None
                    and np.asarray(eps_hits[:n_active] >= int(self.xeps_cfg.eps_consecutive), dtype=bool)
                )
                stop_mask = (~step_feasible) | reached_tsim | reached_eps
                stopped_slots = np.flatnonzero(stop_mask)

                if stopped_slots.size > 0:
                    if is_batch_solver(self.solver):
                        for slot in stopped_slots.tolist():
                            self.solver.ocp_solvers[slot].reset()
                    else:
                        self.solver.reset()

                    if only_feasible:
                        accepted_mask = stop_mask & step_feasible
                    else:
                        accepted_mask = stop_mask

                    accepted_slots = np.flatnonzero(accepted_mask)
                    if accepted_slots.size > 0:
                        self.generated_count += int(accepted_slots.size)
                        self.iter_count += int(accepted_slots.size)
                        accepted_entries_feasible = np.asarray(
                            [dataset[int(active_idxs[s])].meta.feasible for s in accepted_slots],
                            dtype=bool,
                        )
                        self.feasible_count += int(np.sum(accepted_entries_feasible))
                        pbar.update(int(accepted_slots.size))

                    rejected_slots = np.flatnonzero(stop_mask & (~accepted_mask))
                    for slot in rejected_slots.tolist():
                        data_idx = int(active_idxs[slot])
                        dataset.memory_buffer[data_idx] = self._new_empty_entry(data_idx)

                    for slot in stopped_slots.tolist():
                        if slot in rejected_slots:
                            refill_slots.append(slot)
                            continue

                        keep_mask[slot] = False
                        if next_data_idx < n_samples:
                            active_idxs[slot] = next_data_idx
                            dataset.memory_buffer[next_data_idx] = self._new_empty_entry(next_data_idx)
                            refill_slots.append(slot)
                            keep_mask[slot] = True
                            next_data_idx += 1

                if refill_slots:
                    refill_arr = np.asarray(refill_slots, dtype=int)
                    x0_new = self.sampler.sample_x0(refill_arr.size)
                    current_x0[refill_arr, :] = x0_new
                    current_x_flat[refill_arr, :] = np.tile(x0_new, (1, self.mpc_cfg.N + 1))
                    eps_hits[refill_arr] = 0

                active_idxs = active_idxs[keep_mask]
                current_x0[:active_idxs.shape[0], :] = current_x0[:n_active, :][keep_mask, :]
                current_x_flat[:active_idxs.shape[0], :] = current_x_flat[:n_active, :][keep_mask, :]
                eps_hits[:active_idxs.shape[0]] = eps_hits[:n_active][keep_mask]

                pbar.set_postfix_str(f"feasible: {self.feasibility_percentage:.1f}%")

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