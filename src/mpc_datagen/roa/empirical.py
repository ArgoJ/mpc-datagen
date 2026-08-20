import logging
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError, cKDTree
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)

from ..mpc_data import MPCData, MPCDataset
from ..plots.utils import _extract_trajectory_v
from .reports import (
    TrajectoryStatus,
    SampledPoint,
    EmpiricalROAReport,
)

__logger__ = logging.getLogger(__name__)


class EmpiricalROAEstimator:
    r"""Estimates the empirical Region of Attraction (ROA) from MPC rollout transitions.

    The empirical sublevel set c_empirical is chosen as the maximal verified Lyapunov
    descent level below all observed failure boundaries and domain bounds:

        c_empirical = max { V(x_i) : x_i is feasible & decreasing,
                                     V(x_i) < min(failure_levels) }

    A failure is:
    - MPC solver infeasibility (status code != 0 or meta.feasible is False),
    - State / input constraint violation,
    - Evaluated non-decreasing Lyapunov transition (V(x_{t+1}) > V(x_t)).

    For infeasible states without a finite V(x), the cost of the nearest verified
    successful state (via cKDTree) defines the local boundary level.
    """

    def __init__(
        self,
        dataset: MPCDataset,
        eps_descent: float = 1e-4,
    ):
        if not isinstance(dataset, MPCDataset):
            raise TypeError(f"Expected MPCDataset, got {type(dataset).__name__}.")

        if len(dataset) == 0:
            raise ValueError("Dataset is empty. Cannot initialize EmpiricalROAEstimator.")

        self.dataset = dataset
        self.eps_descent = float(eps_descent)
        self._ref_entry = dataset[0]
        self.nx = int(self._ref_entry.config.nx)

        self._sampled_points: list[SampledPoint] = []
        self._roa_points: list[SampledPoint] = []
        self._last_report: EmpiricalROAReport | None = None

    def classify_trajectory(
        self,
        entry: MPCData,
        index: int = 0,
    ) -> list[SampledPoint]:
        """Classify all transitions of a single closed-loop rollout."""
        traj = entry.trajectory
        cfg = entry.config
        meta = entry.meta

        if traj.states is None or len(traj.states) < 2:
            return [
                SampledPoint(
                    trajectory_index=index,
                    step_index=0,
                    x=np.zeros(self.nx, dtype=float),
                    x_next=None,
                    V=None,
                    V_next=None,
                    is_feasible=False,
                    is_decreased=False,
                    status=TrajectoryStatus.INVALID_DATA,
                )
            ]

        states = np.asarray(traj.states, dtype=float)
        if states.ndim != 2 or states.shape[1] != self.nx or not np.all(np.isfinite(states)):
            return [
                SampledPoint(
                    trajectory_index=index,
                    step_index=0,
                    x=states[0].copy() if len(states) > 0 else np.zeros(self.nx, dtype=float),
                    x_next=states[1].copy() if len(states) > 1 else None,
                    V=None,
                    V_next=None,
                    is_feasible=False,
                    is_decreased=False,
                    status=TrajectoryStatus.INVALID_DATA,
                )
            ]

        n_steps = len(states) - 1
        x_t = states[:-1]
        x_next = states[1:]

        # Extract optimal value sequence V(x_t)
        v_opt = _extract_trajectory_v(traj, entry, use_solver_fallback=True)

        inputs = (
            np.asarray(traj.inputs, dtype=float)
            if traj.inputs is not None and len(traj.inputs) > 0
            else None
        )

        V_t = np.full(n_steps, np.nan, dtype=float)
        V_next = np.full(n_steps, np.nan, dtype=float)

        if v_opt is not None:
            v_opt_arr = np.asarray(v_opt, dtype=float).reshape(-1)
            n_curr = min(n_steps, len(v_opt_arr))
            if n_curr > 0:
                V_t[:n_curr] = v_opt_arr[:n_curr]
            n_nxt = min(n_steps, max(0, len(v_opt_arr) - 1))
            if n_nxt > 0:
                V_next[:n_nxt] = v_opt_arr[1 : n_nxt + 1]

        # Solver status check
        solver_ok = np.ones(n_steps, dtype=bool)
        if meta is not None and meta.status_codes is not None:
            status_arr = np.asarray(meta.status_codes).reshape(-1)
            n_status = min(n_steps, len(status_arr))
            if n_status > 0:
                solver_ok[:n_status] = status_arr[:n_status] == 0
        elif meta is not None and not meta.feasible:
            solver_ok[:] = False

        # Constraint check
        constraints_ok = np.ones(n_steps, dtype=bool)
        cons = cfg.constraints if cfg is not None else None
        if cons is not None:
            if cons.has_bx():
                lbx = np.asarray(cons.lbx, dtype=float) - 1e-4
                ubx = np.asarray(cons.ubx, dtype=float) + 1e-4
                cx_t = (x_t < lbx) | (x_t > ubx)
                cx_next = (x_next < lbx) | (x_next > ubx)
                constraints_ok &= ~np.any(cx_t, axis=1)
                constraints_ok &= ~np.any(cx_next, axis=1)

            if cons.has_bu():
                if inputs is None:
                    constraints_ok[:] = False
                else:
                    lbu = np.asarray(cons.lbu, dtype=float) - 1e-4
                    ubu = np.asarray(cons.ubu, dtype=float) + 1e-4
                    n_u = min(n_steps, len(inputs))
                    if n_u < n_steps:
                        constraints_ok[n_u:] = False
                    if n_u > 0:
                        u_arr = np.asarray(inputs[:n_u], dtype=float)
                        cu = (u_arr < lbu) | (u_arr > ubu)
                        constraints_ok[:n_u] &= ~np.any(cu, axis=1)

        is_feas = solver_ok & constraints_ok
        has_both_v = np.isfinite(V_t) & np.isfinite(V_next)

        is_dec = np.zeros(n_steps, dtype=bool)
        evaluable = is_feas & has_both_v
        is_dec[evaluable] = V_next[evaluable] <= V_t[evaluable] + self.eps_descent

        # Fully vectorized integer status array
        status_arr = np.full(n_steps, int(TrajectoryStatus.FEASIBLE_DECREASED), dtype=np.int32)
        status_arr[~is_dec] = int(TrajectoryStatus.FEASIBLE_INCREASED)
        status_arr[~has_both_v] = int(TrajectoryStatus.INVALID_DATA)
        status_arr[~constraints_ok] = int(TrajectoryStatus.CONSTRAINT_VIOLATED)
        status_arr[~solver_ok] = int(TrajectoryStatus.INFEASIBLE)

        vt_list = [float(v) if np.isfinite(v) else None for v in V_t]
        vnext_list = [float(v) if np.isfinite(v) else None for v in V_next]

        return [
            SampledPoint(
                trajectory_index=index,
                step_index=t,
                x=x_t[t],
                x_next=x_next[t],
                V=vt_list[t],
                V_next=vnext_list[t],
                is_feasible=bool(is_feas[t]),
                is_decreased=bool(is_dec[t]),
                status=TrajectoryStatus(int(status_arr[t])),
            )
            for t in range(n_steps)
        ]

    def estimate(self, show_progress: bool = True) -> EmpiricalROAReport:
        """Evaluate all rollouts and estimate the empirical ROA."""
        self._sampled_points = []
        self._roa_points = []

        n_trajs = len(self.dataset)
        verified_count = 0

        if show_progress:
            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("verified steps: {task.fields[verified]:d}/{task.fields[total_points]:d}"),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(
                    "Empirical ROA Estimation",
                    total=n_trajs,
                    verified=0,
                    total_points=0,
                )
                for idx, entry in enumerate(self.dataset):
                    pts = self.classify_trajectory(entry, index=idx)
                    self._sampled_points.extend(pts)
                    verified_count += sum(p.is_feasible and p.is_decreased and p.V is not None for p in pts)
                    progress.update(task, advance=1, verified=verified_count, total_points=len(self._sampled_points))
        else:
            for idx, entry in enumerate(self.dataset):
                pts = self.classify_trajectory(entry, index=idx)
                self._sampled_points.extend(pts)

        total_transitions = len(self._sampled_points)

        successful_points = [
            p for p in self._sampled_points
            if p.is_feasible and p.is_decreased and p.V is not None and np.isfinite(p.V)
        ]

        failure_points = [
            p for p in self._sampled_points
            if (not p.is_feasible) or (p.V is not None and p.V_next is not None and not p.is_decreased)
        ]

        c_empirical, mapped_count = self._compute_empirical_level_set(
            successful_points=successful_points,
            failure_points=failure_points,
        )

        if c_empirical is not None:
            self._roa_points = [
                p for p in successful_points
                if float(p.V) <= c_empirical + self.eps_descent
            ]

        state_bounds_empirical: dict[int, tuple[float, float]] = {}
        if self._roa_points:
            x_roa = np.asarray([p.x for p in self._roa_points], dtype=float)
            for dim in range(self.nx):
                state_bounds_empirical[dim] = (float(np.min(x_roa[:, dim])), float(np.max(x_roa[:, dim])))

        hull_vol, hull_verts = self._compute_convex_hull(self._roa_points)

        num_decreased = len(successful_points)
        num_feasible = sum(1 for p in self._sampled_points if p.is_feasible)
        num_failed = len(failure_points)

        descent_rate = 100.0 * num_decreased / max(1, total_transitions)
        feasibility_rate = 100.0 * num_feasible / max(1, total_transitions)
        is_valid = c_empirical is not None and len(self._roa_points) > 0

        c_str = f", c_emp={c_empirical:.4f}" if c_empirical is not None else ""
        hull_str = f", hull_vol={hull_vol:.4f}" if hull_vol is not None else ""

        message = (
            f"{'PASS' if is_valid else 'NO_EMPIRICAL_SUBLEVEL_SET'}: "
            f"{num_decreased}/{total_transitions} Lyapunov-decreasing steps ({descent_rate:.1f}%), "
            f"{num_feasible}/{total_transitions} feasible{c_str}{hull_str}."
        )

        report = EmpiricalROAReport(
            method="Empirical ROA Estimation",
            is_valid=is_valid,
            total_trajectories=n_trajs,
            total_transitions=total_transitions,
            num_feasible=num_feasible,
            num_decreased=num_decreased,
            num_failed=num_failed,
            descent_rate=descent_rate,
            feasibility_rate=feasibility_rate,
            c_empirical=c_empirical,
            state_bounds_empirical=state_bounds_empirical,
            convex_hull_volume=hull_vol,
            convex_hull_vertices=hull_verts,
            sampled_points=self._sampled_points,
            message=message,
        )

        self._last_report = report
        return report

    def _compute_empirical_level_set(
        self,
        successful_points: list[SampledPoint],
        failure_points: list[SampledPoint],
    ) -> tuple[float | None, int]:
        r"""Find maximal level set c_empirical such that all verified points with V(x) <= c_empirical are safe."""
        if not successful_points:
            __logger__.warning("No verified feasible decreasing transitions found.")
            return None, 0

        success_coords = np.asarray([p.x for p in successful_points], dtype=float)
        success_values = np.asarray([float(p.V) for p in successful_points], dtype=float)

        tree = cKDTree(success_coords)

        finite_v_failures = [float(p.V) for p in failure_points if p.V is not None and np.isfinite(p.V)]
        infeas_missing_v = [p for p in failure_points if (p.V is None or not np.isfinite(p.V)) and not p.is_feasible]

        failure_levels: list[float] = list(finite_v_failures)
        mapped_count = len(infeas_missing_v)

        # Batch KDTree query for all missing-V infeasible points at once
        if infeas_missing_v:
            query_pts = np.asarray([p.x for p in infeas_missing_v], dtype=float)
            _, nearest_indices = tree.query(query_pts, k=1)
            nearest_vals = success_values[nearest_indices]
            failure_levels.extend(nearest_vals.tolist())
            for i, p in enumerate(infeas_missing_v):
                p.V = float(nearest_vals[i])

        sorted_success = np.sort(success_values)

        if not failure_levels:
            c_emp = float(sorted_success[-1])
        else:
            first_fail = float(np.min(failure_levels))
            safe_values = sorted_success[sorted_success < first_fail]
            if safe_values.size > 0:
                c_emp = float(safe_values[-1])
            else:
                c_emp = 0.0

        if c_emp is not None and infeas_missing_v:
            for p in infeas_missing_v:
                p.V = min(c_emp, float(p.V))

        return c_emp, mapped_count

    def _compute_convex_hull(
        self,
        roa_points: list[SampledPoint],
    ) -> tuple[float | None, NDArray | None]:
        """Compute convex hull of verified ROA states."""
        if self.nx < 2 or len(roa_points) <= self.nx:
            return None, None

        x_roa = np.asarray([p.x for p in roa_points], dtype=float)
        try:
            hull = ConvexHull(x_roa)
            return float(hull.volume), x_roa[hull.vertices]
        except QhullError as e:
            __logger__.warning("Could not compute ConvexHull: %s", e)
            return None, None

    def get_empirical_roa_states(self) -> NDArray:
        """Return array of state coordinates inside the verified empirical ROA."""
        if not self._roa_points:
            return np.empty((0, self.nx), dtype=float)
        return np.asarray([p.x for p in self._roa_points], dtype=float)

    def get_successful_states(self) -> NDArray:
        """Return array of states for all verified decreasing transitions."""
        pts = [p.x for p in self._sampled_points if p.is_feasible and p.is_decreased and p.V is not None]
        if not pts:
            return np.empty((0, self.nx), dtype=float)
        return np.asarray(pts, dtype=float)

    def get_failed_states(self) -> NDArray:
        """Return array of states for all infeasible, constraint-violating, or non-decreasing transitions."""
        pts = [
            p.x for p in self._sampled_points
            if (not p.is_feasible) or (p.V is not None and p.V_next is not None and not p.is_decreased)
        ]
        if not pts:
            return np.empty((0, self.nx), dtype=float)
        return np.asarray(pts, dtype=float)

    def get_empirical_roa_initial_states(self) -> NDArray:
        """Return initial states x0 inside the verified empirical ROA."""
        pts = [p.x for p in self._roa_points if p.step_index == 0]
        if not pts:
            return np.empty((0, self.nx), dtype=float)
        return np.asarray(pts, dtype=float)

    def get_successful_initial_states(self) -> NDArray:
        """Return initial states x0 with verified Lyapunov descent."""
        pts = [p.x for p in self._sampled_points if p.step_index == 0 and p.is_feasible and p.is_decreased]
        if not pts:
            return np.empty((0, self.nx), dtype=float)
        return np.asarray(pts, dtype=float)

    def get_failed_initial_states(self) -> NDArray:
        """Return initial states x0 with failed transitions."""
        pts = [
            p.x for p in self._sampled_points
            if p.step_index == 0 and ((not p.is_feasible) or (p.V is not None and p.V_next is not None and not p.is_decreased))
        ]
        if not pts:
            return np.empty((0, self.nx), dtype=float)
        return np.asarray(pts, dtype=float)

    def get_sampled_points(self) -> list[SampledPoint]:
        """Return all classified transition points."""
        return self._sampled_points