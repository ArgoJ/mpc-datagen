import logging
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)

from ..mpc_data import MPCData, MPCDataset
from .reports import (
    TrajectoryStatus,
    SampledPoint,
    EmpiricalROAReport,
)

__logger__ = logging.getLogger(__name__)


class EmpiricalROAEstimator:
    """Estimates the empirical Region of Attraction (ROA) directly from an MPC dataset.

    This class analyzes closed-loop rollouts, classifies feasibility and convergence
    of each sampled initial state $x_0$, and computes empirical geometric boundaries
    (bounding box, convex hull) and value function level sets.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing generated MPC rollouts.
    eps_terminal : float, optional
        Tolerance for terminal state convergence (distance to target / origin). Default is 1e-2.
    """

    def __init__(
        self,
        dataset: MPCDataset,
        eps_terminal: float = 1e-2,
    ):
        if not isinstance(dataset, MPCDataset):
            raise TypeError(f"Expected MPCDataset, got {type(dataset).__name__}.")

        self.dataset = dataset
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty. Cannot initialize EmpiricalROAEstimator.")

        self.eps_terminal = float(eps_terminal)
        self._ref_entry = dataset[0]
        self.nx = int(self._ref_entry.config.nx)

        self._sampled_points: list[SampledPoint] = []
        self._last_report: EmpiricalROAReport | None = None

    def classify_trajectory(self, entry: MPCData, index: int = 0) -> SampledPoint:
        """Classify a single MPC rollout entry.

        Parameters
        ----------
        entry : MPCData
            The MPC rollout data entry.
        index : int, optional
            Dataset index of this trajectory.

        Returns
        -------
        SampledPoint
            Structured classification and metrics for the trajectory.
        """
        traj = entry.trajectory
        cfg = entry.config

        # Check for missing or empty state arrays
        if traj.states is None or len(traj.states) == 0:
            return SampledPoint(
                index=index,
                x0=np.zeros(self.nx),
                x_terminal=np.zeros(self.nx),
                V_0=None,
                V_terminal=None,
                is_feasible=False,
                is_converged=False,
                status=TrajectoryStatus.INVALID_DATA,
            )

        states = np.asarray(traj.states, dtype=float)
        x0 = states[0]
        x_term = states[-1]

        if np.any(np.isnan(states)):
            return SampledPoint(
                index=index,
                x0=x0,
                x_terminal=x_term,
                V_0=None,
                V_terminal=None,
                is_feasible=False,
                is_converged=False,
                status=TrajectoryStatus.INVALID_DATA,
            )

        # Extract value function V_0 and V_terminal if available
        V_0: float | None = None
        V_term: float | None = None
        if traj.V_N is not None and len(traj.V_N) > 0 and np.isfinite(traj.V_N[0]):
            V_0 = float(traj.V_N[0])
            V_term = float(traj.V_N[-1]) if np.isfinite(traj.V_N[-1]) else None
        elif traj.V_solver is not None and len(traj.V_solver) > 0 and np.isfinite(traj.V_solver[0]):
            V_0 = float(traj.V_solver[0])
            V_term = float(traj.V_solver[-1]) if np.isfinite(traj.V_solver[-1]) else None

        # Check solver feasibility
        is_feas = entry.is_feasible()

        # Check state constraints along trajectory
        cons = cfg.constraints
        state_constraints_ok = True
        if cons.has_bx():
            for k in range(len(states)):
                if np.any(states[k] < cons.lbx - 1e-4) or np.any(states[k] > cons.ubx + 1e-4):
                    state_constraints_ok = False
                    break

        if not state_constraints_ok:
            return SampledPoint(
                index=index,
                x0=x0,
                x_terminal=x_term,
                V_0=V_0,
                V_terminal=V_term,
                is_feasible=is_feas,
                is_converged=False,
                status=TrajectoryStatus.CONSTRAINT_VIOLATED,
            )

        if not is_feas:
            return SampledPoint(
                index=index,
                x0=x0,
                x_terminal=x_term,
                V_0=V_0,
                V_terminal=V_term,
                is_feasible=False,
                is_converged=False,
                status=TrajectoryStatus.INFEASIBLE,
            )

        # Check convergence to origin / target
        norm_terminal = float(np.linalg.norm(x_term))
        is_converged = norm_terminal <= self.eps_terminal
        if not is_converged and V_term is not None and V_term <= self.eps_terminal:
            is_converged = True

        status = TrajectoryStatus.FEASIBLE_CONVERGED if is_converged else TrajectoryStatus.FEASIBLE_UNCONVERGED

        return SampledPoint(
            index=index,
            x0=x0,
            x_terminal=x_term,
            V_0=V_0,
            V_terminal=V_term,
            is_feasible=True,
            is_converged=is_converged,
            status=status,
        )

    def estimate(self, show_progress: bool = True) -> EmpiricalROAReport:
        """Run empirical ROA estimation across the entire dataset.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display a progress bar during evaluation. Default is True.

        Returns
        -------
        EmpiricalROAReport
            Detailed empirical ROA report with statistics and geometric bounds.
        """
        self._sampled_points = []
        n_total = len(self.dataset)

        if show_progress:
            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("converged: {task.fields[conv]:d}/{task.total:d}"),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("Empirical ROA Estimation", total=n_total, conv=0)
                for idx, entry in enumerate(self.dataset):
                    pt = self.classify_trajectory(entry, index=idx)
                    self._sampled_points.append(pt)
                    conv_count = sum(1 for p in self._sampled_points if p.is_converged)
                    progress.update(task, advance=1, conv=conv_count)
        else:
            for idx, entry in enumerate(self.dataset):
                pt = self.classify_trajectory(entry, index=idx)
                self._sampled_points.append(pt)

        # Aggregate counts
        converged_pts = [p for p in self._sampled_points if p.is_converged]
        failed_pts = [p for p in self._sampled_points if not p.is_converged]
        feasible_pts = [p for p in self._sampled_points if p.is_feasible]

        num_converged = len(converged_pts)
        num_failed = len(failed_pts)
        num_feasible = len(feasible_pts)

        convergence_rate = (num_converged / max(1, n_total)) * 100.0
        feasibility_rate = (num_feasible / max(1, n_total)) * 100.0

        # Empirical maximal level set c_empirical
        c_emp = self._compute_empirical_level_set(self._sampled_points)

        # Spatial Bounding Box of converged x0
        state_bounds_empirical: dict[int, tuple[float, float]] = {}
        if num_converged > 0:
            x0_conv = np.array([p.x0 for p in converged_pts])
            for d in range(self.nx):
                state_bounds_empirical[d] = (float(np.min(x0_conv[:, d])), float(np.max(x0_conv[:, d])))

        # Convex Hull computation for converged points
        hull_vol, hull_verts = self._compute_convex_hull(converged_pts)

        is_valid = num_converged > 0

        c_str = f", c_emp={c_emp:.4f}" if c_emp is not None else ""
        hull_str = f", hull_vol={hull_vol:.4f}" if hull_vol is not None else ""
        message = (
            f"{'PASS' if is_valid else 'NO_CONVERGENCE'}: "
            f"{num_converged}/{n_total} converged ({convergence_rate:.1f}%), "
            f"{num_feasible}/{n_total} feasible{c_str}{hull_str}."
        )

        report = EmpiricalROAReport(
            method="Empirical ROA Estimation",
            is_valid=is_valid,
            total_trajectories=n_total,
            num_feasible=num_feasible,
            num_converged=num_converged,
            num_failed=num_failed,
            convergence_rate=convergence_rate,
            feasibility_rate=feasibility_rate,
            c_empirical=c_emp,
            state_bounds_empirical=state_bounds_empirical,
            convex_hull_volume=hull_vol,
            convex_hull_vertices=hull_verts,
            sampled_points=self._sampled_points,
            message=message,
        )

        self._last_report = report
        return report

    def _compute_empirical_level_set(self, points: list[SampledPoint]) -> float | None:
        """Find the maximal level set c_emp such that all tested points with V(x0) <= c_emp converged."""
        pts_with_v = [p for p in points if p.V_0 is not None and np.isfinite(p.V_0)]
        if not pts_with_v:
            return None

        # Sort points by initial Lyapunov/cost value V(x0)
        sorted_pts = sorted(pts_with_v, key=lambda p: p.V_0)

        # Find first failure
        first_fail_V = None
        for p in sorted_pts:
            if not p.is_converged:
                first_fail_V = p.V_0
                break

        if first_fail_V is None:
            # All sampled points with V_0 converged!
            c_emp = sorted_pts[-1].V_0
        else:
            # All points with V_0 < first_fail_V converged
            conv_below = [p.V_0 for p in sorted_pts if p.V_0 < first_fail_V and p.is_converged]
            if conv_below:
                c_emp = max(conv_below)
            else:
                c_emp = 0.0

        return float(c_emp) if c_emp is not None else None

    def _compute_convex_hull(self, converged_points: list[SampledPoint]) -> tuple[float | None, NDArray | None]:
        """Compute the convex hull volume and vertices for converged initial states."""
        if len(converged_points) <= self.nx or self.nx < 2:
            return None, None

        x0_arr = np.array([p.x0 for p in converged_points], dtype=float)

        try:
            hull = ConvexHull(x0_arr)
            return float(hull.volume), x0_arr[hull.vertices]
        except QhullError as e:
            __logger__.warning(f"Could not compute ConvexHull (likely coplanar or degenerate points): {e}")
            return None, None

    def get_successful_initial_states(self) -> NDArray:
        """Return array of initial states x0 for all converged trajectories."""
        pts = [p.x0 for p in self._sampled_points if p.is_converged]
        if not pts:
            return np.empty((0, self.nx))
        return np.asarray(pts, dtype=float)

    def get_failed_initial_states(self) -> NDArray:
        """Return array of initial states x0 for all failed / unconverged trajectories."""
        pts = [p.x0 for p in self._sampled_points if not p.is_converged]
        if not pts:
            return np.empty((0, self.nx))
        return np.asarray(pts, dtype=float)

    def get_sampled_points(self) -> list[SampledPoint]:
        """Return full list of classified SampledPoint objects."""
        return self._sampled_points
