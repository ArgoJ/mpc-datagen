import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go

from ..mpc_data import MPCDataset
from .utils import (
    COLORS,
    PairPlotResult,
    _add_visibility_toggle,
    _apply_pair_layout,
    _combine_pair_limits,
    _extract_trajectory_v,
    _infer_pair_limits,
    _pair_limits_from_resolved,
    _resolve_indices,
    _resolve_labels,
    _resolve_limits,
    _resolve_num_states,
    _save_pair_figures,
    _state_index_pairs,
    _to_latex,
)

__logger__ = logging.getLogger(__name__)


def _evaluate_lyapunov(
    lyapunov_func: Callable[[NDArray], NDArray],
    points: NDArray,
) -> NDArray:
    """Evaluate Lyapunov function on a batch with single-point fallback.

    Parameters
    ----------
    lyapunov_func : Callable[[NDArray], NDArray]
        Function computing Lyapunov values.
    points : NDArray
        State points array of shape (N, nx).

    Returns
    -------
    NDArray
        1D array of evaluated Lyapunov values of shape (N,).
    """
    try:
        values = lyapunov_func(points)
    except Exception:
        values = np.array([lyapunov_func(s) for s in points])
    return np.asarray(values, dtype=float).reshape(-1)


def _find_level_ray_intersection(
    lyapunov_func: Callable[[NDArray], NDArray],
    c_level: float,
    direction: NDArray,
    *,
    base_value: float | None = None,
    initial_radius: float = 1.0,
    growth_factor: float = 2.0,
    max_radius: float = 1e6,
    max_bisection_steps: int = 60,
) -> float:
    """Approximate where the level set ``V(x)=c`` intersects a ray from the origin."""
    direction_vec = np.asarray(direction, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(direction_vec))
    if norm <= 0.0:
        raise ValueError("direction must be non-zero.")
    direction_vec = direction_vec / norm

    if base_value is None:
        base_point = np.zeros((1, direction_vec.size), dtype=float)
        base_value = float(_evaluate_lyapunov(lyapunov_func, base_point)[0])

    if base_value > c_level:
        raise ValueError(
            f"Cannot infer ROA limits from c_level={c_level:.6g} because V(0)={base_value:.6g} > c_level."
        )
    if base_value == c_level:
        return 0.0

    low = 0.0
    high = float(initial_radius)
    point = np.zeros((1, direction_vec.size), dtype=float)

    while True:
        point[0, :] = high * direction_vec
        high_value = float(_evaluate_lyapunov(lyapunov_func, point)[0])

        if not np.isfinite(high_value) or high_value > c_level:
            break

        low = high
        high *= growth_factor
        if high > max_radius:
            raise ValueError(
                f"Could not bracket the ROA boundary before reaching max_radius={max_radius:.6g}."
            )

    for _ in range(max_bisection_steps):
        mid = 0.5 * (low + high)
        point[0, :] = mid * direction_vec
        mid_value = float(_evaluate_lyapunov(lyapunov_func, point)[0])
        if np.isfinite(mid_value) and mid_value <= c_level:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def _infer_roa_pair_limits(
    lyapunov_func: Callable[[NDArray], NDArray],
    c_level: float,
    num_states: int,
    idx_x: int,
    idx_y: int,
    *,
    num_directions: int = 181,
    pad_ratio: float = 0.1,
    min_pad: float = 1e-12,
) -> list[tuple[float, float]]:
    """Infer ROA plotting limits from the level set contour in the plotted 2D plane."""
    if num_directions < 8:
        raise ValueError("num_directions must be at least 8.")

    base_point = np.zeros((1, num_states), dtype=float)
    base_value = float(_evaluate_lyapunov(lyapunov_func, base_point)[0])

    angles = np.linspace(0.0, 2.0 * np.pi, num_directions, endpoint=False)
    boundary_points = np.zeros((num_directions, 2), dtype=float)

    for angle_idx, angle in enumerate(angles):
        direction = np.zeros(num_states, dtype=float)
        direction[idx_x] = np.cos(angle)
        direction[idx_y] = np.sin(angle)
        radius = _find_level_ray_intersection(
            lyapunov_func,
            c_level,
            direction,
            base_value=base_value,
        )
        boundary_points[angle_idx, 0] = radius * direction[idx_x]
        boundary_points[angle_idx, 1] = radius * direction[idx_y]

    return _infer_pair_limits(
        boundary_points[:, 0],
        boundary_points[:, 1],
        pad_ratio=pad_ratio,
        min_pad=min_pad,
    )


def _make_pair_grid(
    pair_limits: list[tuple[float, float]],
    resolution: int,
    num_states: int,
    idx_x: int,
    idx_y: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Create a 2D plotting grid embedded in full state dimension."""
    x_vec = np.linspace(pair_limits[0][0], pair_limits[0][1], resolution)
    y_vec = np.linspace(pair_limits[1][0], pair_limits[1][1], resolution)
    X, Y = np.meshgrid(x_vec, y_vec)

    full_points = np.zeros((X.size, num_states), dtype=float)
    full_points[:, idx_x] = X.ravel()
    full_points[:, idx_y] = Y.ravel()
    return x_vec, y_vec, X, Y, full_points


def _interpolate_dataset_v_grid(
    x_pts: NDArray,
    y_pts: NDArray,
    v_pts: NDArray,
    X: NDArray,
    Y: NDArray,
    fill_nearest: bool = False,
) -> NDArray | None:
    """Interpolate dataset optimal value function onto a 2D grid (X, Y)."""
    if len(x_pts) < 3:
        return None

    try:
        from scipy.interpolate import griddata

        # Subsample if dataset is very large to keep Delaunay triangulation fast (<0.2s)
        if len(x_pts) > 25000:
            rng = np.random.default_rng(42)
            sub_idx = rng.choice(len(x_pts), size=25000, replace=False)
            pts_interp = np.column_stack([x_pts[sub_idx], y_pts[sub_idx]])
            v_interp = v_pts[sub_idx]
        else:
            pts_interp = np.column_stack([x_pts, y_pts])
            v_interp = v_pts

        Z_linear = griddata(pts_interp, v_interp, (X, Y), method='linear')
        nan_mask = np.isnan(Z_linear)
        if np.any(nan_mask) and fill_nearest:
            Z_nearest = griddata(pts_interp, v_interp, (X, Y), method='nearest')
            Z_fill = np.where(nan_mask, Z_nearest, Z_linear)
            try:
                from scipy.ndimage import gaussian_filter
                # Smooth extrapolated region so contours remain continuous without staircase steps
                Z_smooth = gaussian_filter(Z_fill, sigma=2.0)
                return np.where(nan_mask, Z_smooth, Z_linear)
            except Exception:
                return Z_fill
        return Z_linear
    except Exception as e:
        __logger__.warning(f"Could not interpolate dataset value function on grid: {e}")
        return None


def _extract_dataset_plot_data(
    dataset: MPCDataset,
    use_solver_v: bool = False,
    extract_failed: bool = True,
    max_entries: int | None = 5000,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """Single-pass extraction of all states, values, and failed states from dataset."""
    total_len = len(dataset)
    if max_entries is not None and total_len > max_entries:
        indices = np.linspace(0, total_len - 1, max_entries, dtype=int)
        entries_to_read = [dataset[int(i)] for i in indices]
    else:
        entries_to_read = list(dataset)

    state_chunks: list[NDArray] = []
    v_chunks: list[NDArray] = []
    fail_chunks: list[NDArray] = []

    for entry in entries_to_read:
        traj = entry.trajectory
        if traj.states is None or len(traj.states) == 0:
            continue

        states = np.asarray(traj.states, dtype=float)

        if use_solver_v:
            v_opt = (
                np.asarray(traj.V_solver, dtype=float).reshape(-1)
                if traj.V_solver is not None and traj.V_solver.size > 0
                else _extract_trajectory_v(traj, entry)
            )
        else:
            v_opt = _extract_trajectory_v(traj, entry)

        if v_opt is not None and v_opt.size > 0:
            n_pts = min(len(states), len(v_opt))
            if n_pts > 0:
                state_chunks.append(states[:n_pts])
                v_chunks.append(v_opt[:n_pts])

        if extract_failed and entry.meta is not None:
            is_infeas = not entry.meta.feasible
            if not is_infeas and entry.meta.status_codes is not None:
                is_infeas = any(c != 0 for c in entry.meta.status_codes)
            if is_infeas:
                fail_chunks.append(states)

    all_states = np.vstack(state_chunks) if state_chunks else np.empty((0, 2), dtype=float)
    all_v = np.concatenate(v_chunks) if v_chunks else np.empty(0, dtype=float)
    failed_states = np.vstack(fail_chunks) if fail_chunks else None

    return all_states, all_v, failed_states


def create_lyapunov_from_dataset(
    dataset: MPCDataset,
    method: str = "linear",
    use_solver_v: bool = False,
    extrapolate: bool = False,
) -> Callable[[NDArray], NDArray]:
    """Create a callable Lyapunov function by interpolating the optimal value function from a dataset.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing solved MPC trajectories and optimal costs.
    method : str, optional
        Interpolation method ('linear', 'nearest', 'cubic'). Default is 'linear'.
    use_solver_v : bool, optional
        If True, extracts the solver objective value `traj.V_solver` directly from
        trajectories. If False, extracts `traj.V_N` (or recalculates costs). Default is False.
    extrapolate : bool, optional
        If True and `method="linear"`, falls back to nearest-neighbor for points outside
        the convex hull. If False, returns NaN outside the convex hull. Default is False.

    Returns
    -------
    Callable[[NDArray], NDArray]
        A function accepting an array of states (shape (N, nx) or (nx,)) and returning
        the interpolated optimal value function V(x).
    """
    pts_mat, vals_arr, _ = _extract_dataset_plot_data(
        dataset,
        use_solver_v=use_solver_v,
        extract_failed=False,
        max_entries=None,
    )

    if len(pts_mat) == 0:
        raise ValueError("Cannot create Lyapunov interpolator: Dataset has no valid (states, values) pairs.")

    from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

    if method == "nearest":
        interp = NearestNDInterpolator(pts_mat, vals_arr)
    else:
        interp_linear = LinearNDInterpolator(pts_mat, vals_arr, fill_value=np.nan)
        if extrapolate:
            interp_nearest = NearestNDInterpolator(pts_mat, vals_arr)

            def combined_interp(xi: NDArray) -> NDArray:
                res = interp_linear(xi)
                nan_mask = np.isnan(res)
                if np.any(nan_mask):
                    res[nan_mask] = interp_nearest(xi[nan_mask])
                return res

            interp = combined_interp
        else:
            interp = interp_linear

    def lyapunov_eval(x: NDArray) -> NDArray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
            return float(interp(x_arr)[0])
        return np.asarray(interp(x_arr), dtype=float).reshape(-1)

    return lyapunov_eval


def _add_lyapunov_landscape(
    fig: go.Figure,
    Z: NDArray,
    x_vec: NDArray,
    y_vec: NDArray,
    *,
    plot_3d: bool,
    name: str,
) -> None:
    """Add Lyapunov landscape to a figure."""
    if plot_3d:
        fig.add_trace(
            go.Surface(
                z=Z,
                x=x_vec,
                y=y_vec,
                colorscale='Viridis',
                name=_to_latex(name),
                opacity=0.8,
                showscale=True,
                showlegend=False,
            )
        )
    else:
        fig.add_trace(
            go.Contour(
                z=Z,
                x=x_vec,
                y=y_vec,
                colorscale='Viridis',
                name=_to_latex(name),
                showscale=True,
                showlegend=False,
                contours=dict(coloring='heatmap', showlabels=True),
            )
        )


def _add_trajectory_traces(
    fig: go.Figure,
    dataset: MPCDataset,
    idx_x: int,
    idx_y: int,
    *,
    plot_3d: bool,
    lyapunov_func: Callable[[NDArray], NDArray] | None,
    use_dataset_v: bool,
    use_solver_v: bool = False,
    max_trajectories: int | None = None,
) -> list[int]:
    """Add closed-loop trajectories and return trace indices for toggling."""
    indices: list[int] = []
    trajs_to_plot = dataset if max_trajectories is None else list(dataset)[:max_trajectories]
    for idx, entry in enumerate(trajs_to_plot):
        traj = entry.trajectory
        color = COLORS[idx % len(COLORS)]

        if use_dataset_v:
            if use_solver_v:
                v_opt = (
                    np.asarray(traj.V_solver, dtype=float).reshape(-1)
                    if traj.V_solver is not None and traj.V_solver.size > 0
                    else _extract_trajectory_v(traj, entry)
                )
            else:
                v_opt = _extract_trajectory_v(traj, entry)
        else:
            v_opt = None

        states_arr = np.asarray(traj.states, dtype=float)

        if plot_3d:
            if v_opt is not None and v_opt.size > 0:
                n_pts = min(states_arr.shape[0], v_opt.shape[0])
                states = states_arr[:n_pts]
                v_traj = v_opt[:n_pts]
            elif lyapunov_func is not None:
                states = states_arr
                v_traj = _evaluate_lyapunov(lyapunov_func, states)
            else:
                __logger__.warning(
                    f"Trajectory {idx+1}: Neither V_N nor lyapunov_func available for 3D plot; skipping."
                )
                continue

            x = states[:, idx_x].reshape(-1)
            y = states[:, idx_y].reshape(-1)

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=v_traj,
                    mode='lines',
                    name=_to_latex(f'Run ${idx+1}$'),
                    line=dict(color=color, width=4),
                    showlegend=False,
                )
            )
        else:
            states = states_arr
            x = states[:, idx_x].reshape(-1)
            y = states[:, idx_y].reshape(-1)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=_to_latex(f'Run ${idx+1}$'),
                    line=dict(color=color, width=2),
                    opacity=0.7,
                    showlegend=False,
                )
            )

        indices.append(len(fig.data) - 1)
    return indices


def _add_roa_boundary_trace(
    fig: go.Figure,
    x_vec: NDArray,
    y_vec: NDArray,
    Z: NDArray,
    c_level: float,
    *,
    plot_3d: bool,
) -> None:
    """Add Region of Attraction (ROA) sublevel set boundary trace."""
    if plot_3d:
        fig.add_trace(
            go.Surface(
                z=Z,
                x=x_vec,
                y=y_vec,
                showscale=False,
                opacity=0.1,
                contours=dict(
                    z=dict(
                        show=True,
                        start=c_level,
                        end=c_level,
                        size=1,
                        color='red',
                        width=4,
                        project=dict(z=True),
                    )
                ),
                name=_to_latex('ROA'),
                showlegend=False,
            )
        )
    else:
        fig.add_trace(
            go.Contour(
                z=Z,
                x=x_vec,
                y=y_vec,
                contours=dict(
                    start=c_level,
                    end=c_level,
                    size=1,
                    coloring='none',
                ),
                line=dict(color='red', width=3, dash='dash', smoothing=1.1),
                showscale=False,
                name=_to_latex(f'ROA ($c={c_level:.3g}$)'),
                showlegend=True,
            )
        )


def _add_dataset_scatter_trace(
    fig: go.Figure,
    x_pts: NDArray,
    y_pts: NDArray,
    v_pts: NDArray | None = None,
    *,
    plot_3d: bool,
) -> None:
    """Add scatter points of valid dataset samples in 2D or 3D."""
    if len(x_pts) == 0:
        return

    # Subsample if dataset is huge to prevent multi-megabyte HTML and browser lags
    if len(x_pts) > 10000:
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(len(x_pts), size=10000, replace=False)
        x_pts = x_pts[sub_idx]
        y_pts = y_pts[sub_idx]
        if v_pts is not None and len(v_pts) > 0:
            v_pts = v_pts[sub_idx]

    if plot_3d:
        fig.add_trace(
            go.Scatter3d(
                x=x_pts,
                y=y_pts,
                z=v_pts if v_pts is not None else np.zeros_like(x_pts),
                mode='markers',
                marker=dict(size=2, color=v_pts, colorscale='Viridis', opacity=0.6),
                name=_to_latex('Dataset Points'),
                showlegend=True,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_pts,
                y=y_pts,
                mode='markers',
                marker=dict(size=3, color='rgba(200, 200, 200, 0.4)', symbol='circle'),
                name=_to_latex('Dataset Points'),
                showlegend=True,
            )
        )


def _resolve_pair_limits(
    limits: list[tuple[float, float]] | None,
    dataset: MPCDataset | None,
    all_states: NDArray | None,
    roa_level: float | None,
    lyapunov_func: Callable[[NDArray], NDArray] | None,
    num_states: int,
    idx_x: int,
    idx_y: int,
) -> list[tuple[float, float]]:
    """Determine the 2D plotting bounding box for a given state pair."""
    pair_limits = _pair_limits_from_resolved(limits, idx_x, idx_y, num_states)
    if pair_limits is not None:
        return pair_limits

    # 1. State constraints from dataset if available
    if dataset is not None and len(dataset) > 0:
        cfg = dataset[0].config
        if cfg.constraints is not None and cfg.constraints.has_bx():
            lbx = getattr(cfg.constraints, 'lbx', None)
            ubx = getattr(cfg.constraints, 'ubx', None)
            if lbx is not None and ubx is not None and len(lbx) > max(idx_x, idx_y) and len(ubx) > max(idx_x, idx_y):
                return [
                    (float(lbx[idx_x]), float(ubx[idx_x])),
                    (float(lbx[idx_y]), float(ubx[idx_y])),
                ]

    # 2. Inferred limits from data points / ROA
    dataset_limits = None
    roa_limits = None

    if all_states is not None and len(all_states) > 0:
        dataset_limits = _infer_pair_limits(all_states[:, idx_x], all_states[:, idx_y])
    if roa_level is not None and roa_level > 0.0 and lyapunov_func is not None:
        roa_limits = _infer_roa_pair_limits(lyapunov_func, roa_level, num_states, idx_x, idx_y)

    combined = _combine_pair_limits(dataset_limits, roa_limits)
    if combined is not None:
        return combined

    __logger__.warning("Could not infer limits; falling back to [-1, 1]^2.")
    return [(-1.0, 1.0), (-1.0, 1.0)]


def _add_failed_states_trace(
    fig: go.Figure,
    fail_arr: NDArray,
    idx_x: int,
    idx_y: int,
    *,
    plot_3d: bool,
    lyapunov_func: Callable[[NDArray], NDArray] | None = None,
) -> None:
    """Add scatter markers for infeasible or failed states in 2D or 3D."""
    if fail_arr is None or len(fail_arr) == 0:
        return

    if plot_3d:
        if lyapunov_func is not None:
            z_fail = _evaluate_lyapunov(lyapunov_func, fail_arr)
        else:
            z_fail = np.zeros(len(fail_arr))
        fig.add_trace(
            go.Scatter3d(
                x=fail_arr[:, idx_x],
                y=fail_arr[:, idx_y],
                z=z_fail,
                mode='markers',
                marker=dict(size=4, color='rgba(255, 40, 40, 0.85)', symbol='x'),
                name=_to_latex('Infeasible / Failed States'),
                showlegend=True,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=fail_arr[:, idx_x],
                y=fail_arr[:, idx_y],
                mode='markers',
                marker=dict(size=6, color='rgba(255, 40, 40, 0.9)', symbol='x'),
                name=_to_latex('Infeasible / Failed States'),
                showlegend=True,
            )
        )


def lyapunov(
    lyapunov_func: Callable[[NDArray], NDArray] | None = None,
    dataset: MPCDataset | None = None,
    roa_level: float | None = None,
    state_indices: list[int] | None = None,
    state_labels: list[str] | None = None,
    num_states: int | None = None,
    limits: list[tuple[float, float]] | None = None,
    resolution: int = 100,
    plot_3d: bool = False,
    use_solver_v: bool = False,
    html_path: Path | str | None = None,
    use_dataset_v: bool = True,
    scatter_points: bool = False,
    scatter_failed: bool = False,
    fill_nearest: bool = False,
    max_trajectories: int | None = None,
) -> list[PairPlotResult] | None:
    """Plot Lyapunov landscapes and trajectories for all 2D state pairs.

    If more than two state indices are provided, one figure per 2D combination
    is generated. Supports both analytical/custom Lyapunov functions and direct
    MPC optimal value function V_N / V_solver plotting extracted from datasets.

    Parameters
    ----------
    lyapunov_func : Callable[[NDArray], NDArray], optional
        A function that takes a state vector and returns the Lyapunov value.
        If None and `dataset` is provided, the Lyapunov landscape is constructed
        from the optimal value function / solver cost in the dataset.
    dataset : MPCDataset, optional
        The dataset containing trajectories to plot. If None, only the
        Lyapunov landscape and optional regions are shown. Default is None.
    roa_level : float, optional
        Sublevel set level c for the Region of Attraction {x : V(x) <= c}.
    state_indices : list[int], optional
        State indices to consider. If None, all states are used and all pairwise
        combinations are plotted.
    state_labels : list[str], optional
        All labels for the plotted state dimensions. Defaults to ["State i", ...].
    num_states : int, optional
        State dimension. Inferred if None.
    limits : list of tuples, optional
        Either `[(min_x, max_x), (min_y, max_y)]` for all pairs or one tuple per
        state dimension. If None, inferred from the dataset and optional ROA.
    resolution : int, optional
        Grid resolution for the Lyapunov function contour/surface plot. Default is 100.
    plot_3d : bool, optional
        If True, plot a 3D surface and 3D trajectories. Default is False.
    use_solver_v : bool, optional
        If True, uses `traj.V_solver` directly when reading costs from dataset.
        If False, uses `traj.V_N` (or recalculates). Default is False.
    html_path : Path | str, optional
        If provided, saves the plot(s) to the specified HTML file.
    use_dataset_v : bool, optional
        If True, uses the dataset's optimal value function / solver cost for trajectory
        z-coordinates instead of evaluating `lyapunov_func`. Default is True.
    scatter_points : bool, optional
        If True, adds scatter points of all dataset (x, y, V) samples.
    scatter_failed : bool, optional
        If True, automatically extracts and displays failed/infeasible rollout states as scatter markers.
    fill_nearest : bool, optional
        If True and interpolating from dataset, fills NaNs outside convex hull with
        nearest-neighbor extrapolation. Default is False (preserves NaNs).
    max_trajectories : int, optional
        Maximum number of rollout trajectory lines to draw on the figure.
        Default is None (draws all trajectories in the dataset). Allows using the full
        dataset for dense contour estimation while drawing only a clean subset of trajectory curves.

    Returns
    -------
    list[PairPlotResult] | None
        A list of results containing the state indices, labels, and figure for each pair.
        If `html_path` is provided, the figures are saved to HTML and None is returned.
    """
    has_dataset = dataset is not None and len(dataset) > 0
    has_roa = roa_level is not None and roa_level > 0.0

    num_states = _resolve_num_states(num_states, dataset, state_labels, state_indices)
    state_indices = _resolve_indices(state_indices, num_states)
    labels_full = _resolve_labels(state_labels, num_states)
    limits = _resolve_limits(limits)

    all_states: NDArray | None = None
    all_v: NDArray | None = None
    fail_arr: NDArray | None = None

    if has_dataset:
        all_states, all_v, fail_arr = _extract_dataset_plot_data(
            dataset,
            use_solver_v=use_solver_v,
            extract_failed=scatter_failed,
            max_entries=5000,
        )

    pairs = _state_index_pairs(state_indices)
    results: list[PairPlotResult] = []

    for idx_x, idx_y in pairs:
        pair_labels = [labels_full[idx_x], labels_full[idx_y]]
        pair_limits = _resolve_pair_limits(
            limits, dataset, all_states, roa_level, lyapunov_func, num_states, idx_x, idx_y
        )

        fig = go.Figure()
        x_range, y_range, X, Y, grid_points = _make_pair_grid(
            pair_limits, resolution, num_states, idx_x, idx_y
        )

        if lyapunov_func is not None:
            landscape_name = "$V(x)$"
            Z = _evaluate_lyapunov(lyapunov_func, grid_points).reshape(X.shape)
        elif all_states is not None and all_v is not None and len(all_states) > 0:
            landscape_name = "$V_{\\mathrm{solver}}$" if use_solver_v else "$V_N(x)$"
            Z = _interpolate_dataset_v_grid(
                all_states[:, idx_x],
                all_states[:, idx_y],
                all_v,
                X,
                Y,
                fill_nearest=fill_nearest,
            )
        else:
            landscape_name = "$V_N(x)$"
            Z = None

        if Z is not None:
            _add_lyapunov_landscape(
                fig,
                Z,
                x_range,
                y_range,
                plot_3d=plot_3d,
                name=landscape_name,
            )

        if has_roa and Z is not None:
            _add_roa_boundary_trace(
                fig,
                x_range,
                y_range,
                Z,
                roa_level,
                plot_3d=plot_3d,
            )

        trajectory_indices = []
        if has_dataset:
            trajectory_indices = _add_trajectory_traces(
                fig,
                dataset,
                idx_x,
                idx_y,
                plot_3d=plot_3d,
                lyapunov_func=lyapunov_func,
                use_dataset_v=use_dataset_v,
                use_solver_v=use_solver_v,
                max_trajectories=max_trajectories,
            )

        if scatter_points and all_states is not None and len(all_states) > 0:
            _add_dataset_scatter_trace(
                fig,
                all_states[:, idx_x],
                all_states[:, idx_y],
                all_v,
                plot_3d=plot_3d,
            )

        if fail_arr is not None and len(fail_arr) > 0:
            _add_failed_states_trace(
                fig,
                fail_arr,
                idx_x,
                idx_y,
                plot_3d=plot_3d,
                lyapunov_func=lyapunov_func,
            )

        roa_string = f" with ROA level ${roa_level:.3g}$" if (has_roa and Z is not None) else ""
        title_base = "MPC Value Function" if lyapunov_func is None else "Lyapunov Landscape"
        _apply_pair_layout(
            fig,
            plot_3d=plot_3d,
            pair_labels=pair_labels,
            pair_limits=pair_limits,
            title_2d=_to_latex(f"{title_base} ({pair_labels[0]} vs {pair_labels[1]}){roa_string}"),
            title_3d=_to_latex(f"{title_base} 3D ({pair_labels[0]} vs {pair_labels[1]}){roa_string}"),
            zaxis_title=_to_latex(landscape_name),
        )

        _add_visibility_toggle(fig, trajectory_indices, label="Trajectories")

        results.append(PairPlotResult(
            idx_x=idx_x,
            idx_y=idx_y,
            label_x=labels_full[idx_x],
            label_y=labels_full[idx_y],
            figure=fig,
        ))

    if html_path is not None:
        _save_pair_figures(results, html_path, kind="Lyapunov")
        return None

    return results
