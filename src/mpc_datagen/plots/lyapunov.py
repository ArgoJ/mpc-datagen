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
    _extract_dataset_state_value_pairs,
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
    dataset: MPCDataset,
    idx_x: int,
    idx_y: int,
    X: NDArray,
    Y: NDArray,
) -> NDArray | None:
    """Interpolate optimal value function V_N from dataset points onto a 2D meshgrid.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories.
    idx_x : int
        State index for x-axis.
    idx_y : int
        State index for y-axis.
    X : NDArray
        Meshgrid X coordinates.
    Y : NDArray
        Meshgrid Y coordinates.

    Returns
    -------
    NDArray | None
        2D array of interpolated V_N values matching X.shape, or None if interpolation is not possible.
    """
    x_pts, y_pts, v_pts = _extract_dataset_state_value_pairs(dataset, idx_x, idx_y)
    if len(x_pts) < 3:
        return None

    try:
        from scipy.interpolate import griddata
        points = np.column_stack([x_pts, y_pts])
        # First linear interpolation, fill NaNs with nearest neighbor
        Z_linear = griddata(points, v_pts, (X, Y), method='linear')
        Z_nearest = griddata(points, v_pts, (X, Y), method='nearest')
        nan_mask = np.isnan(Z_linear)
        Z = np.where(nan_mask, Z_nearest, Z_linear)
        return Z
    except Exception as e:
        __logger__.warning(f"Could not interpolate dataset value function on grid: {e}")
        return None


def create_lyapunov_from_dataset(
    dataset: MPCDataset,
    method: str = "linear",
) -> Callable[[NDArray], NDArray]:
    """Create a callable Lyapunov function by interpolating the optimal value function V_N from a dataset.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing solved MPC trajectories and optimal costs V_N.
    method : str, optional
        Interpolation method ('linear', 'nearest', 'cubic'). Default is 'linear'.

    Returns
    -------
    Callable[[NDArray], NDArray]
        A function accepting an array of states (shape (N, nx) or (nx,)) and returning
        the interpolated optimal value function V_N(x).
    """
    all_states: list[NDArray] = []
    all_v: list[float] = []

    for entry in dataset:
        traj = entry.trajectory
        v_opt = _extract_trajectory_v(traj, entry)
        if v_opt is None:
            continue

        states = np.asarray(traj.states, dtype=float)
        n_pts = min(states.shape[0], v_opt.shape[0])
        if n_pts <= 0:
            continue

        all_states.append(states[:n_pts])
        all_v.extend(v_opt[:n_pts].tolist())

    if not all_states:
        raise ValueError("Cannot create Lyapunov interpolator: Dataset has no valid (states, V_N) pairs.")

    pts_mat = np.vstack(all_states)
    vals_arr = np.asarray(all_v, dtype=float)

    from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

    if method == "nearest":
        interp = NearestNDInterpolator(pts_mat, vals_arr)
    else:
        interp_linear = LinearNDInterpolator(pts_mat, vals_arr)
        interp_nearest = NearestNDInterpolator(pts_mat, vals_arr)

        def combined_interp(xi: NDArray) -> NDArray:
            res = interp_linear(xi)
            nan_mask = np.isnan(res)
            if np.any(nan_mask):
                res[nan_mask] = interp_nearest(xi[nan_mask])
            return res

        interp = combined_interp

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
) -> list[int]:
    """Add closed-loop trajectories and return trace indices for toggling."""
    indices: list[int] = []
    for idx, entry in enumerate(dataset):
        traj = entry.trajectory
        color = COLORS[idx % len(COLORS)]

        v_opt = _extract_trajectory_v(traj, entry) if use_dataset_v else None
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
                    type='constraint',
                    operation='<=',
                    value=c_level,
                ),
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='red', width=3, dash='dash', smoothing=1.1),
                showscale=False,
                name=_to_latex('ROA'),
                showlegend=False,
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
    html_path: Path | str | None = None,
    use_dataset_v: bool = True,
    scatter_points: bool = False,
) -> list[PairPlotResult] | None:
    """Plot Lyapunov landscapes and trajectories for all 2D state pairs.

    If more than two state indices are provided, one figure per 2D combination
    is generated. Supports both analytical/custom Lyapunov functions and direct
    MPC optimal value function V_N plotting extracted from datasets.

    Parameters
    ----------
    lyapunov_func : Callable[[NDArray], NDArray], optional
        A function that takes a state vector and returns the Lyapunov value.
        If None and `dataset` is provided, the Lyapunov landscape is constructed
        from the optimal value function V_N in the dataset.
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
    html_path : Path | str, optional
        If provided, saves the plot(s) to the specified HTML file.
    use_dataset_v : bool, optional
        If True, uses the dataset's optimal value function V_N for trajectory
        z-coordinates instead of evaluating `lyapunov_func`. Default is True.
    scatter_points : bool, optional
        If True and 3D is active, adds 3D scatter points of all dataset (x, y, V_N) samples.

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
    all_states = np.vstack([d.trajectory.states for d in dataset]) if has_dataset else None

    pairs = _state_index_pairs(state_indices)
    results: list[PairPlotResult] = []

    for idx_x, idx_y in pairs:
        pair_labels = [labels_full[idx_x], labels_full[idx_y]]
        pair_limits = _pair_limits_from_resolved(limits, idx_x, idx_y, num_states)

        if pair_limits is None:
            dataset_limits = None
            roa_limits = None

            if all_states is not None:
                dataset_limits = _infer_pair_limits(
                    all_states[:, idx_x],
                    all_states[:, idx_y],
                )
            if has_roa and lyapunov_func is not None:
                roa_limits = _infer_roa_pair_limits(
                    lyapunov_func,
                    roa_level,
                    num_states,
                    idx_x,
                    idx_y,
                )

            pair_limits = _combine_pair_limits(dataset_limits, roa_limits)

        if pair_limits is None:
            __logger__.warning(
                "Could not infer limits without dataset or lyapunov_func. Falling back to [-1, 1]^2."
            )
            pair_limits = [(-1.0, 1.0), (-1.0, 1.0)]

        x_range, y_range, X, Y, grid_points = _make_pair_grid(
            pair_limits, resolution, num_states, idx_x, idx_y
        )

        Z = None
        if lyapunov_func is not None:
            Z_flat = _evaluate_lyapunov(lyapunov_func, grid_points)
            Z = Z_flat.reshape(X.shape)
        elif has_dataset:
            Z = _interpolate_dataset_v_grid(dataset, idx_x, idx_y, X, Y)

        fig = go.Figure()

        landscape_name = "$V_N(x)$" if lyapunov_func is None else "$V(x)$"
        if Z is not None:
            _add_lyapunov_landscape(
                fig,
                Z,
                x_range,
                y_range,
                plot_3d=plot_3d,
                name=landscape_name,
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
            )

        if scatter_points and has_dataset and plot_3d:
            x_pts, y_pts, v_pts = _extract_dataset_state_value_pairs(dataset, idx_x, idx_y)
            if len(x_pts) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=x_pts,
                        y=y_pts,
                        z=v_pts,
                        mode='markers',
                        marker=dict(size=2, color=v_pts, colorscale='Viridis', opacity=0.6),
                        name=_to_latex('Dataset Points'),
                        showlegend=False,
                    )
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

        roa_string = f" with ROA level ${roa_level:.3g}$" if has_roa else ""
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
