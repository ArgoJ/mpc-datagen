import numpy as np
import os
import re
import logging
import plotly.graph_objects as go

from numpy.typing import NDArray
from plotly.subplots import make_subplots
from collections.abc import Callable
from itertools import combinations
from skimage import measure

from .mpc_data import MPCDataset

__logger__ = logging.getLogger(__name__)


COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


SUMMARY_LINE_THRESHOLD = 100


def _nanpad_stack_1d(series_list: list[NDArray]) -> NDArray:
    """Stack 1D series with NaN padding to common length.

    Parameters
    ----------
    series_list : list[NDArray]
        Each entry is a 1D array of potentially different length.

    Returns
    -------
    NDArray
        Array of shape (n_series, max_len) padded with NaNs.
    """
    if len(series_list) == 0:
        return np.empty((0, 0), dtype=float)

    lengths = [int(np.asarray(s).reshape(-1).shape[0]) for s in series_list]
    max_len = int(max(lengths))
    stacked = np.full((len(series_list), max_len), np.nan, dtype=float)

    for i, s in enumerate(series_list):
        s1 = np.asarray(s, dtype=float).reshape(-1)
        if s1.size == 0:
            continue
        stacked[i, : s1.size] = s1

    return stacked


def _slug(label: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "state"


def _to_latex(label: str) -> str:
    """Convert mixed text/math input to a LaTeX-safe math-body fragment.

    Examples
    --------
    Input: ``sometext $var1$ vs $var2$``
    Output fragment: ``$\text{sometext } var1 \text{ vs } var2$``
    """
    default = r"$\text{}$"
    s = str(label).strip()
    if not s:
        return default

    parts: list[str] = []
    chunks = re.split(r"(\$[^$]*\$)", s)

    for chunk in chunks:
        if not chunk:
            continue

        if len(chunk) >= 2 and chunk.startswith("$") and chunk.endswith("$"):
            math_expr = chunk[1:-1].strip()
            if math_expr:
                parts.append(math_expr)
        else:
            escaped = (
                chunk
                .replace("\\", r"\\")
                .replace("{", r"\{")
                .replace("}", r"\}")
            )
            if escaped:
                parts.append(rf"\text{{{escaped}}}")

    if not parts:
        return default
    
    raw = " ".join(parts)
    if not raw.startswith("$"):
        raw = "$" + raw
    if not raw.endswith("$"):
        raw = raw + "$"
    return raw


def _resolve_bounds(bounds: NDArray) -> NDArray:
    """Validate and normalize state bounds array."""
    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.ndim != 2:
        raise ValueError("bounds must be a 2D array with shape (n_points, nx).")
    if int(bounds_arr.shape[1]) < 2:
        raise ValueError("bounds must contain at least 2 state dimensions.")
    return bounds_arr


def _resolve_labels(state_labels: list[str] | None, num_states: int) -> list[str]:
    """Resolve default state labels and validate provided labels."""
    if state_labels is None:
        return [f"State {i}" for i in range(num_states)]
    if len(state_labels) != num_states:
        raise ValueError(
            f"state_labels length ({len(state_labels)}) must match state dimension ({num_states})."
        )
    return state_labels

def _resolve_indices(state_indices: list[int] | None, num_states: int) -> list[int]:
    """Resolve default state indices and validate provided indices."""
    if state_indices is None:
        return list(range(num_states))
    if any(i < 0 or i >= num_states for i in state_indices):
        raise ValueError(
            f"state_indices values must be between 0 and {num_states - 1}."
        )
    return state_indices


def _resolve_limits(
    limits: list[tuple[float, float]] | None,
) -> list[tuple[float, float]] | None:
    """Validate user-provided 2D limits and normalize to float tuples."""
    if limits is None:
        return None
    if len(limits) != 2:
        raise ValueError("limits must contain exactly two axis bounds tuples.")

    resolved: list[tuple[float, float]] = []
    for i, lim in enumerate(limits):
        if len(lim) != 2:
            raise ValueError(f"limits[{i}] must contain exactly two values.")
        lo = float(lim[0])
        hi = float(lim[1])
        if lo >= hi:
            raise ValueError(f"limits[{i}] must satisfy min < max, got ({lo}, {hi}).")
        resolved.append((lo, hi))
    return resolved


def _infer_pair_limits(
    x_values: NDArray,
    y_values: NDArray,
    *,
    pad_ratio: float = 0.1,
    min_pad: float = 1e-12,
) -> list[tuple[float, float]]:
    """Infer plotting limits for one state pair from samples with padding."""
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("Cannot infer limits from empty arrays.")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    pad_x = pad_ratio * max(min_pad, x_max - x_min)
    pad_y = pad_ratio * max(min_pad, y_max - y_min)

    return [
        (x_min - pad_x, x_max + pad_x),
        (y_min - pad_y, y_max + pad_y),
    ]

def _state_index_pairs(state_indices: list[int]) -> list[tuple[int, int]]:
    if len(state_indices) < 2:
        raise ValueError("state_indices must contain at least 2 indices.")
    return list(combinations(state_indices, 2))

def _plotly_multiline(x: NDArray, axis: int=0):
    if axis == 0:
        return np.hstack([x, np.full((x.shape[0], 1), np.nan)]).flatten()
    elif axis == 1:
        return np.vstack([x, np.full((x.shape[1], 1), np.nan)]).flatten()

def _extract_roa_boundary(
        x_vec: NDArray, y_vec: NDArray, Z: NDArray, c_level: float
    ) -> tuple[NDArray, NDArray]:
    """
    Extrahiert die (x,y) Koordinaten der V(x)=c Kontur aus einem Grid.
    Funktioniert als Blackbox für Matrizen und Neuronale Netze.
    """    
    contours = measure.find_contours(Z, c_level)
    
    if not contours:
        # Falls das Level c_level nicht im Z-Grid existiert
        return np.array([]), np.array([])
        
    main_contour = max(contours, key=len)
    y_idx = main_contour[:, 0]
    x_idx = main_contour[:, 1]
    
    x_points = np.interp(x_idx, np.arange(len(x_vec)), x_vec)
    y_points = np.interp(y_idx, np.arange(len(y_vec)), y_vec)
    
    return x_points, y_points


def _evaluate_lyapunov(
    lyapunov_func: Callable[[NDArray], NDArray], points: NDArray
) -> NDArray:
    """Evaluate Lyapunov function on a batch with single-point fallback."""
    try:
        values = lyapunov_func(points)
    except Exception:
        values = np.array([lyapunov_func(s) for s in points])
    return np.asarray(values, dtype=float).reshape(-1)


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
    lyapunov_func: Callable[[NDArray], NDArray],
    use_dataset_v: bool,
) -> list[int]:
    """Add closed-loop trajectories and return trace indices for toggling."""
    indices: list[int] = []
    for idx, entry in enumerate(dataset):
        traj = entry.trajectory
        color = COLORS[idx % len(COLORS)]

        x = traj.states[:-1, idx_x].reshape(-1)
        y = traj.states[:-1, idx_y].reshape(-1)

        if plot_3d:
            v_traj = traj.V_N \
                if use_dataset_v and traj.V_N is not None and traj.V_N.size > 0 \
                else _evaluate_lyapunov(lyapunov_func, traj.states)

            if v_traj is None or v_traj.shape != x.shape:
                __logger__.warning(
                    f"Trajectory {idx+1} cost shape {None if v_traj is None else v_traj.shape} does not match state shape {x.shape}; skipping."
                )
                continue

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
    """Extract and add V(x)=c boundary trace to a figure."""
    b_x, b_y = _extract_roa_boundary(x_vec, y_vec, Z, c_level)
    if b_x.size > 0:
        b_x = np.concatenate([b_x, b_x[:1]])
        b_y = np.concatenate([b_y, b_y[:1]])

    if plot_3d:
        b_z = np.full_like(b_x, c_level)
        fig.add_trace(
            go.Scatter3d(
                x=b_x,
                y=b_y,
                z=b_z,
                mode='lines',
                line=dict(color='red', width=4),
                name=_to_latex(f'ROA'),
                showlegend=False,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=b_x,
                y=b_y,
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                name=_to_latex(f'ROA'),
                showlegend=False,
            )
        )


def _apply_pair_layout(
    fig: go.Figure,
    *,
    plot_3d: bool,
    pair_labels: list[str],
    pair_limits: list[tuple[float, float]],
    title_2d: str,
    title_3d: str,
    zaxis_title: str = "$V(x)$",
) -> None:
    """Apply standard 2D/3D layout for a state-pair figure."""
    if plot_3d:
        fig.update_layout(
            title_text=_to_latex(title_3d),
            scene=dict(
                xaxis_title=_to_latex(pair_labels[0]),
                yaxis_title=_to_latex(pair_labels[1]),
                zaxis_title=_to_latex(zaxis_title),
            ),
            width=1000,
            height=800,
            autosize=True,
            legend=dict(x=1.05, y=1),
            margin=dict(l=0, r=50, b=0, t=50),
        )
    else:
        fig.update_layout(
            title_text=_to_latex(title_2d),
            xaxis=dict(
                title=_to_latex(pair_labels[0]),
                range=[pair_limits[0][0], pair_limits[0][1]],
            ),
            yaxis=dict(
                title=_to_latex(pair_labels[1]),
                range=[pair_limits[1][0], pair_limits[1][1]],
            ),
            legend=dict(x=1.05, y=1),
            autosize=True,
            margin=dict(l=0, r=50, b=0, t=50),
        )


def _add_visibility_toggle(fig: go.Figure, trace_indices: list[int], label: str) -> None:
    """Add toggle button for a subset of traces."""
    if not trace_indices:
        return

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": True}, trace_indices],
                        args2=[{"visible": False}, trace_indices],
                        label=label,
                        method="restyle",
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.0,
                xanchor="right",
                y=-0.05,
                yanchor="top",
            ),
        ]
    )


def _save_pair_figures(
    figs: dict[tuple[int, int], go.Figure],
    html_path: str,
    labels_full: list[str],
    *,
    kind: str,
) -> None:
    """Save one or multiple pair figures to html files."""
    dir_path = os.path.dirname(html_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    if len(figs) == 1:
        next(iter(figs.values())).write_html(html_path, include_mathjax='cdn')
        __logger__.info(f"{kind} plot saved to {html_path}.")
        return

    root, ext = os.path.splitext(html_path)
    suffix = ext if ext else ".html"
    for (idx_x, idx_y), fig in figs.items():
        file_path = f"{root}_{_slug(labels_full[idx_x])}_vs_{_slug(labels_full[idx_y])}{suffix}"
        fig.write_html(file_path, include_mathjax='cdn')
    __logger__.info(f"{len(figs)} {kind} plots saved to {root}.")


def _add_summary_band(
    fig: go.Figure,
    stacked: NDArray,
    steps: NDArray | None = None,
) -> None:
    """Add max/min envelope plus mean and median lines for stacked data."""
    if stacked.size == 0:
        return

    x = np.arange(stacked.shape[1]) if steps is None else np.asarray(steps)
    y_min = np.nanmin(stacked, axis=0)
    y_max = np.nanmax(stacked, axis=0)
    y_mean = np.nanmean(stacked, axis=0)
    y_median = np.nanmedian(stacked, axis=0)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_max,
            mode='lines',
            name='max',
            line=dict(color='red', width=2),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_min,
            mode='lines',
            name='min',
            line=dict(color='red', width=2),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.3)',
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_mean,
            mode='lines',
            name='mean',
            line=dict(color='black', width=2, dash='dash'),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_median,
            mode='lines',
            name='median',
            line=dict(color='blue', width=2),
            showlegend=True,
        )
    )


def _add_zero_reference_line(fig: go.Figure, x_start: float, x_end: float) -> None:
    """Add horizontal zero reference line to a figure."""
    fig.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip',
        )
    )


def _apply_timeseries_layout(
    fig: go.Figure,
    *,
    title_text: str,
    xaxis_title: str,
    yaxis_title: str,
    margin_top: int = 120,
) -> None:
    """Apply common layout for time-series style plots."""
    fig.update_layout(
        title_text=_to_latex(title_text),
        xaxis_title=_to_latex(xaxis_title),
        yaxis_title=_to_latex(yaxis_title),
        hovermode="x unified",
        margin=dict(t=margin_top),
    )


def mpc_trajectories(
    dataset: MPCDataset,
    state_labels: list[str],
    control_labels: list[str],
    plot_predictions: bool = False,
    time_bound: float | None = None,  
    html_path: str | None = None
) -> None:
    """Plot MPC trajectories for states and controls using Plotly.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories to plot.
    state_labels : list[str]
        List of labels for each state variable.
    control_labels : list[str]
        List of labels for each control variable.
    plot_predictions : bool, optional
        If True, plot the OCP predictions at each step. Default is False.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.
    time_bound : float, optional
        If provided, limits the x-axis to the specified time range [0, time_bound].
    """
    if len(dataset) == 0:
        __logger__.warning("Dataset is empty.")
        return

    # Extract dimensions from the first trajectory
    first_traj = dataset[0].trajectory
    num_states = first_traj.states.shape[1]
    num_controls = first_traj.inputs.shape[1]

    # Create subplots
    fig = make_subplots(
        rows=num_states + num_controls, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    for i, label in enumerate(state_labels):
        fig.update_yaxes(title_text=_to_latex(label), row=i + 1, col=1)
    for i, label in enumerate(control_labels):
        fig.update_yaxes(title_text=_to_latex(label), row=num_states + i + 1, col=1)
    fig.update_xaxes(title_text=_to_latex("$t$"), row=num_states + num_controls, col=1)

    prediction_indices = []

    # Plot states
    for i in range(num_states):
        row = i + 1
        for idx in range(len(dataset)):
            traj = dataset[idx].trajectory
            color = COLORS[idx % len(COLORS)]
            
            # Main Trajectory
            fig.add_trace(
                go.Scatter(
                    x=traj.times, 
                    y=traj.states[:, i],
                    mode='lines',
                    name=_to_latex(f'Run ${idx+1}$'),
                    line=dict(color=color),
                    legendgroup=_to_latex(f'Run ${idx+1}$'),
                    showlegend=(i == 0)
                ),
                row=row, col=1
            )
            
            if plot_predictions and traj.predicted_states is not None and not np.all(np.isnan(traj.predicted_states)):
                dt = traj.times[1] - traj.times[0] if len(traj.times) > 1 else 0.1
                
                # Consolidate prediction lines into one trace with None gaps for performance
                x_lines = []
                y_lines = []
                
                for k in range(traj.predicted_states.shape[0]):
                    pred_state = traj.predicted_states[k, :, i]
                    if np.isnan(pred_state).all():
                        continue
                    
                    t_start = traj.times[k]
                    t_pred = t_start + np.arange(len(pred_state)) * dt
                    
                    x_lines.extend(t_pred)
                    x_lines.append(None)
                    y_lines.extend(pred_state)
                    y_lines.append(None)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_lines,
                        y=y_lines,
                        mode='lines',
                        line=dict(color=color, width=1),
                        opacity=0.3,
                        showlegend=False,
                        legendgroup=f'Run {idx+1}',
                        hoverinfo='skip'
                    ),
                    row=row, col=1
                )
                prediction_indices.append(len(fig.data) - 1)

    # Plot controls
    for i in range(num_controls):
        plot_idx = num_states + i
        row = plot_idx + 1
        for idx in range(len(dataset)):
            traj = dataset[idx].trajectory
            color = COLORS[idx % len(COLORS)]
            
            # Controls (Step plot)
            fig.add_trace(
                go.Scatter(
                    x=traj.times[:-1],
                    y=traj.inputs[:, i],
                    mode='lines',
                    line=dict(color=color, shape='hv'), # 'hv' for step-after behavior
                    name=_to_latex(f'Run ${idx+1}$ - {control_labels[i]}'),
                    legendgroup=_to_latex(f'Run ${idx+1}$'),
                    showlegend=False
                ),
                row=row, col=1
            )
            
            if plot_predictions and traj.predicted_inputs is not None and not np.all(np.isnan(traj.predicted_inputs)):
                dt = traj.times[1] - traj.times[0] if len(traj.times) > 1 else 0.1
                
                x_lines = []
                y_lines = []
                
                for k in range(traj.predicted_inputs.shape[0]):
                    pred_input = traj.predicted_inputs[k, :, i]
                    if np.isnan(pred_input).all():
                        continue
                    
                    t_start = traj.times[k]
                    t_pred = t_start + np.arange(len(pred_input)) * dt
                    
                    x_lines.extend(t_pred)
                    x_lines.append(None)
                    y_lines.extend(pred_input)
                    y_lines.append(None)

                fig.add_trace(
                    go.Scatter(
                        x=x_lines,
                        y=y_lines,
                        mode='lines',
                        line=dict(color=color, width=1, shape='hv'),
                        opacity=0.3,
                        showlegend=False,
                        legendgroup=_to_latex(f'Run ${idx+1}$'),
                        hoverinfo='skip'
                    ),
                    row=row, col=1
                )
                prediction_indices.append(len(fig.data) - 1)

    fig.update_layout(
        height=300 * (num_states + num_controls), 
        title_text=_to_latex("MPC Trajectories"),
        hovermode="x unified"
    )

    if time_bound is not None:
        fig.update_xaxes(range=[0, time_bound])
    
    if plot_predictions and prediction_indices:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": True}, prediction_indices],
                            args2=[{"visible": False}, prediction_indices],
                            label="Predictions",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=1.0,
                    xanchor="right",
                    y=-0.05,
                    yanchor="top"
                ),
            ]
        )
    
    if html_path is not None:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path, include_mathjax='cdn')
        __logger__.info(f"Trajectories plot saved to {html_path}.")
    else:
        fig.show()


def lyapunov(
    lyapunov_func: Callable[[NDArray], NDArray],
    dataset: MPCDataset | None = None,
    state_indices: list[int] | None = None,
    state_labels: list[str] | None = None,
    limits: list[tuple[float, float]] | None = None,
    resolution: int = 100,
    plot_3d: bool = False,
    html_path: str | None = None,
    use_dataset_v: bool = False,
):
    """Plot Lyapunov landscapes and trajectories for all 2D state pairs.

    If more than two state indices are provided, one figure per 2D combination
    is generated.

    Parameters
    ----------
    lyapunov_func : Callable[[NDArray], NDArray]
        A function that takes a state vector and returns the Lyapunov value.
    dataset : MPCDataset, optional
        The dataset containing trajectories to plot. If None, only the
        Lyapunov landscape and optional regions are shown. Default is None.
    state_indices : list[int], optional
        State indices to consider. If None, all states are used and all pairwise
        combinations are plotted.
    state_labels : list[str], optional
        All Labels for the plotted state dimensions. Defaults to ["State i", ...].
    limits : list of tuples, optional
        ((min_x, max_x), (min_y, max_y)). If None, inferred from data with padding.
    resolution : int, optional
        Grid resolution for the Lyapunov function contour plot.
    plot_3d : bool, optional
        If True, plot a 3D surface and 3D trajectories. Default is False.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.
    use_dataset_v : bool, optional
        If True, uses the dataset's value function for trajectory coloring instead of the horizon cost.
    """
    has_dataset = dataset is not None and len(dataset) > 0

    if has_dataset:
        first_traj = dataset[0].trajectory
        num_states = int(first_traj.states.shape[1])
    else:
        if state_indices is None:
            raise ValueError(
                "Without dataset, state_indices must be provided to infer state dimension."
            )
        num_states = int(
            max(max(state_indices) + 1, len(state_labels) if state_labels is not None else 0)
        )

    state_indices = _resolve_indices(state_indices, num_states)
    labels_full = _resolve_labels(state_labels, num_states)
    limits = _resolve_limits(limits)

    pairs = _state_index_pairs(state_indices)
    figs: dict[tuple[int, int], go.Figure] = {}

    for idx_x, idx_y in pairs:
        pair_labels = [labels_full[idx_x], labels_full[idx_y]]

        if limits is None:
            if has_dataset:
                all_states = np.vstack([d.trajectory.states for d in dataset])
                pair_limits = _infer_pair_limits(
                    all_states[:, idx_x],
                    all_states[:, idx_y],
                    pad_ratio=0.1,
                    min_pad=1e-12,
                )
            else:
                __logger__.warning(
                    "Could not infer limits without dataset. Falling back to [-1, 1]^2."
                )
                pair_limits = [(-1.0, 1.0), (-1.0, 1.0)]
        else:
            pair_limits = limits

        x_range, y_range, X, _, grid_points = _make_pair_grid(
            pair_limits, resolution, num_states, idx_x, idx_y
        )
        Z_flat = _evaluate_lyapunov(lyapunov_func, grid_points)
        Z = Z_flat.reshape(X.shape)

        fig = go.Figure()
        _add_lyapunov_landscape(
            fig,
            Z,
            x_range,
            y_range,
            plot_3d=plot_3d,
            name='$V(x)$',
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

        _apply_pair_layout(
            fig,
            plot_3d=plot_3d,
            pair_labels=pair_labels,
            pair_limits=pair_limits,
            title_2d=(
                _to_latex(f"Lyapunov Landscape ({pair_labels[0]} vs {pair_labels[1]})")
            ),
            title_3d=(
                _to_latex(f"Lyapunov Landscape 3D ({pair_labels[0]} vs {pair_labels[1]})")
            ),
            zaxis_title=_to_latex("$V(x)$"),
        )

        _add_visibility_toggle(fig, trajectory_indices, label="Trajectories")

        figs[(idx_x, idx_y)] = fig

    if html_path is not None:
        _save_pair_figures(figs, html_path, labels_full, kind="Lyapunov")
        return None

    if len(figs) == 1:
        return next(iter(figs.values()))
    return figs

def relaxed_dp_residual(
    dataset: MPCDataset,
    alpha: float = 1.0,
    html_path: str | None = None
) -> go.Figure | None:
    """Plot Lyapunov-style one-step descent check.

    For each trajectory entry, plots

    $$s_n(\alpha) = V_N(x_{n+1}) - V_N(x_n) + \alpha\,\ell(x_n, u_n)$$

    where $V_N$ is the MPC cost-to-go at time step $n$ (taken from the stored
    per-step value function / objective), and $\ell(x_n,u_n)$ is the *single*
    stage cost at time step $n$ along the closed-loop trajectory.

    Visual interpretation: values above 0 violate the one-step descent
    inequality $V_N(x_{n+1}) - V_N(x_n) \le -\alpha\,\ell(x_n,u_n)$.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories to plot.
    alpha : float, optional
        The relaxation factor in the descent inequality. Use `alpha=1.0` to
        visualize the strict DP inequality, or a smaller `alpha` to match an
        empirically verified decay rate.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.
    """
    if len(dataset) == 0:
        __logger__.warning("Dataset is empty.")
        return None

    fig = go.Figure()
    if 0.0 >= alpha or alpha > 1.0:
        __logger__.warning(f"alpha must be in the range (0, 1], got {alpha}. Setting alpha=1.0.")
        alpha = 1.0

    if alpha == 1.0:
        title = _to_latex(r"DP Lyapunov residual: $s_n = V_N(x_{n+1}) - V_N(x_n) + \ell(x_n,u_n)$")
    elif alpha < 1.0:
        title = _to_latex(r"Relaxed DP Lyapunov residual: $s_n(\alpha) = V_N(x_{n+1}) - V_N(x_n) + \alpha \ell(x_n,u_n)$ with $\alpha = " + f"{alpha:.3f}$")

    per_entry = []  # list of tuples (id, deltas)
    for entry in dataset:
        traj = entry.trajectory
        cost = entry.config.cost
        id = entry.meta.id

        if traj.V_N is None:
            __logger__.info(f"Entry {id} missing V_N; skipping.")
            continue

        # Ensure consistent lengths across V_N, states, and inputs.
        num_steps = min(
            int(len(traj.V_N) - 1),
            int(traj.inputs.shape[0]),
            int(traj.states.shape[0] - 1),
        )
        if num_steps <= 0:
            __logger__.info(f"Entry {id} has insufficient steps; skipping.")
            continue

        x = traj.states[:num_steps]
        u = traj.inputs[:num_steps]
        l_n = np.asarray(cost.get_stage_cost(x, u), dtype=float).reshape(-1)
        V_curr = np.asarray(traj.V_N[:num_steps], dtype=float).reshape(-1)
        V_next = np.asarray(traj.V_N[1 : num_steps + 1], dtype=float).reshape(-1)

        deltas = V_next - V_curr + float(alpha) * l_n
        per_entry.append((id, deltas))

    n_lines = len(per_entry)
    max_len = max((int(d.shape[0]) for _, d in per_entry), default=0)

    if n_lines > SUMMARY_LINE_THRESHOLD:
        __logger__.info(
            f"relaxed_dp_residual: {n_lines} lines exceed threshold {SUMMARY_LINE_THRESHOLD}; plotting summary stats."
        )
        stacked = _nanpad_stack_1d([d for _, d in per_entry])
        _add_summary_band(fig, stacked)
    else:
        for id, deltas in per_entry:
            color = COLORS[id % len(COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=np.arange(deltas.shape[0]),
                    y=deltas,
                    mode='lines',
                    name=_to_latex(f'Run ${id+1}$ - $s_n(\\alpha)$'),
                    line=dict(color=color, width=2),
                    legendgroup=_to_latex(f'Run ${id+1}$'),
                    showlegend=True,
                )
            )

    _add_zero_reference_line(fig, 0, max(1, max_len - 1))
    _apply_timeseries_layout(
        fig,
        title_text=title,
        xaxis_title=r"$n$",
        yaxis_title=r"$s_n(\alpha)$",
    )

    if html_path is not None:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path, include_mathjax='cdn')
        __logger__.info(f"Relaxed DP residual plot saved to {html_path}.")
    else:
        return fig

def cost_descent(
    dataset: MPCDataset,
    html_path: str = None,
    use_optimal_v: bool = False
) -> go.Figure | None:
    """Plot cost descent check.

    For each trajectory entry, plots

    $$\Delta V = V_{k+1} - V_k$$

    or

    $$\Delta V = V_N(x_{k+1}) - V_N(x_{k})$$

    where $V_k$ is the MPC cost-to-go at time step $k$ (taken from the stored
    per-step value function / objective).

    Visual interpretation
    ---------------------
    Values above 0 violate the one-step descent
    inequality $V(x_{k+1}) - V(x_k) \le 0$.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories to plot.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.
    use_optimal_v : bool, optional
        If True, uses the optimal value function along the predicted trajectory
    """
    if len(dataset) == 0:
        __logger__.warning("Dataset is empty.")
        return

    fig = go.Figure()

    if use_optimal_v:
        dim_idx = 0
        V_getter = (lambda t: t.V_N)
        title = r"Cost descent check ($V_N$): $\Delta V = V_N(x_{k+1}) - V_N(x_k)$"
    else:
        dim_idx = 1
        V_getter = lambda t: t.V_pred
        title = r"Cost to go descent check ($V_k$): $\Delta V = V_{k+1} - V_k$"

    per_entry_deltas = []  # list of tuples (id, deltas_2d)
    total_lines = 0

    for entry in dataset:
        traj = entry.trajectory
        id = entry.meta.id

        V = V_getter(traj)
        if V is None:
            __logger__.info(f"Entry {id} missing {'V_N' if use_optimal_v else 'V_preds'}; skipping.")
            continue

        V_arr = np.asarray(V, dtype=float)
        deltas = np.diff(V_arr, axis=dim_idx)

        per_entry_deltas.append((id, deltas))
        total_lines += int(deltas.shape[0])

    if total_lines > SUMMARY_LINE_THRESHOLD:
        __logger__.info(
            f"cost_descent: {total_lines} lines exceed threshold {SUMMARY_LINE_THRESHOLD}; plotting summary stats."
        )

        if use_optimal_v:
            stacked = _nanpad_stack_1d([d for _, d in per_entry_deltas])
            steps = np.arange(stacked.shape[1])
        else:
            lines_1d: list[NDArray] = []
            for _, d in per_entry_deltas:
                d2 = np.asarray(d, dtype=float)
                if d2.ndim != 2:
                    raise ValueError(f"Expected 2D deltas for use_optimal_v=False, got shape {d2.shape}")
                lines_1d.extend([d2[i, :] for i in range(d2.shape[0])])

            stacked = _nanpad_stack_1d(lines_1d)  # (total_lines, max_n_steps)
            steps = np.arange(stacked.shape[1])

        _add_summary_band(fig, stacked, steps)
    else:
        if use_optimal_v:
            for id, deltas_1d in per_entry_deltas:
                color = COLORS[id % len(COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(int(deltas_1d.shape[0])),
                        y=np.asarray(deltas_1d, dtype=float).reshape(-1),
                        mode='lines',
                        name=_to_latex(f'Run ${id+1}$ - $\Delta V$'),
                        line=dict(color=color, width=2),
                        legendgroup=_to_latex(f'Run ${id+1}$'),
                        showlegend=True,
                    )
                )
        else:
            for id, deltas_2d in per_entry_deltas:
                steps = np.tile(np.arange(deltas_2d.shape[1]), (deltas_2d.shape[0], 1))

                deltas = _plotly_multiline(deltas_2d)
                steps = _plotly_multiline(steps)

                color = COLORS[id % len(COLORS)]

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=deltas,
                        mode='lines',
                        name=_to_latex(f'Run ${id+1}$ - $\Delta V$'),
                        line=dict(color=color, width=2),
                        legendgroup=_to_latex(f'Run ${id+1}$'),
                        showlegend=True,
                    )
                )

    _add_zero_reference_line(fig, 0, 1)
    _apply_timeseries_layout(
        fig,
        title_text=title,
        xaxis_title=r"$k$",
        yaxis_title=r"$\Delta V_k$",
    )

    if html_path is not None:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path, include_mathjax='cdn')
        __logger__.info(f"Cost to go descent plot saved to {html_path}.")
    else:
        return fig

def roa(
    lyapunov_func: Callable[[NDArray], NDArray],
    c_level: float,
    bounds: NDArray,  # shape (n_points, nx)
    state_indices: list[int] | None = None,
    state_labels: list[str] | None = None,
    limits: list[tuple[float, float]] | None = None,
    resolution: int = 100,
    plot_3d: bool = False,
    html_path: str | None = None
) -> dict[tuple[int, int], go.Figure] | go.Figure | None:
    """_summary_

    Parameters
    ----------
    lyapunov_func : Callable[[NDArray], NDArray]
        _description_
    c_level : float
        _description_
    bounds : NDArray
        _description_
    state_labels : list[str] | None, optional
        _description_, by default None
    limits : list[tuple[float, float]] | None, optional
        _description_, by default None
    resolution : int, optional
        _description_, by default 100
    plot_3d : bool, optional
        _description_, by default False
    html_path : str | None, optional
        _description_, by default None

    Returns
    -------
    dict[tuple[int, int], go.Figure] | go.Figure | None
        _description_
    """
    bounds_arr = _resolve_bounds(bounds)
    nx = int(bounds_arr.shape[1])

    state_indices = _resolve_indices(state_indices, nx)
    labels_full = _resolve_labels(state_labels, nx)

    limits = _resolve_limits(limits)

    pairs = _state_index_pairs(state_indices)
    figs: dict[tuple[int, int], go.Figure] = {}

    for idx_x, idx_y in pairs:
        pair_labels = [labels_full[idx_x], labels_full[idx_y]]

        if limits is None:
            pair_limits = _infer_pair_limits(
                bounds_arr[:, idx_x],
                bounds_arr[:, idx_y],
                pad_ratio=0.1,
                min_pad=1e-12,
            )
        else:
            pair_limits = limits

        x_vec, y_vec, X, _, full_points = _make_pair_grid(
            pair_limits, resolution, nx, idx_x, idx_y
        )
        Z_flat = _evaluate_lyapunov(lyapunov_func, full_points)
        Z = Z_flat.reshape(X.shape)

        fig = go.Figure()

        _add_lyapunov_landscape(
            fig,
            Z,
            x_vec,
            y_vec,
            plot_3d=plot_3d,
            name='$V(x)$',
        )
        _add_roa_boundary_trace(
            fig,
            x_vec,
            y_vec,
            Z,
            c_level,
            plot_3d=plot_3d,
        )

        _apply_pair_layout(
            fig,
            plot_3d=plot_3d,
            pair_labels=pair_labels,
            pair_limits=pair_limits,
            title_2d=f"Stability Verification: ROA for $c={c_level:.2f}$",
            title_3d=f"Stability Verification: ROA for $c={c_level:.2f}$ (${pair_labels[0]}$ vs ${pair_labels[1]}$)",
            zaxis_title="$V(x)$",
        )
        figs[(idx_x, idx_y)] = fig

    if html_path:
        _save_pair_figures(figs, html_path, labels_full, kind="ROA")
        return None

    if len(figs) == 1:
        return next(iter(figs.values()))
    return figs


def all(
    dataset: MPCDataset,
    state_labels: list[str] | None = None,
    control_labels: list[str] | None = None,
    limits: list[tuple[float, float]] | None = None,
    base_path: str | None = None,
    resolution: int = 100,
    plot_3d: bool = False,

    # trajectory specific
    time_bound: float | None = None,
    plot_predictions: bool = False,

    # cost descent specific
    use_optimal_v: bool = False,

    # relaxed dp specific
    alpha: float = 1.0,

    # lyapunov specific
    lyapunov_func: Callable[[NDArray], NDArray] = None,
    lyap_state_indices: list[int] | None = None,
    lyap_use_dataset_v: bool = False,

    # roa specific
    roa_lyapunov_func: Callable[[NDArray], NDArray] = None,
    c_level: float = None,
    roa_bounds: NDArray = None,
) -> None:
    """Convenience function to plot trajectories, residuals, Lyapunov, and ROA together."""
    lyap_state_labels = state_labels

    mpc_trajectories(
        dataset=dataset,
        state_labels=state_labels,
        control_labels=control_labels,
        plot_predictions=plot_predictions,
        time_bound=time_bound,
        html_path=f"{base_path}_trajectories.html" if base_path else None
    )

    cost_descent_path = f"{base_path}_cost_descent.html" if base_path else None
    cost_descent(dataset=dataset, html_path=cost_descent_path, use_optimal_v=use_optimal_v)

    relaxed_dp_path = f"{base_path}_relaxed_dp.html" if base_path else None
    relaxed_dp_residual(dataset=dataset, html_path=relaxed_dp_path, alpha=alpha)

    if lyapunov_func is None:
        __logger__.warning("Lyapunov function is required for plotting the lyapunov function.")
        return
    
    lyapunov_path = f"{base_path}_lyapunov.html" if base_path else None
    lyapunov(
        dataset=dataset,
        lyapunov_func=lyapunov_func,
        state_indices=lyap_state_indices,
        state_labels=lyap_state_labels,
        limits=limits,
        resolution=resolution,
        plot_3d=plot_3d,
        html_path=lyapunov_path,
        use_dataset_v=lyap_use_dataset_v
    )

    if roa_lyapunov_func is None:
        __logger__.warning("ROA Lyapunov function is required for plotting the region of attraction.")
        return 

    roa_path = f"{base_path}_roa.html" if base_path else None
    roa(
        lyapunov_func=roa_lyapunov_func,
        c_level=c_level,
        bounds=roa_bounds,
        state_indices=lyap_state_indices,
        state_labels=state_labels,
        limits=limits,
        resolution=resolution,
        plot_3d=plot_3d,
        html_path=roa_path
    )