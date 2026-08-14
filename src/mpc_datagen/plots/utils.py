import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go

from ..mpc_data import MPCData, MPCDataset, MPCTrajectory

__logger__ = logging.getLogger(__name__)

try:
    from plotikz import to_tikz
except ImportError:
    __logger__.warning("plotikz could not be imported. Continuing because it is not relevant for this package.")


COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

SUMMARY_LINE_THRESHOLD = 100


@dataclass
class PairPlotResult:
    """Result container for a 2D state-pair plot.

    Attributes
    ----------
    idx_x : int
        State index for x-axis.
    idx_y : int
        State index for y-axis.
    label_x : str
        Label for x-axis.
    label_y : str
        Label for y-axis.
    figure : go.Figure
        Plotly figure object.
    """
    idx_x: int
    idx_y: int
    label_x: str
    label_y: str
    figure: go.Figure

    @property
    def file_slug(self) -> str:
        """Generate filename slug for this state pair."""
        return f"{_slug(self.label_x)}_vs_{_slug(self.label_y)}"


def _slug(label: str) -> str:
    """Convert label to filesystem-safe lowercase slug."""
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


def _plotly_multiline(x: NDArray, axis: int = 0) -> NDArray:
    """Interleave NaN values to create multiple disjoint lines in a single Plotly trace."""
    if axis == 0:
        return np.hstack([x, np.full((x.shape[0], 1), np.nan)]).flatten()
    elif axis == 1:
        return np.vstack([x, np.full((x.shape[1], 1), np.nan)]).flatten()
    raise ValueError(f"Unsupported axis={axis} for _plotly_multiline, expected 0 or 1.")


def _resolve_num_states(
    num_states: int | None = None,
    dataset: MPCDataset | None = None,
    state_labels: list[str] | None = None,
    state_indices: list[int] | None = None,
) -> int:
    """Resolve state dimension from configuration, dataset, or labels."""
    nx = None

    if dataset is not None and len(dataset) > 0 and dataset[0].trajectory.states is not None:
        nx = int(dataset[0].config.nx)

    if num_states is not None:
        if nx is not None and nx != int(num_states):
            raise ValueError(
                f"Conflict: Provided {num_states=} does not match dataset dimension {nx}."
            )
        nx = int(num_states)

    if nx is None:
        inferred_indices = max(state_indices) + 1 if state_indices else 0
        inferred_labels = len(state_labels) if state_labels else 0
        nx = max(inferred_indices, inferred_labels)

    if not nx:
        raise ValueError(
            "Could not determine the number of states. Please provide "
            "num_states, a valid dataset, state_indices, or state_labels."
        )

    return nx


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
    """Validate user-provided axis limits and normalize to float tuples."""
    if limits is None:
        return None
    if len(limits) < 2:
        raise ValueError("limits must contain at least two axis bounds tuples.")

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


def _pair_limits_from_resolved(
    limits: list[tuple[float, float]] | None,
    idx_x: int,
    idx_y: int,
    num_states: int,
) -> list[tuple[float, float]] | None:
    """Select pair-specific limits from 2D or full-state limits."""
    if limits is None:
        return None
    if len(limits) == 2:
        return limits
    if len(limits) == num_states:
        return [limits[idx_x], limits[idx_y]]
    raise ValueError(
        "limits must contain either exactly two tuples or one tuple per state dimension."
    )


def _combine_pair_limits(
    *limit_sets: list[tuple[float, float]] | None,
) -> list[tuple[float, float]] | None:
    """Combine multiple 2D limit sets into one outer bounding box."""
    resolved = [limit_set for limit_set in limit_sets if limit_set is not None]
    if not resolved:
        return None

    num_axes = len(resolved[0])
    if any(len(limit_set) != num_axes for limit_set in resolved[1:]):
        raise ValueError("All limit sets must have the same number of axes.")

    combined: list[tuple[float, float]] = []
    for axis_idx in range(num_axes):
        lower = min(float(limit_set[axis_idx][0]) for limit_set in resolved)
        upper = max(float(limit_set[axis_idx][1]) for limit_set in resolved)
        combined.append((lower, upper))
    return combined


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


def _infer_state_limits(
    states: NDArray,
    *,
    pad_ratio: float = 0.1,
    min_pad: float = 1e-12,
) -> list[tuple[float, float]]:
    """Infer padded plotting limits for every state dimension."""
    states_arr = np.asarray(states, dtype=float)
    if states_arr.ndim != 2:
        raise ValueError("states must be a 2D array with shape (n_points, nx).")
    if int(states_arr.shape[0]) == 0:
        raise ValueError("Cannot infer limits from an empty states array.")
    if int(states_arr.shape[1]) < 2:
        raise ValueError("states must contain at least 2 state dimensions.")

    limits: list[tuple[float, float]] = []
    for idx in range(int(states_arr.shape[1])):
        values = states_arr[:, idx].reshape(-1)
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        pad = pad_ratio * max(min_pad, value_max - value_min)
        limits.append((value_min - pad, value_max + pad))

    return limits


def _state_index_pairs(state_indices: list[int]) -> list[tuple[int, int]]:
    """Generate all unique pairwise combinations from state indices."""
    if len(state_indices) < 2:
        raise ValueError("state_indices must contain at least 2 indices.")
    return list(combinations(state_indices, 2))


def _extract_trajectory_v(
    traj: MPCTrajectory,
    entry: MPCData | None = None,
    use_solver_fallback: bool = True,
) -> NDArray | None:
    """Extract optimal value function V_N from a trajectory entry.

    Attempts to read `traj.V_N`. If missing and predictions/cost are available,
    attempts recalculating `V_N`. If still missing and `use_solver_fallback` is True,
    falls back to `traj.V_solver`.

    Parameters
    ----------
    traj : MPCTrajectory
        Trajectory containing simulation data.
    entry : MPCData, optional
        Full MPC dataset entry containing configuration (used for cost recalculation).
    use_solver_fallback : bool, optional
        Whether to fall back to solver objective values if V_N is None.

    Returns
    -------
    NDArray | None
        1D array of optimal values V_N per step, or None if unavailable.
    """
    if traj.V_N is not None and traj.V_N.size > 0:
        return np.asarray(traj.V_N, dtype=float).reshape(-1)

    if (
        entry is not None
        and hasattr(entry, "config")
        and entry.config is not None
        and traj.predicted_states is not None
        and traj.predicted_inputs is not None
    ):
        try:
            traj.recalculate_costs(entry.config.cost)
            if traj.V_N is not None and traj.V_N.size > 0:
                return np.asarray(traj.V_N, dtype=float).reshape(-1)
        except Exception as e:
            __logger__.debug(f"Could not recalculate costs: {e}")

    if use_solver_fallback and traj.V_solver is not None and traj.V_solver.size > 0:
        return np.asarray(traj.V_solver, dtype=float).reshape(-1)

    return None


def _extract_dataset_state_value_pairs(
    dataset: MPCDataset,
    idx_x: int,
    idx_y: int,
    use_solver_v: bool = True,
) -> tuple[NDArray, NDArray, NDArray]:
    """Extract (x, y, V) tuples across all trajectories in a dataset.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories.
    idx_x : int
        State index for x-axis.
    idx_y : int
        State index for y-axis.
    use_solver_v : bool, optional
        If True, extracts `traj.V_solver` directly. If False, extracts `traj.V_N`
        via `_extract_trajectory_v`. Default is True.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Arrays (x_all, y_all, v_all) of corresponding state coordinates and optimal values.
    """
    x_list: list[float] = []
    y_list: list[float] = []
    v_list: list[float] = []

    for entry in dataset:
        traj = entry.trajectory
        if use_solver_v:
            v_opt = (
                np.asarray(traj.V_solver, dtype=float).reshape(-1)
                if traj.V_solver is not None and traj.V_solver.size > 0
                else None
            )
        else:
            v_opt = _extract_trajectory_v(traj, entry)

        if v_opt is None:
            continue

        states = np.asarray(traj.states, dtype=float)
        n_pts = min(states.shape[0], v_opt.shape[0])
        if n_pts <= 0:
            continue

        x_pts = states[:n_pts, idx_x].tolist()
        y_pts = states[:n_pts, idx_y].tolist()
        v_pts = v_opt[:n_pts].tolist()

        x_list.extend(x_pts)
        y_list.extend(y_pts)
        v_list.extend(v_pts)

    return (
        np.asarray(x_list, dtype=float),
        np.asarray(y_list, dtype=float),
        np.asarray(v_list, dtype=float),
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
    results: list[PairPlotResult],
    html_path: Path | str,
    *,
    kind: str,
    tikz: bool = True,
) -> None:
    """Save one or multiple pair figures to html files."""
    path = Path(html_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".html"
    base_name = path.stem

    for result in results:
        file_name = f"{base_name}_{result.file_slug}{suffix}"
        file_path = path.parent / file_name
        result.figure.write_html(str(file_path), include_mathjax='cdn')
        if tikz:
            try:
                to_tikz(result.figure, file_path.with_suffix(".tex"))
            except Exception as e:
                __logger__.warning(f"Could not save tikz figure {file_name}. Continuing!\n{e}")
    __logger__.info(f"{len(results)} {kind} plots saved to {path.parent}.")


def _handle_figure_output(
    fig: go.Figure,
    html_path: Path | str | None,
    log_message: str,
    tikz: bool = True,
) -> go.Figure | None:
    """Helper to handle HTML export or return the figure."""
    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(html_path, include_mathjax='cdn')
        if tikz:
            try:
                to_tikz(fig, html_path.with_suffix(".tex"))
            except Exception as e:
                __logger__.warning(f"Could not save tikz figure {html_path.name}. Continuing!\n{e}")
        __logger__.info(f"{log_message} saved to {html_path}.")
        return None

    return fig


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


def _add_traces_or_summary(
    fig: go.Figure,
    per_entry_deltas: list[tuple[int, NDArray]],
    total_lines: int,
    trace_name_func: Callable[[int], str],
    is_2d: bool = False,
) -> None:
    """Helper to either plot summary bands or individual traces based on line count."""
    if total_lines > SUMMARY_LINE_THRESHOLD:
        __logger__.info(
            f"Plotting summary stats: {total_lines} lines exceed threshold {SUMMARY_LINE_THRESHOLD}."
        )
        if is_2d:
            lines_1d = []
            for _, d in per_entry_deltas:
                d2 = np.asarray(d, dtype=float)
                lines_1d.extend([d2[i, :] for i in range(d2.shape[0])])
            stacked = _nanpad_stack_1d(lines_1d)
        else:
            stacked = _nanpad_stack_1d([d for _, d in per_entry_deltas])

        steps = np.arange(stacked.shape[1])
        _add_summary_band(fig, stacked, steps)
    else:
        if is_2d:
            for entry_id, deltas_2d in per_entry_deltas:
                steps_arr = np.tile(np.arange(deltas_2d.shape[1]), (deltas_2d.shape[0], 1))
                deltas_flat = _plotly_multiline(deltas_2d)
                steps_flat = _plotly_multiline(steps_arr)
                color = COLORS[entry_id % len(COLORS)]

                fig.add_trace(
                    go.Scatter(
                        x=steps_flat,
                        y=deltas_flat,
                        mode='lines',
                        name=_to_latex(trace_name_func(entry_id)),
                        line=dict(color=color, width=2),
                        legendgroup=_to_latex(f'Run ${entry_id+1}$'),
                        showlegend=True,
                    )
                )
        else:
            for entry_id, deltas in per_entry_deltas:
                color = COLORS[entry_id % len(COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(int(deltas.shape[0])),
                        y=np.asarray(deltas, dtype=float).reshape(-1),
                        mode='lines',
                        name=_to_latex(trace_name_func(entry_id)),
                        line=dict(color=color, width=2),
                        legendgroup=_to_latex(f'Run ${entry_id+1}$'),
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
