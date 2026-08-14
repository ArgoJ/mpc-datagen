import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from ..mpc_data import MPCDataset
from .utils import (
    _add_traces_or_summary,
    _add_zero_reference_line,
    _apply_timeseries_layout,
    _extract_trajectory_v,
    _handle_figure_output,
    _to_latex,
)

__logger__ = logging.getLogger(__name__)


def relaxed_dp_residual(
    dataset: MPCDataset,
    alpha: float = 1.0,
    html_path: Path | str | None = None,
) -> go.Figure | None:
    r"""Plot Lyapunov-style one-step descent check.

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
        empirically verified decay rate. Default is 1.0.
    html_path : Path | str, optional
        If provided, saves the plot to the specified HTML file.

    Returns
    -------
    go.Figure | None
        Plotly Figure object if html_path is None, else None.
    """
    if len(dataset) == 0:
        __logger__.warning("Dataset is empty.")
        return None

    fig = go.Figure()
    if alpha <= 0.0 or alpha > 1.0:
        __logger__.warning(f"alpha must be in the range (0, 1], got {alpha}. Setting alpha=1.0.")
        alpha = 1.0

    if np.isclose(alpha, 1.0):
        title = _to_latex(r"DP Lyapunov residual: $s_n = V_N(x_{n+1}) - V_N(x_n) + \ell(x_n,u_n)$")
    else:
        title = _to_latex(
            r"Relaxed DP Lyapunov residual: $s_n(\alpha) = V_N(x_{n+1}) - V_N(x_n) + \alpha \ell(x_n,u_n)$ with $\alpha = "
            + f"{alpha:.3f}$"
        )

    per_entry = []  # list of tuples (id, deltas)
    for entry in dataset:
        traj = entry.trajectory
        cost = entry.config.cost
        entry_id = entry.meta.id

        v_opt = _extract_trajectory_v(traj, entry)
        if v_opt is None:
            __logger__.info(f"Entry {entry_id} missing V_N; skipping.")
            continue

        # Ensure consistent lengths across V_N, states, and inputs.
        num_steps = min(
            int(len(v_opt) - 1),
            int(traj.inputs.shape[0]),
            int(traj.states.shape[0] - 1),
        )
        if num_steps <= 0:
            __logger__.info(f"Entry {entry_id} has insufficient steps; skipping.")
            continue

        x = traj.states[:num_steps]
        u = traj.inputs[:num_steps]
        l_n = np.asarray(cost.get_stage_cost(x, u), dtype=float).reshape(-1)
        v_curr = np.asarray(v_opt[:num_steps], dtype=float).reshape(-1)
        v_next = np.asarray(v_opt[1 : num_steps + 1], dtype=float).reshape(-1)

        deltas = v_next - v_curr + float(alpha) * l_n
        per_entry.append((entry_id, deltas))

    if not per_entry:
        __logger__.warning("No valid entries with V_N found for relaxed DP residual.")
        return None

    n_lines = len(per_entry)
    max_len = max((int(d.shape[0]) for _, d in per_entry), default=0)

    _add_traces_or_summary(
        fig,
        per_entry,
        n_lines,
        trace_name_func=lambda id_: f'Run ${id_+1}$ - $s_n(\\alpha)$',
        is_2d=False,
    )
    _add_zero_reference_line(fig, 0, max(1, max_len - 1))
    _apply_timeseries_layout(
        fig,
        title_text=title,
        xaxis_title=r"$n$",
        yaxis_title=r"$s_n(\alpha)$",
    )

    return _handle_figure_output(fig, html_path, "Relaxed DP residual plot")


def cost_descent(
    dataset: MPCDataset,
    html_path: Path | str | None = None,
    use_optimal_v: bool = False,
) -> go.Figure | None:
    r"""Plot cost descent check.

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
    html_path : Path | str, optional
        If provided, saves the plot to the specified HTML file.
    use_optimal_v : bool, optional
        If True, uses the optimal value function V_N along the closed-loop trajectory.
        If False, uses the cost-to-go along the predicted open-loop trajectories.

    Returns
    -------
    go.Figure | None
        Plotly Figure object if html_path is None, else None.
    """
    if len(dataset) == 0:
        __logger__.warning("Dataset is empty.")
        return None

    fig = go.Figure()

    if use_optimal_v:
        dim_idx = 0
        title = r"Cost descent check ($V_N$): $\Delta V = V_N(x_{k+1}) - V_N(x_k)$"
    else:
        dim_idx = 1
        title = r"Cost to go descent check ($V_k$): $\Delta V = V_{k+1} - V_k$"

    per_entry_deltas = []  # list of tuples (id, deltas)
    total_lines = 0
    is_2d = False

    for entry in dataset:
        traj = entry.trajectory
        entry_id = entry.meta.id

        if use_optimal_v:
            v_val = _extract_trajectory_v(traj, entry)
        else:
            try:
                v_val = traj.V_pred
            except ValueError:
                v_val = None

        if v_val is None or (isinstance(v_val, np.ndarray) and v_val.size == 0):
            __logger__.info(f"Entry {entry_id} missing {'V_N' if use_optimal_v else 'V_preds'}; skipping.")
            continue

        v_arr = np.asarray(v_val, dtype=float)
        if use_optimal_v:
            deltas = np.diff(v_arr)
            is_2d = False
            total_lines += 1
        else:
            deltas = np.diff(v_arr, axis=dim_idx)
            is_2d = (deltas.ndim == 2)
            total_lines += deltas.shape[0] if is_2d else 1

        per_entry_deltas.append((entry_id, deltas))

    if not per_entry_deltas:
        __logger__.warning("No valid cost data found for cost descent plot.")
        return None

    _add_traces_or_summary(
        fig,
        per_entry_deltas,
        total_lines,
        trace_name_func=lambda id_: f'Run ${id_+1}$ - $\\Delta V$',
        is_2d=is_2d,
    )
    _add_zero_reference_line(fig, 0, 1)
    _apply_timeseries_layout(
        fig,
        title_text=title,
        xaxis_title=r"$k$",
        yaxis_title=r"$\Delta V_k$",
    )

    return _handle_figure_output(fig, html_path, "Cost descent plot")
