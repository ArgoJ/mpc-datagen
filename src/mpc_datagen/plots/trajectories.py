import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..mpc_data import MPCDataset
from .utils import COLORS, _handle_figure_output, _to_latex

__logger__ = logging.getLogger(__name__)


def mpc_trajectories(
    dataset: MPCDataset,
    state_labels: list[str],
    control_labels: list[str],
    plot_predictions: bool = False,
    time_bound: float | None = None,
    html_path: Path | str | None = None,
) -> go.Figure | None:
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
    time_bound : float, optional
        If provided, limits the x-axis to the specified time range [0, time_bound].
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

    # Extract dimensions from the first trajectory
    first_traj = dataset[0].trajectory
    num_states = first_traj.states.shape[1]
    num_controls = first_traj.inputs.shape[1]

    # Create subplots
    fig = make_subplots(
        rows=num_states + num_controls,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
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
                    showlegend=(i == 0),
                ),
                row=row,
                col=1,
            )

            if (
                plot_predictions
                and traj.predicted_states is not None
                and not np.all(np.isnan(traj.predicted_states))
            ):
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

                    x_lines.extend(t_pred.tolist())
                    x_lines.append(None)
                    y_lines.extend(pred_state.tolist())
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
                        hoverinfo='skip',
                    ),
                    row=row,
                    col=1,
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
                    line=dict(color=color, shape='hv'),  # 'hv' for step-after behavior
                    name=_to_latex(f'Run ${idx+1}$ - {control_labels[i]}'),
                    legendgroup=_to_latex(f'Run ${idx+1}$'),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

            if (
                plot_predictions
                and traj.predicted_inputs is not None
                and not np.all(np.isnan(traj.predicted_inputs))
            ):
                dt = traj.times[1] - traj.times[0] if len(traj.times) > 1 else 0.1

                x_lines = []
                y_lines = []

                for k in range(traj.predicted_inputs.shape[0]):
                    pred_input = traj.predicted_inputs[k, :, i]
                    if np.isnan(pred_input).all():
                        continue

                    t_start = traj.times[k]
                    t_pred = t_start + np.arange(len(pred_input)) * dt

                    x_lines.extend(t_pred.tolist())
                    x_lines.append(None)
                    y_lines.extend(pred_input.tolist())
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
                        hoverinfo='skip',
                    ),
                    row=row,
                    col=1,
                )
                prediction_indices.append(len(fig.data) - 1)

    fig.update_layout(
        height=300 * (num_states + num_controls),
        title_text=_to_latex("MPC Trajectories"),
        hovermode="x unified",
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

    return _handle_figure_output(fig, html_path, "Trajectories plot")


# Convenient alias
trajectories = mpc_trajectories
