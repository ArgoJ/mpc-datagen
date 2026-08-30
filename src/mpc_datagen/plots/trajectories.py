import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..mpc_data import MPCDataset
from .utils import COLORS, _handle_figure_output, _to_latex

__logger__ = logging.getLogger(__name__)


def mpc_trajectories(
    *dataset: MPCDataset,
    dataset_labels: list[str] | None = None,
    state_labels: list[str] | None = None,
    control_labels: list[str] | None = None,
    plot_predictions: bool = False,
    time_bound: float | None = None,
    html_path: Path | str | None = None,
    **kwargs,
) -> go.Figure | None:
    """Plot MPC trajectories for states and controls using Plotly.

    Parameters
    ----------
    *dataset : MPCDataset
        One or more datasets containing trajectories to plot. If multiple datasets
        are provided, each dataset is assigned its own color.
    dataset_labels : list[str], optional
        List of labels for each dataset. If provided, used for trace names and legend.
    state_labels : list[str], optional
        List of labels for each state variable. Defaults to ["$x_1$", "$x_2$", ...].
    control_labels : list[str], optional
        List of labels for each control variable. Defaults to ["$u_1$", "$u_2$", ...].
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
    datasets = [d for d in dataset if isinstance(d, MPCDataset)]
    if not datasets and "dataset" in kwargs and isinstance(kwargs["dataset"], MPCDataset):
        datasets = [kwargs["dataset"]]

    trajs = [
        (d_idx, run_idx, entry.trajectory)
        for d_idx, ds in enumerate(datasets)
        for run_idx, entry in enumerate(ds)
    ]
    if not trajs:
        __logger__.warning("Dataset is empty.")
        return None

    multi = len(datasets) > 1
    first_traj = trajs[0][2]
    num_states = first_traj.states.shape[1]
    num_controls = first_traj.inputs.shape[1]

    if state_labels is None:
        state_labels = [f"$x_{{{i+1}}}$" for i in range(num_states)]
    if control_labels is None:
        control_labels = [f"$u_{{{i+1}}}$" for i in range(num_controls)]

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
        for d_idx, run_idx, traj in trajs:
            color = COLORS[(d_idx if multi else run_idx) % len(COLORS)]
            if multi:
                name = dataset_labels[d_idx] if dataset_labels and d_idx < len(dataset_labels) else f"Dataset ${d_idx+1}$"
            else:
                name = f"{dataset_labels[0]} - Run ${run_idx+1}$" if dataset_labels else f"Run ${run_idx+1}$"
            showlegend = (i == 0 and (run_idx == 0 if multi else True))

            # Main Trajectory
            fig.add_trace(
                go.Scatter(
                    x=traj.times,
                    y=traj.states[:, i],
                    mode="lines",
                    name=_to_latex(name),
                    line=dict(color=color),
                    legendgroup=_to_latex(name),
                    showlegend=showlegend,
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
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.3,
                        showlegend=False,
                        legendgroup=_to_latex(name),
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=1,
                )
                prediction_indices.append(len(fig.data) - 1)

    # Plot controls
    for i in range(num_controls):
        plot_idx = num_states + i
        row = plot_idx + 1
        for d_idx, run_idx, traj in trajs:
            color = COLORS[(d_idx if multi else run_idx) % len(COLORS)]
            if multi:
                name = dataset_labels[d_idx] if dataset_labels and d_idx < len(dataset_labels) else f"Dataset ${d_idx+1}$"
            else:
                name = f"{dataset_labels[0]} - Run ${run_idx+1}$" if dataset_labels else f"Run ${run_idx+1}$"

            # Controls (Step plot)
            fig.add_trace(
                go.Scatter(
                    x=traj.times[:-1],
                    y=traj.inputs[:, i],
                    mode="lines",
                    line=dict(color=color, shape="hv"),  # "hv" for step-after behavior
                    name=_to_latex(f"{name} - {control_labels[i]}"),
                    legendgroup=_to_latex(name),
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
                        mode="lines",
                        line=dict(color=color, width=1, shape="hv"),
                        opacity=0.3,
                        showlegend=False,
                        legendgroup=_to_latex(name),
                        hoverinfo="skip",
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


def trajectory_error_bands(
    errors_dataset: MPCDataset,
    state_labels: list[str] | None = None,
    control_labels: list[str] | None = None,
    plot_controls: bool = False,
    show_individual: bool = False,
    show_median: bool = True,
    time_bound: float | None = None,
    html_path: Path | str | None = None,
    tikz: bool = True,
) -> go.Figure | None:
    """Plot error bands (min, max, mean, median) for a dataset of trajectory errors per state.

    Parameters
    ----------
    errors_dataset : MPCDataset
        The dataset containing trajectory errors (e.g. diff between NN rollout and MPC expert).
    state_labels : list[str], optional
        Labels for each state error dimension. Defaults to ["$\\Delta x_1$", "$\\Delta x_2$", ...].
    control_labels : list[str], optional
        Labels for each control error dimension. Defaults to ["$\\Delta u_1$", "$\\Delta u_2$", ...] if plot_controls is True.
    plot_controls : bool, optional
        If True, also plots error bands for control input errors. Default is False.
    show_individual : bool, optional
        If True, renders individual trajectory error traces initially. Default is False (toggleable via button).
    show_median : bool, optional
        If True, includes the median error curve alongside the mean. Default is True.
    time_bound : float, optional
        Limits the x-axis to [0, time_bound].
    html_path : Path | str, optional
        If provided, saves the interactive plot to an HTML file.
    tikz : bool, optional
        If True and html_path is provided, also attempts saving TikZ format. Default is True.

    Returns
    -------
    go.Figure | None
        Plotly Figure object if html_path is None, else None.
    """
    if len(errors_dataset) == 0:
        __logger__.warning("Errors dataset is empty.")
        return None

    num_runs = len(errors_dataset)
    first_traj = errors_dataset[0].trajectory
    num_states = first_traj.states.shape[1]
    num_controls = first_traj.inputs.shape[1] if first_traj.inputs is not None and first_traj.inputs.ndim == 2 else 0

    if state_labels is None:
        state_labels = [f"$\\Delta x_{{{i+1}}}$" for i in range(num_states)]
    if control_labels is None and num_controls > 0:
        control_labels = [f"$\\Delta u_{{{i+1}}}$" for i in range(num_controls)]

    num_plots = num_states + (num_controls if plot_controls and num_controls > 0 else 0)

    # Collect state error trajectories: (M, T_state, num_states)
    state_errors_list = [entry.trajectory.states[:, :num_states] for entry in errors_dataset]
    max_steps = max(arr.shape[0] for arr in state_errors_list)
    state_errors = np.stack([
        np.pad(arr, ((0, max_steps - len(arr)), (0, 0)), constant_values=np.nan)
        for arr in state_errors_list
    ])

    try:
        dt = errors_dataset.global_config.dt
    except Exception:
        dt = getattr(errors_dataset[0].config, "dt", 0.1)
    times = np.arange(max_steps) * dt

    # Collect control error trajectories if requested
    if plot_controls and num_controls > 0:
        ctrl_errors_list = [entry.trajectory.inputs[:, :num_controls] for entry in errors_dataset if entry.trajectory.inputs is not None]
        if ctrl_errors_list:
            max_ctrl_steps = max(arr.shape[0] for arr in ctrl_errors_list)
            ctrl_errors = np.stack([
                np.pad(arr, ((0, max_ctrl_steps - len(arr)), (0, 0)), constant_values=np.nan)
                for arr in ctrl_errors_list
            ])
            ctrl_times = times[:max_ctrl_steps]
        else:
            plot_controls = False
            num_plots = num_states

    fig = make_subplots(
        rows=num_plots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    individual_indices = []

    # Plot State Error Bands
    for i in range(num_states):
        row = i + 1
        err_i = state_errors[:, :, i]  # (M, T)

        e_min = np.nanmin(err_i, axis=0)
        e_max = np.nanmax(err_i, axis=0)
        e_mean = np.nanmean(err_i, axis=0)
        e_median = np.nanmedian(err_i, axis=0)

        # Zero reference line
        fig.add_trace(
            go.Scatter(
                x=[times[0], times[-1]],
                y=[0.0, 0.0],
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )

        # Individual error traces
        for m in range(num_runs):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=err_i[m, :],
                    mode="lines",
                    line=dict(color="rgba(128, 128, 128, 0.25)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                    visible=show_individual,
                ),
                row=row,
                col=1,
            )
            individual_indices.append(len(fig.data) - 1)

        # Max envelope boundary
        fig.add_trace(
            go.Scatter(
                x=times,
                y=e_max,
                mode="lines",
                name="max",
                line=dict(color="red", width=1.5),
                legendgroup="max",
                showlegend=(i == 0),
            ),
            row=row,
            col=1,
        )

        # Min envelope boundary + Fill to Max
        fig.add_trace(
            go.Scatter(
                x=times,
                y=e_min,
                mode="lines",
                name="min",
                line=dict(color="red", width=1.5),
                fill="tonexty",
                fillcolor="rgba(255, 0, 0, 0.25)",
                legendgroup="min",
                showlegend=(i == 0),
            ),
            row=row,
            col=1,
        )

        # Mean curve
        fig.add_trace(
            go.Scatter(
                x=times,
                y=e_mean,
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name="mean",
                legendgroup="mean",
                showlegend=(i == 0),
            ),
            row=row,
            col=1,
        )

        # Median curve
        if show_median:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=e_median,
                    mode="lines",
                    line=dict(color="blue", width=2),
                    name="median",
                    legendgroup="median",
                    showlegend=(i == 0),
                ),
                row=row,
                col=1,
            )

        lbl = state_labels[i]
        fig.update_yaxes(title_text=_to_latex(lbl), row=row, col=1)

    # Plot Control Error Bands if requested
    if plot_controls and num_controls > 0:
        for i in range(num_controls):
            row = num_states + i + 1
            err_u = ctrl_errors[:, :, i]

            u_min = np.nanmin(err_u, axis=0)
            u_max = np.nanmax(err_u, axis=0)
            u_mean = np.nanmean(err_u, axis=0)
            u_median = np.nanmedian(err_u, axis=0)

            # Zero reference line
            fig.add_trace(
                go.Scatter(
                    x=[ctrl_times[0], ctrl_times[-1]],
                    y=[0.0, 0.0],
                    mode="lines",
                    line=dict(color="black", width=1, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )

            # Individual control error traces
            for m in range(num_runs):
                fig.add_trace(
                    go.Scatter(
                        x=ctrl_times,
                        y=err_u[m, :],
                        mode="lines",
                        line=dict(color="rgba(128, 128, 128, 0.25)", width=1, shape="hv"),
                        showlegend=False,
                        hoverinfo="skip",
                        visible=show_individual,
                    ),
                    row=row,
                    col=1,
                )
                individual_indices.append(len(fig.data) - 1)

            # Max boundary
            fig.add_trace(
                go.Scatter(
                    x=ctrl_times,
                    y=u_max,
                    mode="lines",
                    name="max",
                    line=dict(color="red", width=1.5, shape="hv"),
                    legendgroup="max",
                    showlegend=(num_states == 0 and i == 0),
                ),
                row=row,
                col=1,
            )

            # Min boundary + Fill
            fig.add_trace(
                go.Scatter(
                    x=ctrl_times,
                    y=u_min,
                    mode="lines",
                    name="min",
                    line=dict(color="red", width=1.5, shape="hv"),
                    fill="tonexty",
                    fillcolor="rgba(255, 0, 0, 0.25)",
                    legendgroup="min",
                    showlegend=(num_states == 0 and i == 0),
                ),
                row=row,
                col=1,
            )

            # Mean curve
            fig.add_trace(
                go.Scatter(
                    x=ctrl_times,
                    y=u_mean,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dash", shape="hv"),
                    name="mean",
                    legendgroup="mean",
                    showlegend=(num_states == 0 and i == 0),
                ),
                row=row,
                col=1,
            )

            # Median curve
            if show_median:
                fig.add_trace(
                    go.Scatter(
                        x=ctrl_times,
                        y=u_median,
                        mode="lines",
                        line=dict(color="blue", width=2, shape="hv"),
                        name="median",
                        legendgroup="median",
                        showlegend=(num_states == 0 and i == 0),
                    ),
                    row=row,
                    col=1,
                )

            lbl = control_labels[i]
            fig.update_yaxes(title_text=_to_latex(lbl), row=row, col=1)

    fig.update_xaxes(title_text=_to_latex("$t$"), row=num_plots, col=1)
    fig.update_layout(
        height=280 * num_plots,
        title_text=_to_latex("Trajectory Error Bands"),
        hovermode="x unified",
    )

    if time_bound is not None:
        fig.update_xaxes(range=[0, time_bound])

    if individual_indices:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": True}, individual_indices],
                            args2=[{"visible": False}, individual_indices],
                            label="Individual Runs",
                            method="restyle",
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=show_individual,
                    x=1.0,
                    xanchor="right",
                    y=-0.05,
                    yanchor="top",
                ),
            ]
        )

    return _handle_figure_output(fig, html_path, "Trajectory error bands plot", tikz=tikz)


# Convenient aliases
trajectories = mpc_trajectories
error_band = trajectory_error_bands
error_bands = trajectory_error_bands
