import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Callable, List, Optional, Tuple

from .mpc_data import MPCDataset
from .package_logger import PackageLogger

__logger__ = PackageLogger.get_logger(__name__)


COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

def _plotly_multiline(x: np.ndarray, axis: int=0):
    if axis == 0:
        return np.hstack([x, np.full((x.shape[0], 1), np.nan)]).flatten()
    elif axis == 1:
        return np.vstack([x, np.full((x.shape[1], 1), np.nan)]).flatten()


def mpc_trajectories(
    dataset: MPCDataset,
    state_labels: list,
    control_labels: list,
    plot_predictions: bool = False,
    time_bound: Optional[float] = None,  
    html_path: Optional[str] = None
) -> go.Figure:
    """Plot MPC trajectories for states and controls using Plotly.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories to plot.
    state_labels : list
        List of labels for each state variable.
    control_labels : list
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
        subplot_titles=(state_labels + control_labels),
        vertical_spacing=0.05
    )

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
                    name=f'Run {idx+1} - {state_labels[i]}',
                    line=dict(color=color),
                    legendgroup=f'Run {idx+1}',
                    showlegend=(i == 0)
                ),
                row=row, col=1
            )
            
            if plot_predictions and traj.solved_states is not None and not np.all(np.isnan(traj.solved_states)):
                dt = traj.times[1] - traj.times[0] if len(traj.times) > 1 else 0.1
                
                # Consolidate prediction lines into one trace with None gaps for performance
                x_lines = []
                y_lines = []
                
                for k in range(traj.solved_states.shape[0]):
                    pred_state = traj.solved_states[k, :, i]
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
                    name=f'Run {idx+1} - {control_labels[i]}',
                    legendgroup=f'Run {idx+1}',
                    showlegend=False
                ),
                row=row, col=1
            )
            
            if plot_predictions and traj.solved_inputs is not None and not np.all(np.isnan(traj.solved_inputs)):
                dt = traj.times[1] - traj.times[0] if len(traj.times) > 1 else 0.1
                
                x_lines = []
                y_lines = []
                
                for k in range(traj.solved_inputs.shape[0]):
                    pred_input = traj.solved_inputs[k, :, i]
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
                        legendgroup=f'Run {idx+1}',
                        hoverinfo='skip'
                    ),
                    row=row, col=1
                )
                prediction_indices.append(len(fig.data) - 1)

    fig.update_layout(
        height=300 * (num_states + num_controls), 
        title_text="MPC Trajectories",
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
    dataset: MPCDataset,
    lyapunov_func: Callable[[np.ndarray], np.ndarray],
    state_indices: List[int] = [0, 1],
    state_labels: Optional[List[str]] = None,
    limits: Optional[List[Tuple[float, float]]] = None,
    resolution: int = 100,
    plot_3d: bool = False,
    html_path: Optional[str] = None,
    use_optimal_v: bool = False
) -> Optional[go.Figure]:
    """Plot the Lyapunov function landscape and MPC trajectories in 2D or 3D.
    Only two state dimensions can be visualized at once.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories to plot.
    lyapunov_func : Callable[[np.ndarray], np.ndarray]
        A function that takes a state vector and returns the Lyapunov value.
    state_indices : list, optional
        Indices of the two state variables to plot (x, y axes). Default is [0, 1].
    limits : list of tuples, optional
        ((min_x, max_x), (min_y, max_y)). If None, inferred from data with padding.
    resolution : int, optional
        Grid resolution for the Lyapunov function contour plot.
    plot_3d : bool, optional
        If True, plot a 3D surface and 3D trajectories. Default is False.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.
    """
    if len(dataset) == 0:
        __logger__.warning("Dataset is empty.")
        return

    # Infer dimensions
    first_traj = dataset[0].trajectory
    num_states = first_traj.states.shape[1]
    
    if len(state_indices) != 2:
        raise ValueError("state_indices must contain exactly 2 indices.")

    idx_x, idx_y = state_indices
    
    if state_labels is None:
        state_labels = [f"State {idx_x}", f"State {idx_y}"]
    if len(state_labels) != 2:
        raise ValueError("state_labels must contain exactly 2 labels.")

    # Determine limits if not provided
    if limits is None:
        all_states = np.vstack([d.trajectory.states for d in dataset])
        min_x, max_x = all_states[:, idx_x].min(), all_states[:, idx_x].max()
        min_y, max_y = all_states[:, idx_y].min(), all_states[:, idx_y].max()
        
        # Add some padding
        pad_x = (max_x - min_x) * 0.2 if max_x != min_x else 1.0
        pad_y = (max_y - min_y) * 0.2 if max_y != min_y else 1.0
        
        limits = [
            (min_x - pad_x, max_x + pad_x),
            (min_y - pad_y, max_y + pad_y)
        ]

    # Create grid for Lyapunov function
    x_range = np.linspace(limits[0][0], limits[0][1], resolution)
    y_range = np.linspace(limits[1][0], limits[1][1], resolution)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Prepare grid points for evaluation
    grid_points = np.zeros((X.size, num_states))
    grid_points[:, idx_x] = X.flatten()
    grid_points[:, idx_y] = Y.flatten()
    
    # Evaluate Lyapunov function
    try:
        Z_flat = lyapunov_func(grid_points)
    except Exception:
        Z_flat = np.array([lyapunov_func(s) for s in grid_points])
        
    if hasattr(Z_flat, 'ndim') and Z_flat.ndim > 1:
        Z_flat = Z_flat.flatten()
    elif isinstance(Z_flat, list):
        Z_flat = np.array(Z_flat)
        
    Z = Z_flat.reshape(X.shape)

    fig = go.Figure()

    # Plot Lyapunov Landscape
    if plot_3d:
        fig.add_trace(
            go.Surface(
                z=Z,
                x=x_range,
                y=y_range,
                colorscale='Viridis',
                name='Lyapunov Function',
                opacity=0.8,
                showscale=True
            )
        )
    else:
        fig.add_trace(
            go.Contour(
                z=Z,
                x=x_range,
                y=y_range,
                colorscale='Viridis',
                name='Lyapunov Function',
                showscale=True,
                contours=dict(
                    coloring='heatmap',
                    showlabels=True,
                )
            )
        )

    trajectory_indices = []
    
    # Plot MPC Trajectories
    for idx in range(len(dataset)):
        traj = dataset[idx].trajectory
        color = COLORS[idx % len(COLORS)]
        
        if plot_3d:
            if use_optimal_v and traj.V_horizon is not None:
                v_traj = _plotly_multiline(traj.V_horizon)
                x = _plotly_multiline(traj.predicted_states[:, :, idx_x])
                y = _plotly_multiline(traj.predicted_states[:, :, idx_y])
            else:
                v_traj = traj.V_N.flatten()
                x = traj.states[:-1, idx_x].flatten()
                y = traj.states[:-1, idx_y].flatten()

            if v_traj.shape != x.shape or v_traj.shape != y.shape:
                __logger__.warning(
                    f"Trajectory {idx+1} cost shape {v_traj.shape} does not match state shape {x.shape}; skipping."
                )
                continue

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=v_traj,
                    mode='lines',
                    name=f'Run {idx+1}',
                    line=dict(color=color, width=4),
                    showlegend=False
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=traj.states[:, idx_x],
                    y=traj.states[:, idx_y],
                    mode='lines',
                    name=f'Run {idx+1}',
                    line=dict(color=color, width=2),
                    opacity=0.7,
                    showlegend=False
                )
            )
        trajectory_indices.append(len(fig.data) - 1)

    # Layout Configuration
    if plot_3d:
        fig.update_layout(
            title_text=f"Lyapunov Landscape 3D ({state_labels[0]} vs {state_labels[1]})",
            scene=dict(
                xaxis_title=state_labels[0],
                yaxis_title=state_labels[1],
                zaxis_title="V(x)",
            ),
            width=1000,
            height=800,
            autosize=True
        )
    else:
        fig.update_layout(
            title_text=f"Lyapunov Landscape ({state_labels[0]} vs {state_labels[1]})",
            xaxis_title=state_labels[0],
            yaxis_title=state_labels[1],
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        )
    
    # Toggle Button for Trajectories
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": True}, trajectory_indices],
                        args2=[{"visible": False}, trajectory_indices],
                        label="Trajectories",
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

def relaxed_dp_residual(
    dataset: MPCDataset,
    html_path: Optional[str] = None
) -> go.Figure:
    """Plot Lyapunov-style one-step descent check.

    For each trajectory entry, plots

    $$s_n = V_N(x_{n+1}) - V_N(x_n) + \ell(x_n, u_n)$$

    where $V_N$ is the MPC cost-to-go at time step $n$ (taken from the stored
    per-step value function / objective), and $\ell(x_n,u_n)$ is the *single*
    stage cost at time step $n$ along the closed-loop trajectory.

    Visual interpretation: values above 0 violate the one-step descent
    inequality $V_N(x_{n+1}) - V_N(x_n) \le -\ell(x_n,u_n)$.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories to plot.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The resulting Plotly figure.
    """
    if len(dataset) == 0:
        __logger__.warning("Dataset is empty.")
        return go.Figure()

    fig = go.Figure()

    for entry in dataset:
        traj = entry.trajectory
        cost = entry.config.cost
        id = entry.meta.id

        if traj.V_N is None or np.all(np.isnan(traj.V_N)):
            __logger__.warning(f"Entry {id} has invalid costs; skipping.")
            continue

        num_steps = min(len(traj.V_N) - 1, traj.inputs.shape[0], traj.states.shape[0] - 1)

        if num_steps <= 0:
            __logger__.warning(f"Entry {id} has insufficient steps; skipping.")
            continue

        deltas = np.full(num_steps, np.nan)

        for n in range(num_steps):
            x_n = traj.states[n]
            u_n = traj.inputs[n]

            if not (np.all(np.isfinite(x_n)) and np.all(np.isfinite(u_n))):
                continue

            l_n = cost.get_stage_cost(x_n, u_n)
            deltas[n] = traj.V_N[n + 1] - traj.V_N[n] + l_n
            

        color = COLORS[id % len(COLORS)]

        fig.add_trace(
            go.Scatter(
                x=np.arange(num_steps),
                y=deltas,
                mode='lines',
                name=f'Run {id+1} - s<sub>n</sub>',
                line=dict(color=color, width=2),
                legendgroup=f'Run {id+1}',
                showlegend=True
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        )
    )

    fig.update_layout(
        title_text=(
            "Relaxed DP residual: "
            "s<sub>n</sub> = V<sub>N</sub>(x<sub>n+1</sub>) - V<sub>N</sub>(x<sub>n</sub>) + &#8467;(x<sub>n</sub>,u<sub>n</sub>)"
        ),
        xaxis_title=r"$n$",
        yaxis_title=r"$s_n$",
        hovermode="x unified",
        margin=dict(t=120),
    )

    if html_path is not None:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path, include_mathjax='cdn')
        __logger__.info(f"Relaxed DP residual plot saved to {html_path}.")
    else:
        fig.show()

    return fig

def cost_descent(
    dataset: MPCDataset,
    html_path: str = None
) -> go.Figure:
    """Plot cost descent check.

    For each trajectory entry, plots

    $$\Delta V_k = V_{k+1} - V_k$$

    where $V_k$ is the MPC cost-to-go at time step $k$ (taken from the stored
    per-step value function / objective).

    Visual interpretation: values above 0 violate the one-step descent
    inequality $V(x_{k+1}) - V(x_k) \le 0$.
    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories to plot.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The resulting Plotly figure.
    """
    if len(dataset) == 0:
        __logger__.warning("Dataset is empty.")
        return go.Figure()

    fig = go.Figure()

    for entry in dataset:
        traj = entry.trajectory
        id = entry.meta.id

        if traj.V_horizon is None:
            __logger__.warning(f"Entry {id} missing horizon_costs; skipping.")
            continue

        V = traj.V_pred
        deltas = V[:, 1:] - V[:, :-1]
        steps = np.tile(np.arange(deltas.shape[1]), (deltas.shape[0], 1))

        deltas = _plotly_multiline(deltas)
        steps = _plotly_multiline(steps)

        color = COLORS[id % len(COLORS)]

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=deltas,
                mode='lines',
                name=f'Run {id+1} - ΔV',
                line=dict(color=color, width=2),
                legendgroup=f'Run {id+1}',
                showlegend=True
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        )
    )

    fig.update_layout(
        title_text=(
            "Cost to go descent check: "
            "ΔV = V<sub>k+1</sub> - V<sub>k</sub>"
        ),
        xaxis_title=r"$k$",
        yaxis_title=r"$\Delta V_k$",
        hovermode="x unified",
        margin=dict(t=120),
    )

    if html_path is not None:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path, include_mathjax='cdn')
        __logger__.info(f"Cost to go descent plot saved to {html_path}.")
    else:
        fig.show()

    return fig

def roa(
    lyapunov_func: Callable[[np.ndarray], np.ndarray],
    c_level: float,
    bounds: np.ndarray,  # shape (n_points, nx)
    state_indices: List[int] = [0, 1],
    state_labels: Optional[List[str]] = None,
    limits: Optional[List[Tuple[float, float]]] = None,
    resolution: int = 100,
    plot_3d: bool = False,
    html_path: Optional[str] = None
) -> go.Figure:
    
    if len(state_indices) != 2:
        raise ValueError("state_indices must contain exactly 2 indices.")

    idx_x, idx_y = state_indices
    
    if state_labels is None:
        state_labels = [f"State {idx_x}", f"State {idx_y}"]
    
    if limits is None:
        limits = [(-2.0, 2.0), (-2.0, 2.0)]

    # Grid for Lyapunov function
    x_vec = np.linspace(limits[0][0], limits[0][1], resolution)
    y_vec = np.linspace(limits[1][0], limits[1][1], resolution)
    X, Y = np.meshgrid(x_vec, y_vec)
    
    dim_x = max(max(state_indices) + 1, bounds.shape[1])
    points_2d = np.vstack([X.ravel(), Y.ravel()]).T
    full_points = np.zeros((points_2d.shape[0], dim_x))
    full_points[:, idx_x] = points_2d[:, 0]
    full_points[:, idx_y] = points_2d[:, 1]
    
    Z_flat = lyapunov_func(full_points)
    Z = Z_flat.reshape(X.shape)

    fig = go.Figure()

    # --- Lyapunov Function V(x) ---
    if plot_3d:
        fig.add_trace(go.Surface(
            z=Z, x=x_vec, y=y_vec,
            colorscale='Viridis', opacity=0.8,
            name='Lyapunov Surface', showlegend=True,
            colorbar=dict(title="V(x)", x=-0.1)
        ))
    else:
        fig.add_trace(go.Contour(
            z=Z, x=x_vec, y=y_vec,
            colorscale='Viridis', name='V(x) Contours',
            opacity=0.6, contours=dict(showlabels=True),
            showlegend=True
        ))

    # --- ROA Boundary (Scatter/Line) ---
    b_x = bounds[:, idx_x]
    b_y = bounds[:, idx_y]
    # For 3D, we need to calculate V(x) for the bounds (should be approximately c_level)
    b_z = np.full(b_x.shape, c_level + 0.01) # Slightly offset for visibility
    if plot_3d:
        # Points as red line/markers in 3D space
        fig.add_trace(go.Scatter3d(
            x=b_x, y=b_y, z=b_z,
            mode='lines+markers',
            marker=dict(size=3, color='red'),
            line=dict(color='red', width=4),
            name=f'ROA Boundary (c={c_level:.2f})'
        ))
        # Optional: Transparent plane for better orientation
        z_plane = np.full_like(Z, c_level)
        fig.add_trace(go.Surface(
            z=z_plane, x=x_vec, y=y_vec,
            colorscale=[[0, 'rgba(255,0,0,0.2)'], [1, 'rgba(255,0,0,0.2)']],
            showscale=False, name='Safety Level', showlegend=False
        ))
    else:
        # In 2D plot the exact points of the ellipse
        fig.add_trace(go.Scatter(
            x=b_x, y=b_y,
            mode='lines', # 'lines' closes the ellipse nicer than 'markers'
            line=dict(color='red', width=3, dash='dash'),
            fill='toself', # Optional: Slightly fill the area
            fillcolor='rgba(255,0,0,0.1)',
            name=f'ROA Boundary (c={c_level:.2f})'
        ))

    # --- Layout ---
    layout_args = dict(
        title=f"Stability Verification: ROA for c={c_level:.2f}",
        legend=dict(x=1.05, y=1),
        autosize=True,
        margin=dict(l=0, r=50, b=0, t=50)
    )
    
    if plot_3d:
        layout_args['scene'] = dict(
            xaxis_title=state_labels[0],
            yaxis_title=state_labels[1],
            zaxis_title="V(x)"
        )
    else:
        layout_args['xaxis_title'] = state_labels[0]
        layout_args['yaxis_title'] = state_labels[1]
        layout_args['yaxis'] = dict(scaleanchor="x", scaleratio=1)

    fig.update_layout(**layout_args)

    if html_path:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path)
    
    return fig