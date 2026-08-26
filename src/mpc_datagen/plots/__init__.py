import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..mpc_data import MPCDataset
from . import utils
from .descent import cost_descent, relaxed_dp_residual
from .lyapunov import create_lyapunov_from_dataset, lyapunov
from .trajectories import mpc_trajectories, trajectories, trajectory_error_bands, error_bands, error_band
from .utils import COLORS, PairPlotResult, _infer_state_limits

__logger__ = logging.getLogger(__name__)


def all(
    dataset: MPCDataset,
    state_labels: list[str] | None = None,
    control_labels: list[str] | None = None,
    limits: list[tuple[float, float]] | None = None,
    lyapunov_limits: list[tuple[float, float]] | None = None,
    base_path: Path | str | None = None,
    resolution: int = 100,
    plot_3d: bool = False,
    limit_pad_ratio: float = 0.1,
    limit_min_pad: float = 1e-12,

    # Trajectory specific
    time_bound: float | None = None,
    plot_predictions: bool = False,

    # Cost descent specific
    use_optimal_v: bool = False,

    # Relaxed DP specific
    alpha: float = 1.0,

    # Lyapunov specific
    lyapunov_func: Callable[[NDArray], NDArray] | None = None,
    lyap_state_indices: list[int] | None = None,
    lyap_use_dataset_v: bool = True,
    lyap_use_optimal_v: bool | None = None,
    lyap_fill_nearest: bool = False,
    roa_lyapunov_func: Callable[[NDArray], NDArray] | None = None,
    c_level: float | None = None,
    roa_nx: int | None = None,
) -> None:
    """Convenience function to plot trajectories, residuals, Lyapunov landscapes, and ROA together.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories to plot.
    state_labels : list[str], optional
        Labels for state variables.
    control_labels : list[str], optional
        Labels for control variables.
    limits : list of tuples, optional
        General limits for state axes.
    lyapunov_limits : list of tuples, optional
        Specific limits for Lyapunov state axes.
    base_path : Path | str, optional
        Base file path for saving generated HTML plots.
    resolution : int, optional
        Grid resolution for Lyapunov function contour/surface plots. Default is 100.
    plot_3d : bool, optional
        If True, plots 3D Lyapunov surfaces and 3D trajectory lines. Default is False.
    limit_pad_ratio : float, optional
        Padding factor applied when inferring state limits. Default is 0.1.
    limit_min_pad : float, optional
        Minimum padding for inferred limits. Default is 1e-12.
    time_bound : float, optional
        Time bound limit for trajectory plots.
    plot_predictions : bool, optional
        If True, plots open-loop predictions in trajectory plots.
    use_optimal_v : bool, optional
        If True, uses V_N in cost descent plot.
    alpha : float, optional
        Relaxation factor for relaxed DP residual plot.
    lyapunov_func : Callable[[NDArray], NDArray], optional
        Lyapunov function to evaluate for landscape. If None and dataset is present,
        V_N from the dataset is used.
    lyap_state_indices : list[int], optional
        State indices to plot for Lyapunov pairs.
    lyap_use_dataset_v : bool, optional
        If True, uses dataset's optimal value function V_N for trajectory z-coordinates.
    lyap_use_optimal_v : bool, optional
        Alias for lyap_use_dataset_v.
    lyap_fill_nearest : bool, optional
        If True, fills NaNs outside convex hull with nearest-neighbor extrapolation.
        Default is False (preserves NaNs).
    roa_lyapunov_func : Callable[[NDArray], NDArray], optional
        Separate Lyapunov function for ROA level set computation if different from lyapunov_func.
    c_level : float, optional
        Sublevel set level c for ROA.
    roa_nx : int, optional
        Dimension of state space for ROA calculations.
    """
    if lyap_use_optimal_v is not None:
        lyap_use_dataset_v = lyap_use_optimal_v

    lyap_func = lyapunov_func if lyapunov_func is not None else roa_lyapunov_func

    all_states = np.vstack([entry.trajectory.states for entry in dataset]) if len(dataset) > 0 else None
    auto_limits = None
    if all_states is not None:
        auto_limits = _infer_state_limits(
            all_states,
            pad_ratio=limit_pad_ratio,
            min_pad=limit_min_pad,
        )

    lyapunov_plot_limits = (
        lyapunov_limits
        if lyapunov_limits is not None
        else limits if limits is not None else auto_limits
    )

    if state_labels is not None and control_labels is not None:
        mpc_trajectories(
            dataset=dataset,
            state_labels=state_labels,
            control_labels=control_labels,
            plot_predictions=plot_predictions,
            time_bound=time_bound,
            html_path=Path(f"{base_path}_trajectories.html") if base_path else None,
        )

    cost_descent_path = Path(f"{base_path}_cost_descent.html") if base_path else None
    cost_descent(dataset=dataset, html_path=cost_descent_path, use_optimal_v=use_optimal_v)

    relaxed_dp_path = Path(f"{base_path}_relaxed_dp.html") if base_path else None
    relaxed_dp_residual(dataset=dataset, html_path=relaxed_dp_path, alpha=alpha)

    lyapunov_path = Path(f"{base_path}_lyapunov.html") if base_path else None
    lyapunov(
        lyapunov_func=lyap_func,
        dataset=dataset,
        roa_level=c_level,
        state_indices=lyap_state_indices,
        state_labels=state_labels,
        limits=lyapunov_plot_limits,
        resolution=resolution,
        plot_3d=plot_3d,
        html_path=lyapunov_path,
        use_dataset_v=lyap_use_dataset_v,
        fill_nearest=lyap_fill_nearest,
    )


__all__ = [
    # Top-level API functions
    "all",
    "lyapunov",
    "create_lyapunov_from_dataset",
    "mpc_trajectories",
    "trajectories",
    "trajectory_error_bands",
    "error_bands",
    "error_band",
    "relaxed_dp_residual",
    "cost_descent",

    # Data structures & constants
    "PairPlotResult",
    "COLORS",

    # Submodules
    "utils",
]
