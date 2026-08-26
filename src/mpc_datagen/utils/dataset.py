"""Dataset comparison, initial state matching, and error dataset generation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..mpc_data import MPCConfig, MPCData, MPCDataset, MPCMeta, MPCTrajectory


__logger__ = logging.getLogger(__name__)


@dataclass
class MatchedStatePair:
    """Pair of matched dataset entries sharing the same initial state.

    Attributes
    ----------
    idx_a : int
        Index of entry in dataset A (0-based positional index in dataset).
    idx_b : int
        Index of entry in dataset B (0-based positional index in dataset).
    id_a : int
        Metadata ID of entry in dataset A (from entry.meta.id).
    id_b : int
        Metadata ID of entry in dataset B (from entry.meta.id).
    x0_a : NDArray
        Initial state vector from dataset A. Shape: (nx,).
    x0_b : NDArray
        Initial state vector from dataset B. Shape: (nx,).
    diff_norm : float
        Euclidean norm of initial state difference: ||x0_a - x0_b||_2.
    """

    idx_a: int
    idx_b: int
    id_a: int
    id_b: int
    x0_a: NDArray
    x0_b: NDArray
    diff_norm: float


def get_initial_states(
    dataset: MPCDataset | Sequence[MPCData],
) -> list[tuple[int, int, NDArray]]:
    """Extract initial states for all entries in a dataset.

    Parameters
    ----------
    dataset : MPCDataset | Sequence[MPCData]
        The dataset or sequence of MPCData entries.

    Returns
    -------
    list[tuple[int, int, NDArray]]
        List of tuples (dataset_index, meta_id, x0) for each entry in the dataset.
    """
    results: list[tuple[int, int, NDArray]] = []
    for idx, entry in enumerate(dataset):
        if entry.trajectory.states is None or entry.trajectory.states.shape[0] == 0:
            __logger__.warning(f"Entry at index {idx} has no states. Skipping.")
            continue
        x0 = np.asarray(entry.trajectory.states[0], dtype=float)
        meta_id = int(getattr(entry.meta, "id", idx))
        results.append((idx, meta_id, x0))
    return results


def find_matching_initial_states(
    dataset_a: MPCDataset | Sequence[MPCData],
    dataset_b: MPCDataset | Sequence[MPCData],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> list[MatchedStatePair]:
    """Find pairs of entries between two datasets with matching initial states.

    For each entry in `dataset_a`, searches `dataset_b` for an entry whose initial
    state $x_0$ satisfies `np.isclose(x0_a, x0_b, atol=atol, rtol=rtol)`.
    If multiple matches are found, the closest match (minimal Euclidean distance)
    is selected.

    Parameters
    ----------
    dataset_a : MPCDataset | Sequence[MPCData]
        First dataset (e.g. evaluation or learned policy rollouts).
    dataset_b : MPCDataset | Sequence[MPCData]
        Second dataset (e.g. reference or MPC expert rollouts).
    atol : float, optional
        Absolute tolerance for initial state comparison. Default is 1e-6.
    rtol : float, optional
        Relative tolerance for initial state comparison. Default is 1e-5.

    Returns
    -------
    list[MatchedStatePair]
        List of matched pairs containing dataset indices, IDs, initial states, and error norms.
    """
    states_a = get_initial_states(dataset_a)
    states_b = get_initial_states(dataset_b)

    if not states_a or not states_b:
        __logger__.warning("One or both datasets contain no initial states.")
        return []

    matched_pairs: list[MatchedStatePair] = []
    matched_b_indices: set[int] = set()

    for idx_a, id_a, x0_a in states_a:
        best_match: tuple[int, int, NDArray, float] | None = None
        min_diff = float("inf")

        for idx_b, id_b, x0_b in states_b:
            if x0_a.shape != x0_b.shape:
                continue

            if np.allclose(x0_a, x0_b, atol=atol, rtol=rtol):
                diff = float(np.linalg.norm(x0_a - x0_b))
                if diff < min_diff:
                    min_diff = diff
                    best_match = (idx_b, id_b, x0_b, diff)

        if best_match is not None:
            idx_b, id_b, x0_b, diff = best_match
            if idx_b in matched_b_indices:
                __logger__.debug(
                    f"Dataset B index {idx_b} matched multiple times. Assigned to Dataset A index {idx_a}."
                )
            matched_b_indices.add(idx_b)
            matched_pairs.append(
                MatchedStatePair(
                    idx_a=idx_a,
                    idx_b=idx_b,
                    id_a=id_a,
                    id_b=id_b,
                    x0_a=x0_a,
                    x0_b=x0_b,
                    diff_norm=diff,
                )
            )
        else:
            __logger__.debug(f"No matching initial state found in dataset B for dataset A index {idx_a} (x0={x0_a}).")

    __logger__.info(
        f"Found {len(matched_pairs)} matching initial state pairs between dataset A ({len(dataset_a)} entries) "
        f"and dataset B ({len(dataset_b)} entries)."
    )
    return matched_pairs


def compute_entry_error(
    entry_a: MPCData,
    entry_b: MPCData,
    relative: bool = False,
    eps: float = 1e-8,
    error_id: int | None = None,
) -> MPCData:
    """Compute the error trajectory (entry_a - entry_b) between two MPCData entries.

    Parameters
    ----------
    entry_a : MPCData
        First entry (e.g. actual / test rollout).
    entry_b : MPCData
        Second entry (e.g. nominal / expert rollout).
    relative : bool, optional
        If True, computes relative error (x_a - x_b) / (|x_b| + eps). Default is False.
    eps : float, optional
        Small epsilon used in relative error denominator to prevent division by zero. Default is 1e-8.
    error_id : int, optional
        ID to assign to the resulting error entry. If None, inherits `entry_a.meta.id`.

    Returns
    -------
    MPCData
        New MPCData entry containing the error trajectory.
    """
    from ..mpc_data import MPCConfig, MPCData, MPCMeta, MPCTrajectory

    traj_a = entry_a.trajectory
    traj_b = entry_b.trajectory

    # Determine common simulated length
    min_steps = min(traj_a.states.shape[0], traj_b.states.shape[0])
    if traj_a.states.shape[0] != traj_b.states.shape[0]:
        __logger__.debug(
            f"State trajectory lengths differ ({traj_a.states.shape[0]} vs {traj_b.states.shape[0]}). "
            f"Truncating error computation to {min_steps} steps."
        )

    # State error: (min_steps, nx)
    states_a = traj_a.states[:min_steps]
    states_b = traj_b.states[:min_steps]
    diff_states = states_a - states_b
    if relative:
        denom = np.abs(states_b) + eps
        err_states = diff_states / denom
    else:
        err_states = diff_states

    # Input error: (min_ctrl_steps, nu) if available
    if (
        traj_a.inputs is not None
        and traj_b.inputs is not None
        and traj_a.inputs.ndim == 2
        and traj_b.inputs.ndim == 2
        and traj_a.inputs.shape[1] == traj_b.inputs.shape[1]
    ):
        min_ctrl = min(traj_a.inputs.shape[0], traj_b.inputs.shape[0])
        inputs_a = traj_a.inputs[:min_ctrl]
        inputs_b = traj_b.inputs[:min_ctrl]
        diff_inputs = inputs_a - inputs_b
        if relative:
            denom_u = np.abs(inputs_b) + eps
            err_inputs = diff_inputs / denom_u
        else:
            err_inputs = diff_inputs
    else:
        err_inputs = np.empty((0, 0), dtype=float)

    # Timestamps
    times = traj_a.times[:min_steps] if traj_a.times is not None else np.arange(min_steps, dtype=float)

    # Value function / costs
    min_v = min(
        traj_a.V_solver.shape[0] if traj_a.V_solver is not None else 0,
        traj_b.V_solver.shape[0] if traj_b.V_solver is not None else 0,
    )
    if min_v > 0:
        err_V_solver = traj_a.V_solver[:min_v] - traj_b.V_solver[:min_v]
    else:
        err_V_solver = np.zeros(max(0, min_steps - 1), dtype=float)

    err_V_N: NDArray | None = None
    if traj_a.V_N is not None and traj_b.V_N is not None:
        min_vn = min(traj_a.V_N.shape[0], traj_b.V_N.shape[0])
        err_V_N = traj_a.V_N[:min_vn] - traj_b.V_N[:min_vn]

    err_V_horizon: NDArray | None = None
    if (
        traj_a.V_horizon is not None
        and traj_b.V_horizon is not None
        and traj_a.V_horizon.ndim == 2
        and traj_b.V_horizon.ndim == 2
        and traj_a.V_horizon.shape[1] == traj_b.V_horizon.shape[1]
    ):
        min_vh = min(traj_a.V_horizon.shape[0], traj_b.V_horizon.shape[0])
        err_V_horizon = traj_a.V_horizon[:min_vh, :] - traj_b.V_horizon[:min_vh, :]

    # Predicted states & inputs
    err_pred_states: NDArray | None = None
    if (
        traj_a.predicted_states is not None
        and traj_b.predicted_states is not None
        and traj_a.predicted_states.ndim == 3
        and traj_b.predicted_states.ndim == 3
        and traj_a.predicted_states.shape[1:] == traj_b.predicted_states.shape[1:]
    ):
        min_ps = min(traj_a.predicted_states.shape[0], traj_b.predicted_states.shape[0])
        err_pred_states = traj_a.predicted_states[:min_ps] - traj_b.predicted_states[:min_ps]

    err_pred_inputs: NDArray | None = None
    if (
        traj_a.predicted_inputs is not None
        and traj_b.predicted_inputs is not None
        and traj_a.predicted_inputs.ndim == 3
        and traj_b.predicted_inputs.ndim == 3
        and traj_a.predicted_inputs.shape[1:] == traj_b.predicted_inputs.shape[1:]
    ):
        min_pu = min(traj_a.predicted_inputs.shape[0], traj_b.predicted_inputs.shape[0])
        err_pred_inputs = traj_a.predicted_inputs[:min_pu] - traj_b.predicted_inputs[:min_pu]

    err_traj = MPCTrajectory(
        states=err_states,
        inputs=err_inputs,
        times=times,
        V_solver=err_V_solver,
        V_N=err_V_N,
        V_horizon=err_V_horizon,
        predicted_states=err_pred_states,
        predicted_inputs=err_pred_inputs,
    )

    # Metadata
    status_codes = list(entry_a.meta.status_codes[: max(0, min_steps - 1)]) if entry_a.meta.status_codes else []
    assigned_id = error_id if error_id is not None else int(getattr(entry_a.meta, "id", 0))
    meta = MPCMeta(
        id=assigned_id,
        timestamp=entry_a.meta.timestamp,
        solve_time_mean=entry_a.meta.solve_time_mean,
        solve_time_max=entry_a.meta.solve_time_max,
        solve_time_total=entry_a.meta.solve_time_total,
        sim_duration_wall=entry_a.meta.sim_duration_wall,
        steps_simulated=max(0, min_steps - 1),
        status_codes=status_codes,
        feasible=bool(entry_a.meta.feasible and entry_b.meta.feasible),
    )

    # Config (shallow copy config with adjusted T_sim)
    cfg = entry_a.config
    err_cfg = MPCConfig(
        T_sim=max(0, min_steps - 1),
        dt=cfg.dt,
        N=cfg.N,
        nx=cfg.nx,
        nu=cfg.nu,
        cost=cfg.cost,
        constraints=cfg.constraints,
        model=cfg.model,
    )

    return MPCData(trajectory=err_traj, meta=meta, config=err_cfg)


def create_error_dataset(
    dataset_a: MPCDataset | Sequence[MPCData],
    dataset_b: MPCDataset | Sequence[MPCData],
    atol: float = 1e-6,
    rtol: float = 1e-5,
    relative: bool = False,
    eps: float = 1e-8,
    file_path: Path | str | None = None,
) -> MPCDataset:
    """Create an error dataset by matching initial states between two datasets.

    Searches for matching initial states between `dataset_a` and `dataset_b`,
    computes the trajectory error (`entry_a - entry_b`) for each matched pair,
    and returns a new `MPCDataset` containing the resulting error trajectories.

    Parameters
    ----------
    dataset_a : MPCDataset | Sequence[MPCData]
        First dataset (e.g. test or neural network controller rollouts).
    dataset_b : MPCDataset | Sequence[MPCData]
        Second dataset (e.g. reference MPC expert rollouts).
    atol : float, optional
        Absolute tolerance for matching initial states. Default is 1e-6.
    rtol : float, optional
        Relative tolerance for matching initial states. Default is 1e-5.
    relative : bool, optional
        If True, computes relative errors. Default is False.
    eps : float, optional
        Epsilon for relative error calculation. Default is 1e-8.
    file_path : Path | str | None, optional
        If provided, saves the resulting dataset to this HDF5 file path.

    Returns
    -------
    MPCDataset
        Dataset containing the error trajectories for all matched initial states.
    """
    from ..mpc_data import MPCData, MPCDataset

    matched_pairs = find_matching_initial_states(dataset_a, dataset_b, atol=atol, rtol=rtol)

    if not matched_pairs:
        __logger__.warning("No matching initial states found between the two datasets.")
        return MPCDataset()

    error_entries: list[MPCData] = []
    for new_idx, pair in enumerate(matched_pairs):
        entry_a = dataset_a[pair.idx_a]
        entry_b = dataset_b[pair.idx_b]
        err_entry = compute_entry_error(
            entry_a=entry_a,
            entry_b=entry_b,
            relative=relative,
            eps=eps,
            error_id=new_idx,
        )
        error_entries.append(err_entry)

    error_dataset = MPCDataset(data_buffer=error_entries)

    if file_path is not None:
        error_dataset.save(Path(file_path))

    return error_dataset



# Convenient aliases
compute_error_dataset = create_error_dataset
match_initial_states = find_matching_initial_states
extract_initial_states = get_initial_states
