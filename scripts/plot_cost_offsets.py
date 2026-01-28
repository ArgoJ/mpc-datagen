import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from mpc_datagen.mpc_data import MPCDataset, MPCTrajectory, MPCData


def _ensure_costs(entry: MPCData) -> Optional[np.ndarray]:
    traj = entry.trajectory
    if traj.costs is not None and traj.horizon_costs is not None:
        return traj.costs

    # Attempt recompute if predictions are available
    if traj.predicted_states is None or traj.predicted_inputs is None:
        return None

    traj.recalculate_costs(entry.config.cost)
    return traj.costs


def _ensure_scaled_costs(entry: MPCData) -> Optional[np.ndarray]:
    traj = entry.trajectory
    if traj.horizon_costs is None:
        return None

    return traj.get_scaled_costs(
        stage_scale=entry.config.cost.stage_scale,
        terminal_scale=entry.config.cost.terminal_scale,
    )


def _plot_costs(traj: MPCTrajectory, costs: Optional[np.ndarray], scaled_costs: Optional[np.ndarray]) -> None:
    solver_costs = traj.solver_costs
    t = np.arange(solver_costs.shape[0])

    fig, (ax_costs, ax_offsets) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    ax_costs.plot(t, solver_costs, label="solver_costs", linewidth=2.5, linestyle=":", alpha=0.8)
    if costs is not None:
        ax_costs.plot(t, costs, label="costs", linestyle="-.", linewidth=2.5, alpha=0.8)
    if scaled_costs is not None:
        ax_costs.plot(t, scaled_costs, label="scaled_costs", linestyle="--", linewidth=2.5, alpha=0.8)
    ax_costs.set_ylabel("cost")
    ax_costs.grid(True, alpha=0.3)
    ax_costs.legend()

    if costs is not None:
        ax_offsets.plot(t, solver_costs - costs, label="solver - costs", linestyle="-.", linewidth=2.5, alpha=0.8)
    if scaled_costs is not None:
        ax_offsets.plot(t, solver_costs - scaled_costs, label="solver - scaled", linestyle=":", linewidth=2.5, alpha=0.8)
    ax_offsets.axhline(0.0, color="black", linewidth=1.5, alpha=0.5, linestyle="--")
    ax_offsets.set_xlabel("timestep")
    ax_offsets.set_ylabel("offset")
    ax_offsets.grid(True, alpha=0.3)
    ax_offsets.legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    # parser = argparse.ArgumentParser(description="Plot solver/cost offsets for an MPC dataset entry.")
    # parser.add_argument("dataset", type=Path, help="Path to the HDF5 dataset")
    # parser.add_argument("--index", type=int, default=0, help="Trajectory index to plot")
    # args = parser.parse_args()

    file = "/home/josua/programming_stuff/projects/mpc-datagen/data/double_integrator_none_N40_data"
    dataset = MPCDataset.load(file)
    idx = 0
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {file}")

    if idx < 0 or idx >= len(dataset):
        raise IndexError(f"Index {idx} out of range [0, {len(dataset) - 1}]")

    entry = dataset[idx]
    costs = _ensure_costs(entry)
    scaled_costs = _ensure_scaled_costs(entry)

    _plot_costs(entry.trajectory, costs, scaled_costs)


if __name__ == "__main__":
    main()
