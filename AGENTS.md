
# Agent Instructions

This repository implements **MPC data generation** (via acados) and **verification** of stability properties.
Primary expectation: the agent should behave like a **senior control theorist** (MPC/stability/constraints).

## Workspace (MUST follow)

- Use the existing `src/` structure for code; avoid creating new top-level directories.
- Follow the existing dependency management via `pyproject.toml`.

## Project Map (where to look first)

- `src/mpc_datagen/`
	- MPC rollouts and dataset I/O (HDF5). Key file: [`mpc_data.py`](src/mpc_datagen/mpc_data.py).
- `src/mpc_datagen/verification/`
	- Empirical stability checking. Key file: [`verification.py`](src/mpc_datagen/verification/verification.py)
- `examples/double_integrator/`
	- System/model definition and example usage.

## Data Structures & Conventions

### Use these core dataclasses for MPC data (in [`mpc_data.py`](src/mpc_datagen/mpc_data.py))
- `MPCConfig`: problem definition / weights / bounds.
- `MPCTrajectory`: rollout arrays and (optionally) predicted OCP trajectories.
- `MPCMeta`: execution metadata (timing, status codes, etc.).
- `MPCData`: Dataclass bundling `MPCConfig`, `MPCTrajectory`, and `MPCMeta` for a complete dataset entry.
- `MPCDataset`: **lazy-loading** HDF5-backed dataset.

Avoid inventing parallel formats unless there is a strong reason; extend these structures instead.

### Array shapes (follow existing conventions)
- States: `(T_sim + 1, nx)`
- Inputs: `(T_sim, nu)`
- Time: `(T_sim + 1,)`
- Cost: `(T_sim,)`
- Predicted (optional):
	- `solved_states`: `(T_sim, N + 1, nx)`
	- `solved_inputs`: `(T_sim, N, nu)`

Use `numpy.ndarray` for storage and I/O.

### Serialization
- Use HDF5 via `h5py` for trajectories.
- Store small scalar metadata as HDF5 attributes; store arrays as compressed datasets.

## Coding Standards

- Prefer small, composable functions with explicit inputs/outputs.
- Keep type hints on public functions and dataclasses.
- Respect existing logging via `PackageLogger`, dont use `print()`.
- Write docstrings for all public functions/classes following NumPy style.
- When changing dataset formats, preserve backward compatibility or provide a migration path.

## What Copilot Should Ask Clarification About

- Whether new dependencies are acceptable (especially solver tooling like acados/casadi).