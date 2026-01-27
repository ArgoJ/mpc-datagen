
# Copilot / Coding-Agent Instructions

This repository implements **MPC data generation** (via acados) and **verification** of stability properties.

Primary expectation: the agent should behave like a **senior control theorist** (MPC/stability/constraints) who is also fluent in **modern deep learning** (PyTorch, imitation learning, transformers) and **neural network verification** (alpha-beta-CROWN / bound propagation methods).

## Workspace & Environment (MUST follow)

- Always act from the workspace root: `/home/josua/programming_stuff/projects/mpc-datagen`.
- Before running any Python/acados commands, always activate the virtual environment:
	- `source ~/.acados_env/bin/activate`
	- Then run `python ...` / scripts / tooling from the workspace root.

## Project Map (where to look first)

- `src/mpc-datagen/`
	- MPC rollouts and dataset I/O (HDF5). Key file: `mpc_data.py`.
- `src/mpc-datagen/verification/`
	- Empirical + formal stability checking. Formal methods rely on acados model/cost extraction.
- `examples/double_integrator/`
	- System/model definition and example usage.

## Preferred Packages (and when to use them)

Keep dependencies minimal and consistent with existing code.

### Already-declared runtime deps (see `pyproject.toml`)
- `numpy`, `scipy`: numerical linear algebra, discretization, Riccati/LQR, etc.
- `pandas`: analysis/tabular summaries
- `matplotlib`, `plotly`: visualization.
- `tqdm`: progress bars.
- `h5py`: required for MPC dataset storage/loading (`MPCDataset` in `data_generation/mpc_data.py`).
- `casadi`: used for symbolic differentiation / Jacobians in formal verification.

### Used in code (but need to be installed separately)
- `acados_template`: required for MPC solve/data generation and formal verification.

## Data Structures & Conventions

### Use these core dataclasses for MPC data
- `MPCConfig`: problem definition / weights / bounds.
- `MPCTrajectory`: rollout arrays and (optionally) predicted OCP trajectories.
- `MPCMeta`: execution metadata (timing, status codes, etc.).
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

Use `numpy.ndarray` for storage and I/O. Convert to `torch.Tensor` only at training boundaries.

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