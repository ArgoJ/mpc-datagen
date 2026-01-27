# mpc-datagen

Generate datasets from Model Predictive Control (MPC) runs, with optional stability verification for linear-quadratic MPC setups. This package targets workflows such as imitation learning, controller benchmarking, and reproducible MPC data collection using acados.

## What you can do

- **Generate closed-loop MPC rollouts** from an acados OCP solver.
- **Extract MPC configuration and linearized system data** from an acados solver instance.
- **Store trajectories and metadata** in structured containers for downstream learning or analysis.
- **Verify stability properties** (e.g., Grüne-type certificates and terminal-ingredient checks) for linear MPC setups.
- **Log progress cleanly** with tqdm-friendly logging utilities.

## Package layout

- [src/mpc-datagen](src/mpc-datagen)
	- Core data models: MPCData, MPCDataset, MPCConfig, MPCTrajectory
	- Extraction utilities: MPCConfigExtractor, LinearSystemExtractor
	- Generation: MPCDataGenerator, solve_mpc_closed_loop
	- Verification: StabilityVerifier, StabilityCertifier, reports, rendering

## Installation

```bash
pip install -e .
```

## Quick start

```python
from acados_template import AcadosOcpSolver
from mpc_datagen import MPCDataGenerator

# solver: AcadosOcpSolver created elsewhere
samples = 10000
x0_bounds = np.array([])
T_sim = 50
gen = MPCDataGenerator(solver, x0_bounds, T_sim)
data = gen.generate(samples)
```

## Dataset datastructure

Datasets are stored with h5py and mirror the in-memory containers in `MPCData`, `MPCTrajectory`, `MPCMeta`, and `MPCConfig`.  
The dataset can be indexed or sliced:

```python
from mpc_datagen import MPCDataset

dataset = MPCDataset.load("your/path/to/file.h5py")

for entry in dataset:
    print(entry.meta.id)

print(dataset[0:20:2])
```

Layout of one entry:

- entry : *MPCData*
    - `trajectory` : *MPCTrajectory*
        - `x`: state trajectory, shape $(T_{sim}+1, n_x)$
        - `u`: input trajectory, shape $(T_{sim}, n_u)$
        - `predicted_x`: reference states (optional), shape $(T_{sim}, N+1, n_x)$
        - `predicted_u`: reference inputs (optional), shape $(T_{sim}, N, n_u)$
        - `status`: solver status per step, shape $(T_{sim},)$
        - `cost`: stage cost per step, shape $(T_{sim},)$
    - `meta`: *MPCMeta*
        - `solver_stats`: timing/iterations (if available)
        - `timestamp`: creation time
    - `config` : *MPCConfig*
        - `N`, `dt`, `T_sim`, `nx`, `nu`
        - `x0` (optional)

## Verification utilities (linear MPC)

### Lyapunov Decrease

...

### Grüne Condition

$$N_{req}(\gamma) \gt 2 + \frac{\log(\gamma - 1)}{\log(1 + \frac{1}{\gamma - 1})}$$

$$\alpha_N = 1 - \frac{1}{(\gamma / (\gamma - 1))^N - 1}$$


## Closed-loop solve

```python
from mpc_datagen import solve_mpc_closed_loop

data = solve_mpc_closed_loop(solver, T_sim=100)
```

## Logging

```python
from mpc_datagen.package_logger import PackageLogger

PackageLogger.setup()
```

## Notes

- The package assumes an acados installation for solver-related features.
- Verification functions focus on linear-quadratic MPC conditions and are not general nonlinear certificates.