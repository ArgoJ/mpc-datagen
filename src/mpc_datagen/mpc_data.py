import json
import h5py
import numpy as np
import pandas as pd

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Iterator
from pathlib import Path

from .package_logger import PackageLogger

__logger__ = PackageLogger.get_logger(__name__)

def _is_defined_array(arr: Optional[np.ndarray]) -> bool:
    """Check if an array is defined and non-empty."""
    if arr is None:
        return False
    arr = np.asarray(arr)
    if arr.size == 0:
        return False
    return True 

@dataclass
class MPCMeta:
    """Metadata regarding the MPC execution."""
    id: int = -1
    timestamp: str = ""
    solve_time_mean: float = 0.0
    solve_time_max: float = 0.0
    solve_time_total: float = 0.0
    sim_duration_wall: float = 0.0
    steps_simulated: int = 0
    status_codes: List[int] = field(default_factory=list)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "MPCMeta":
        """Load metadata from JSON attribute in the provided group."""
        meta_json = grp.attrs.get("meta_json", "{}")
        meta_dict = json.loads(meta_json)
        return cls(**meta_dict)

    def to_hdf5(self, grp: h5py.Group) -> None:
        """Save metadata as JSON attribute in the provided group."""
        grp.attrs["meta_json"] = json.dumps(asdict(self))

@dataclass
class LinearSystem:
    """Linearized system matrices."""
    A: np.ndarray = field(default_factory=lambda: np.array([[]]))
    B: np.ndarray = field(default_factory=lambda: np.array([[]]))
    gd: np.ndarray = field(default_factory=lambda: np.array([[]]))

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "LinearSystem":
        """Load linear system matrices from a trajectory group (expects a `linear_system` subgroup)."""
        lin_sys_grp = grp.get("linear_system", None)
        if lin_sys_grp is None:
            raise ValueError("No 'linear_system' group found in the provided HDF5 group.")

        return cls(
            A=lin_sys_grp["A"][:] if "A" in lin_sys_grp else np.array([]),
            B=lin_sys_grp["B"][:] if "B" in lin_sys_grp else np.array([]),
            gd=lin_sys_grp["gd"][:] if "gd" in lin_sys_grp else np.array([]),
        )

    def to_hdf5(self, grp: h5py.Group) -> None:
        """Save linear system matrices to a trajectory group (creates a `linear_system` subgroup)."""
        lin_sys_grp = grp.create_group("linear_system")
        lin_sys_grp.create_dataset("A", data=self.A, compression="gzip")
        lin_sys_grp.create_dataset("B", data=self.B, compression="gzip")
        lin_sys_grp.create_dataset("gd", data=self.gd, compression="gzip") 

@dataclass
class LinearLSCost:
    """Linearized system matrices."""
    Vx: np.ndarray = field(default_factory=lambda: np.array([[]]))
    Vu: np.ndarray = field(default_factory=lambda: np.array([[]]))
    W: np.ndarray = field(default_factory=lambda: np.array([[]]))
    yref: np.ndarray = field(default_factory=lambda: np.array([[]]))
    Vx_e: np.ndarray = field(default_factory=lambda: np.array([[]]))
    W_e: np.ndarray = field(default_factory=lambda: np.array([[]]))
    yref_e: np.ndarray = field(default_factory=lambda: np.array([[]]))

    def has_terminal_cost(self) -> bool:
        """Check if terminal cost matrices are defined."""
        return _is_defined_array(self.Vx_e) and _is_defined_array(self.W_e)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "LinearLSCost":
        """Load cost matrices from a trajectory group (expects a `cost` subgroup)."""
        cost_grp = grp.get("cost", None)
        if cost_grp is None:
            raise ValueError("No 'cost' group found in the provided HDF5 group.")

        return cls(
            Vx=cost_grp["Vx"][:] if "Vx" in cost_grp else np.array([]),
            Vu=cost_grp["Vu"][:] if "Vu" in cost_grp else np.array([]),
            W=cost_grp["W"][:] if "W" in cost_grp else np.array([]),
            yref=cost_grp["yref"][:] if "yref" in cost_grp else np.array([]),
            Vx_e=cost_grp["Vx_e"][:] if "Vx_e" in cost_grp else np.array([]),
            W_e=cost_grp["W_e"][:] if "W_e" in cost_grp else np.array([]),
            yref_e=cost_grp["yref_e"][:] if "yref_e" in cost_grp else np.array([]),
        )

    def to_hdf5(self, grp: h5py.Group) -> None:
        """Save cost matrices to a trajectory group (creates a `cost` subgroup)."""
        cost_grp = grp.create_group("cost")
        cost_grp.create_dataset("Vx", data=self.Vx, compression="gzip")
        cost_grp.create_dataset("Vu", data=self.Vu, compression="gzip")
        cost_grp.create_dataset("W", data=self.W, compression="gzip")
        cost_grp.create_dataset("yref", data=self.yref, compression="gzip")
        cost_grp.create_dataset("Vx_e", data=self.Vx_e, compression="gzip")
        cost_grp.create_dataset("W_e", data=self.W_e, compression="gzip")
        cost_grp.create_dataset("yref_e", data=self.yref_e, compression="gzip")

@dataclass
class Constraints:
    """Linearized system matrices."""
    x0: np.ndarray = field(default_factory=lambda: np.array([]))
    lbx: np.ndarray = field(default_factory=lambda: np.array([]))
    ubx: np.ndarray = field(default_factory=lambda: np.array([]))
    lbu: np.ndarray = field(default_factory=lambda: np.array([]))
    ubu: np.ndarray = field(default_factory=lambda: np.array([]))
    lbx_e: np.ndarray = field(default_factory=lambda: np.array([]))
    ubx_e: np.ndarray = field(default_factory=lambda: np.array([]))

    def has_state_bounds(self) -> bool:
        """Check if state bounds are defined."""
        return _is_defined_array(self.lbx) and _is_defined_array(self.ubx)

    def has_terminal_state_bounds(self) -> bool:
        """Check if terminal state bounds are defined."""
        return _is_defined_array(self.lbx_e) and _is_defined_array(self.ubx_e)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "Constraints":
        """Load constraints from a trajectory group (expects a `constraints` subgroup)."""
        cons_grp = grp.get("constraints", None)
        if cons_grp is None:
            raise ValueError("No 'constraints' group found in the provided HDF5 group.")

        return cls(
            x0=cons_grp["x0"][:] if "x0" in cons_grp else np.array([]),
            lbx=cons_grp["lbx"][:] if "lbx" in cons_grp else np.array([]),
            ubx=cons_grp["ubx"][:] if "ubx" in cons_grp else np.array([]),
            lbu=cons_grp["lbu"][:] if "lbu" in cons_grp else np.array([]),
            ubu=cons_grp["ubu"][:] if "ubu" in cons_grp else np.array([]),
            lbx_e=cons_grp["lbx_e"][:] if "lbx_e" in cons_grp else np.array([]),
            ubx_e=cons_grp["ubx_e"][:] if "ubx_e" in cons_grp else np.array([]),
        )

    def to_hdf5(self, grp: h5py.Group) -> None:
        """Save constraints to a trajectory group (creates a `constraints` subgroup)."""
        cons_grp = grp.create_group("constraints")
        cons_grp.create_dataset("x0", data=self.x0, compression="gzip")
        cons_grp.create_dataset("lbx", data=self.lbx, compression="gzip")
        cons_grp.create_dataset("ubx", data=self.ubx, compression="gzip")
        cons_grp.create_dataset("lbu", data=self.lbu, compression="gzip")
        cons_grp.create_dataset("ubu", data=self.ubu, compression="gzip")
        cons_grp.create_dataset("lbx_e", data=self.lbx_e, compression="gzip")
        cons_grp.create_dataset("ubx_e", data=self.ubx_e, compression="gzip")

@dataclass
class MPCConfig:
    """Configuration used for the MPC problem."""
    T_sim: int = 0                  # Simulation steps
    N: int = 0                      # Prediction horizon
    nx: int = 0                     # State dimension
    nu: int = 0                     # Input dimension
    dt: float = 0.1                 # Sampling time
    constraints: Constraints = field(default_factory=Constraints)
    model: LinearSystem = field(default_factory=LinearSystem)
    cost: LinearLSCost = field(default_factory=LinearLSCost)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "MPCConfig":
        """Load config from a trajectory group (expects a `config` subgroup)."""
        cfg_grp = grp.get("config", None)
        if cfg_grp is None:
            raise ValueError("No 'config' group found in the provided HDF5 group.")

        return cls(
            T_sim=int(cfg_grp.attrs.get("T_sim", 0)),
            N=int(cfg_grp.attrs.get("N", 10)),
            nx=int(cfg_grp.attrs.get("nx", 2)),
            nu=int(cfg_grp.attrs.get("nu", 1)),
            dt=float(cfg_grp.attrs.get("dt", 0.1)),

            constraints=Constraints.from_hdf5(cfg_grp),
            model=LinearSystem.from_hdf5(cfg_grp),
            cost=LinearLSCost.from_hdf5(cfg_grp),
        )

    def to_hdf5(self, grp: h5py.Group) -> None:
        """Save config to a trajectory group (creates a `config` subgroup)."""
        cfg_grp = grp.create_group("config")
        cfg_grp.attrs["T_sim"] = int(self.T_sim)
        cfg_grp.attrs["N"] = int(self.N)
        cfg_grp.attrs["nx"] = int(self.nx)
        cfg_grp.attrs["nu"] = int(self.nu)
        cfg_grp.attrs["dt"] = float(self.dt)

        self.constraints.to_hdf5(cfg_grp)
        self.model.to_hdf5(cfg_grp)
        self.cost.to_hdf5(cfg_grp)


@dataclass
class MPCTrajectory:
    """The actual data resulting from a run."""
    states: np.ndarray          # (T, nx)
    inputs: np.ndarray          # (T, nu)
    time: np.ndarray            # (T,)
    cost: np.ndarray            # (T,)
    predicted_states: Optional[np.ndarray] = None   # (T, N+1, nx) - OCP predictions at each step
    predicted_inputs: Optional[np.ndarray] = None   # (T, N, nu)   - OCP predictions at each step
    feasible: bool = True
    
    @property
    def length(self) -> int:
        """Get the simulation length in steps."""
        return self.states.shape[0] - 1
    
    @property
    def horizon(self) -> Optional[int]:
        """Get the prediction horizon length (if available)."""
        if self.predicted_states is not None:
            return self.predicted_states.shape[1] - 1
        return None

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "MPCTrajectory":
        """Load trajectory arrays from a trajectory group."""
        traj_grp = grp.get("trajectory", None)
        if traj_grp is None:
            raise ValueError("No 'trajectory' group found in the provided HDF5 group.")
        
        return cls(
            states=traj_grp["states"][:, :],
            inputs=traj_grp["inputs"][:, :],
            time=traj_grp["time"][:],
            cost=traj_grp["cost"][:],
            predicted_states=traj_grp["predicted_states"][:, :, :] if "predicted_states" in traj_grp else None,
            predicted_inputs=traj_grp["predicted_inputs"][:, :, :] if "predicted_inputs" in traj_grp else None,
            feasible=bool(traj_grp.attrs.get("feasible", True)),
        )

    def to_hdf5(self, grp: h5py.Group, save_ocp_trajs: bool = True) -> None:
        """Save trajectory arrays to a trajectory group."""
        traj_grp = grp.create_group("trajectory")
        traj_grp.create_dataset("states", data=self.states, compression="gzip")
        traj_grp.create_dataset("inputs", data=self.inputs, compression="gzip")
        traj_grp.create_dataset("time", data=self.time, compression="gzip")
        traj_grp.create_dataset("cost", data=self.cost, compression="gzip")

        if save_ocp_trajs and self.predicted_states is not None:
            traj_grp.create_dataset("predicted_states", data=self.predicted_states, compression="gzip")
        if save_ocp_trajs and self.predicted_inputs is not None:
            traj_grp.create_dataset("predicted_inputs", data=self.predicted_inputs, compression="gzip")

    @classmethod
    def empty_from_cfg(cls, cfg: MPCConfig) -> 'MPCTrajectory':
        """Initialize empty trajectory arrays based on the provided config."""
        return cls.empty(
            T_sim=cfg.T_sim,
            N=cfg.N,
            nx=cfg.nx,
            nu=cfg.nu,
            dt=cfg.dt
        )

    @classmethod
    def empty(cls, T_sim: int, N: int, nx: int, nu: int, dt: float = 0.1) -> 'MPCTrajectory':
        """
        Initialize the trajectory with NaNs.
        
        Parameters
        ----------
        T_sim : int
            Number of simulation steps (trajectory will have length T_sim + 1 for states).
        N : int
            Prediction horizon length (steps).
        nx : int
            State dimension.
        nu : int
            Input dimension.
        dt : float
            Sampling time.
        """
        states = np.full((T_sim + 1, nx), np.nan)
        inputs = np.full((T_sim, nu), np.nan)
        time = np.arange(T_sim + 1) * dt
        cost = np.full((T_sim,), np.nan)
        predicted_states = np.full((T_sim, N + 1, nx), np.nan)
        predicted_inputs = np.full((T_sim, N, nu), np.nan)
        
        return cls(
            states=states,
            inputs=inputs,
            time=time,
            cost=cost,
            predicted_states=predicted_states,
            predicted_inputs=predicted_inputs,
            feasible=True
        )


@dataclass
class MPCData:
    """A single dataset entry combining config and result."""
    trajectory: MPCTrajectory
    meta: MPCMeta = field(default_factory=MPCMeta)
    config: MPCConfig = field(default_factory=MPCConfig)

    def verify(self) -> bool:
        """Verify internal consistency of the data entry."""
        T_sim = self.config.T_sim
        nx = self.trajectory.states.shape[1]
        nu = self.trajectory.inputs.shape[1]

        if self.trajectory.states.shape != (T_sim + 1, nx):
            __logger__.error(f"Trajectory states shape mismatch: expected {(T_sim + 1, nx)}, got {self.trajectory.states.shape}")
            return False
        if self.trajectory.inputs.shape != (T_sim, nu):
            __logger__.error(f"Trajectory inputs shape mismatch: expected {(T_sim, nu)}, got {self.trajectory.inputs.shape}")
            return False
        if self.trajectory.time.shape != (T_sim + 1,):
            __logger__.error(f"Trajectory time shape mismatch: expected {(T_sim + 1,)}, got {self.trajectory.time.shape}")
            return False
        if self.trajectory.cost.shape != (T_sim,):
            __logger__.error(f"Trajectory cost shape mismatch: expected {(T_sim,)}, got {self.trajectory.cost.shape}")
            return False
        if self.trajectory.predicted_states is not None:
            N = self.config.N
            if self.trajectory.predicted_states.shape != (T_sim, N + 1, nx):
                __logger__.error(f"Predicted states shape mismatch: expected {(T_sim, N + 1, nx)}, got {self.trajectory.predicted_states.shape}")
                return False
        if self.trajectory.predicted_inputs is not None:
            N = self.config.N
            if self.trajectory.predicted_inputs.shape != (T_sim, N, nu):
                __logger__.error(f"Predicted inputs shape mismatch: expected {(T_sim, N, nu)}, got {self.trajectory.predicted_inputs.shape}")
                return False
        if len(self.meta.status_codes) != T_sim:
            __logger__.error(f"Meta status codes length mismatch: expected {T_sim}, got {len(self.meta.status_codes)}")
            return False
        if not self.is_feasible() and self.trajectory.feasible:
            self.trajectory.feasible = False
        return True

    def is_feasible(self) -> bool:
        """Check if the trajectory is feasible."""
        # Prefer explicit feasibility flag if present
        if hasattr(self.trajectory, "feasible") and (self.trajectory.feasible is False):
            return False

        # Non-zero solver status codes indicate failure
        if self.meta is not None and getattr(self.meta, "status_codes", None):
            if any(int(c) != 0 for c in self.meta.status_codes):
                __logger__.warning(f"Entry ID {getattr(self.meta, 'id', 'unknown')} indicates non-zero solver status codes. "
                                   f"Status Codes: {np.unique(self.meta.status_codes).tolist()}")
                return False

        # NaNs in key arrays indicate an invalid run
        t = self.trajectory
        if np.isnan(t.states).any() or np.isnan(t.inputs).any() or np.isnan(t.cost).any():
            return False
        return True


class MPCDataset:
    """
    True Lazy-Loading Dataset.
    - Holds a file handle (_h5_file) instead of a list of data.
    - Reads arrays from disk only when __getitem__ is called.
    """
    def __init__(self, file_path: Optional[str] = None, data_buffer: List[MPCData] = None):
        self.file_path = Path(file_path) if file_path else None
        self.memory_buffer = data_buffer if data_buffer else []
        
        self._h5_file = None
        self._indices = [] # List of keys ['traj_0', 'traj_1', ...] in the file
        
        # Open file in read mode if it exists
        if self.file_path and self.file_path.exists():
            self._h5_file = h5py.File(self.file_path, 'r')
            self._indices = sorted(list(self._h5_file.keys()), key=lambda x: int(x.split('_')[1]))

    def add(self, entry: MPCData):
        """Add to temporary memory buffer (for generation phase)."""
        entry.meta.id = len(self)  # Assign ID based on current length
        self.memory_buffer.append(entry)

    def save(self, path: Path = None, mode: str = 'w', save_ocp_trajs: bool = True) -> None:
        """Flushes memory buffer to HDF5."""
        target_path = Path(path) if path else self.file_path
        if not target_path: 
            raise ValueError("No path provided")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(target_path, mode) as f:
            start_idx = len(f.keys())
            for i, entry in enumerate(self.memory_buffer):
                group_name = f"traj_{start_idx + i}"
                grp = f.create_group(group_name)

                entry.config.to_hdf5(grp)
                entry.trajectory.to_hdf5(grp, save_ocp_trajs)
                entry.meta.to_hdf5(grp)
                grp.attrs["feasible"] = entry.trajectory.feasible
        
        # Clear buffer and reload file to refresh indices
        self.memory_buffer = []
        self.file_path = target_path
        if self._h5_file: self._h5_file.close()
        self._h5_file = h5py.File(self.file_path, 'r')
        self._indices = sorted(list(self._h5_file.keys()), key=lambda x: int(x.split('_')[1]))

    @classmethod
    def load(cls, path: Path) -> 'MPCDataset':
        """
        Lazy Load: Just opens the file, does NOT read data.
        """
        path = Path(path)
        if not path.exists():
            __logger__.warning(f"File {path} not found.")
            return cls()
        return cls(file_path=path)

    def __len__(self) -> int:
        return len(self.memory_buffer) + len(self._indices)

    def __getitem__(self, idx):
        """
        Reads from disk on-demand.
        Supports both integer indexing and slicing.
        """
        # Handle slice objects
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            subset_data = [self[i] for i in range(start, stop, step or 1)]
            return MPCDataset(data_buffer=subset_data)
        
        # Handle integer indexing
        # Check memory buffer first
        if idx < len(self.memory_buffer):
            return self.memory_buffer[idx]
        
        # Check File
        # Calculate index relative to the file content
        file_idx = idx - len(self.memory_buffer)
        key = self._indices[file_idx]
        grp = self._h5_file[key]

        # This reads binary data from disk into RAM)
        traj = MPCTrajectory.from_hdf5(grp)
        meta = MPCMeta.from_hdf5(grp)
        config = MPCConfig.from_hdf5(grp)

        return MPCData(trajectory=traj, meta=meta, config=config)

    def __iter__(self) -> Iterator[MPCData]:
        """
        Explicit iterator to help static analysis tools infer the type of elements.
        """
        for i in range(len(self)):
            yield self[i]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Fast Filtering: Reads ONLY the metadata attributes (tiny), ignores arrays (huge).
        """
        rows = []
        # From Memory
        for i, entry in enumerate(self.memory_buffer):
            row = {
                "T_sim": int(entry.config.T_sim),
                "N": int(entry.config.N),
                "nx" : int(entry.config.nx),
                "nu" : int(entry.config.nu),
                "dt": float(entry.config.dt),
            }
            row.update(asdict(entry.meta))
            row['original_index'] = i
            row['source'] = 'mem'
            rows.append(row)

        # From File
        for i, key in enumerate(self._indices):
            grp = self._h5_file[key]
            meta = MPCMeta.from_hdf5(grp)
            row = {}
            cfg_grp = grp.get("config", None)
            if cfg_grp is not None:
                row.update({
                    "T_sim": int(cfg_grp.attrs.get("T_sim", 0)),
                    "N": int(cfg_grp.attrs.get("N", 10)),
                    "nx": int(cfg_grp.attrs.get("nx", 2)),
                    "nu": int(cfg_grp.attrs.get("nu", 1)),
                    "dt": float(cfg_grp.attrs.get("dt", 0.1)),
                })
                
            row.update(asdict(meta))
            row['original_index'] = len(self.memory_buffer) + i
            row['source'] = 'file'
            rows.append(row)

        return pd.DataFrame(rows)

    def filter(self, query: str) -> 'MPCDataset':
        """
        Returns a NEW dataset instance containing only the indices that match.
        """
        df = self.to_dataframe()
        filtered_indices = df.query(query)['original_index'].values.astype(int)
        subset_data = [self[i] for i in filtered_indices]
        return MPCDataset(data_buffer=subset_data)
    
    def validate(
        self,
        x_bounds: Optional[np.ndarray] = None,
        u_bounds: Optional[np.ndarray] = None,
        x_ref: Optional[np.ndarray] = None,
        tol_constraints: float = 1e-4,
        tol_stability: float = 1e-2
    ) -> pd.DataFrame:
        """
        Validates the generated dataset for consistency using Lazy Loading.
        Iterates over the dataset one-by-one to keep RAM usage low.
        """
        results = []
        
        for idx, entry in enumerate(self):
            traj = entry.trajectory
            meta = entry.meta
            cfg = entry.config

            traj_has_nan = (
                np.isnan(traj.states).any()
                or np.isnan(traj.inputs).any()
                or np.isnan(traj.cost).any()
            )

            # Determine effective constraints (Priority: function arg > dataset entry > None)
            # State Constraints
            if x_bounds is not None:
                lbx = x_bounds[0]
                ubx = x_bounds[1]
            else:
                lbx = cfg.constraints.lbx
                ubx = cfg.constraints.ubx

            # Input Constraints
            if u_bounds is not None:
                lbu = u_bounds[0]
                ubu = u_bounds[1]
            else:
                lbu = cfg.constraints.lbu
                ubu = cfg.constraints.ubu

            # Reference Goal
            if x_ref is not None:
                eff_goal = x_ref
            else:
                eff_goal = cfg.cost.yref_e if cfg.cost.has_terminal_cost() else None

            # Check Solver Feasibility
            solver_errors = [code for code in meta.status_codes if code != 0]
            solver_success = len(solver_errors) == 0
            
            # Check State Constraints
            state_violations = np.zeros(traj.states.shape[1], dtype=bool)
            if lbx is not None and ubx is not None:
                # Row 0: Lower bounds, Row 1: Upper bounds
                lower_vio = np.any(traj.states[:-1] < (lbx - tol_constraints), axis=0)
                upper_vio = np.any(traj.states[:-1] > (ubx + tol_constraints), axis=0)
                state_violations = lower_vio | upper_vio
            
            # Check Terminal State Constraints
            terminal_state_violations = np.zeros(traj.states.shape[1], dtype=bool)
            if cfg.constraints.lbx_e.size != 0 and cfg.constraints.ubx_e.size != 0:
                # Row 0: Lower bounds, Row 1: Upper bounds
                lower_vio = traj.states[-1] < (cfg.constraints.lbx_e - tol_constraints)
                upper_vio = traj.states[-1] > (cfg.constraints.ubx_e + tol_constraints)
                terminal_state_violations = lower_vio | upper_vio
                
            # Check Input Constraints
            input_violations = np.zeros(traj.inputs.shape[1], dtype=bool)
            if lbu is not None and ubu is not None:
                lower_vio_u = np.any(traj.inputs < (lbu - tol_constraints), axis=0)
                upper_vio_u = np.any(traj.inputs > (ubu + tol_constraints), axis=0)
                input_violations = lower_vio_u | upper_vio_u

            all_constraints_met = not (
                np.any(state_violations) or \
                np.any(terminal_state_violations) or \
                np.any(input_violations))

            # Any NaNs mean the rollout is incomplete/invalid; treat as failing checks.
            if traj_has_nan:
                all_constraints_met = False

            # Check Stability (Convergence to Goal)
            if eff_goal is None:
                eff_goal = np.zeros(traj.states.shape[1])
                
            final_state = traj.states[-1]
            dist_to_goal = float(np.linalg.norm(final_state - eff_goal)) if np.all(np.isfinite(final_state)) else float("inf")
            is_stable = dist_to_goal <= tol_stability

            # Compile Report
            results.append({
                "id": idx,
                "feasible": bool(traj.feasible and solver_success and (not traj_has_nan)),
                "constraints_met": all_constraints_met,
                "stable": is_stable,
                "final_dist": dist_to_goal,
                # Convert list to set to avoid storing 100 identical error codes
                "solver_codes": list(set(meta.status_codes)),
                "violated_state_dims": np.where(state_violations)[0].tolist(),
                "violated_terminal_state_dims": np.where(terminal_state_violations)[0].tolist(),
                "violated_input_dims": np.where(input_violations)[0].tolist()
            })

        df = pd.DataFrame(results)
        
        # Summary Logging
        if not df.empty:
            n_feas = df['feasible'].sum()
            n_stab = df['stable'].sum()
            n_cons = df['constraints_met'].sum()
            total = len(df)
            __logger__.info(f"Validation Results ({total} trajectories):")
            __logger__.info(f"  Feasible:        {n_feas}/{total} ({n_feas/total:.1%})")
            __logger__.info(f"  Stable:          {n_stab}/{total} ({n_stab/total:.1%})")
            __logger__.info(f"  Constraints Met: {n_cons}/{total} ({n_cons/total:.1%})")
        
        return df
    
    def close(self):
        if self._h5_file: self._h5_file.close()

    def __del__(self):
        self.close()