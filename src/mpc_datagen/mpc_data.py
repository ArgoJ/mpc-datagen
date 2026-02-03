import json
import h5py
import numpy as np
import pandas as pd

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Iterator, Tuple, Dict, Set
from pathlib import Path

from .package_logger import PackageLogger

__logger__ = PackageLogger.get_logger(__name__)

def _is_defined_array(arr: Optional[np.ndarray], not_zero: bool = True) -> bool:
    """Check if an array is defined and non-empty."""
    if arr is None:
        return False
    arr = np.asarray(arr)
    if arr.size == 0:
        return False
    if not_zero and np.allclose(arr, 0):
        return False
    return True 

def _arrays_equal(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
    """Robust array equality check for config comparisons."""
    if a is None or b is None:
        return False
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        return False
    if a_arr.size == 0 and b_arr.size == 0:
        return True
    return np.array_equal(a_arr, b_arr)

def _values_equal(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return _arrays_equal(a, b)
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return bool(np.isclose(a, b))
    return a == b

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
            return cls()

        return cls(
            A=lin_sys_grp["A"][:] if "A" in lin_sys_grp else np.array([]),
            B=lin_sys_grp["B"][:] if "B" in lin_sys_grp else np.array([]),
            gd=lin_sys_grp["gd"][:] if "gd" in lin_sys_grp else np.array([]),
        )

    def to_hdf5(self, grp: h5py.Group, exclude_fields: Optional[set] = None) -> None:
        """Save linear system matrices to a trajectory group (creates a `linear_system` subgroup)."""
        exclude_fields = exclude_fields or set()
        field_names = list(self.__dataclass_fields__.keys())
        if all(name in exclude_fields for name in field_names):
            return
        lin_sys_grp = grp.create_group("linear_system")
        if "A" not in exclude_fields:
            lin_sys_grp.create_dataset("A", data=self.A, compression="gzip")
        if "B" not in exclude_fields:
            lin_sys_grp.create_dataset("B", data=self.B, compression="gzip")
        if "gd" not in exclude_fields:
            lin_sys_grp.create_dataset("gd", data=self.gd, compression="gzip") 

@dataclass
class LinearLSCost:
    r"""This class defines the objective function terms for the Optimal Control Problem (OCP)
    using the standard output-based formulation common in NMPC solvers (e.g., Acados).

    Stage Cost:
        l(x, u) = s * 0.5 * || Vx * x + Vu * u - yref ||_W^2
    
    Terminal Cost:
        l_e(x)  = s_e * 0.5 * || Vx_e * x - yref_e ||_W_e^2

    Attributes
    ----------
    Vx : np.ndarray
        Output matrix for the state $x$ in the stage cost. Shape: (ny, nx).
    Vu : np.ndarray
        Output matrix for the input $u$ in the stage cost. Shape: (ny, nu).
    W : np.ndarray
        Weighting matrix for the stage cost. Shape: (ny, ny).
        Should be symmetric and positive (semi-)definite.
    yref : np.ndarray
        Reference signal for the stage outputs. Shape: (ny,).
    Vx_e : np.ndarray
        Output matrix for the state $x$ in the terminal cost. Shape: (ny_e, nx).
    W_e : np.ndarray
        Weighting matrix for the terminal cost. Shape: (ny_e, ny_e).
    yref_e : np.ndarray
        Reference signal for the terminal outputs. Shape: (ny_e,).
    stage_scale : float
        Scaling factor applied to the stage cost term (default: 1.0).
        Useful for numerical conditioning of the solver.
    terminal_scale : float
        Scaling factor applied to the terminal cost term (default: 1.0).
    """
    Vx: np.ndarray = field(default_factory=lambda: np.array([[]]))
    Vu: np.ndarray = field(default_factory=lambda: np.array([[]]))
    W: np.ndarray = field(default_factory=lambda: np.array([[]]))
    yref: np.ndarray = field(default_factory=lambda: np.array([[]]))
    Vx_e: np.ndarray = field(default_factory=lambda: np.array([[]]))
    W_e: np.ndarray = field(default_factory=lambda: np.array([[]]))
    yref_e: np.ndarray = field(default_factory=lambda: np.array([[]]))
    stage_scale: float = 1.0
    terminal_scale: float = 1.0

    def get_stage_cost(self, x: np.ndarray, u: np.ndarray, use_scaled: bool = False) -> float:
        """Compute the cost for a given output vector y."""
        y = self.Vx @ x + self.Vu @ u
        e = y - self.yref
        
        scale = 0.5
        if use_scaled:
            scale *= self.stage_scale
        return scale * float(e.T @ self.W @ e)
    
    def get_terminal_cost(self, x: np.ndarray, use_scaled: bool = False) -> float:
        """Compute the terminal cost for a given output vector y."""
        if not self.has_terminal_cost():
            return 0.0
        y = self.Vx_e @ x
        e = y - self.yref_e
        
        scale = 0.5
        if use_scaled:
            scale *= self.terminal_scale
        return scale * float(e.T @ self.W_e @ e)

    def has_terminal_cost(self) -> bool:
        """Check if terminal cost matrices are defined."""
        return _is_defined_array(self.Vx_e, not_zero=True) and _is_defined_array(self.W_e, not_zero=True)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "LinearLSCost":
        """Load cost matrices from a trajectory group (expects a `cost` subgroup)."""
        cost_grp = grp.get("cost", None)
        if cost_grp is None:
            return cls()

        return cls(
            Vx=cost_grp["Vx"][:] if "Vx" in cost_grp else np.array([]),
            Vu=cost_grp["Vu"][:] if "Vu" in cost_grp else np.array([]),
            W=cost_grp["W"][:] if "W" in cost_grp else np.array([]),
            yref=cost_grp["yref"][:] if "yref" in cost_grp else np.array([]),
            Vx_e=cost_grp["Vx_e"][:] if "Vx_e" in cost_grp else np.array([]),
            W_e=cost_grp["W_e"][:] if "W_e" in cost_grp else np.array([]),
            yref_e=cost_grp["yref_e"][:] if "yref_e" in cost_grp else np.array([]),
            stage_scale=cost_grp.attrs.get("stage_scale", 1.0),
            terminal_scale=cost_grp.attrs.get("terminal_scale", 1.0),
        )

    def to_hdf5(self, grp: h5py.Group, exclude_fields: Optional[set] = None) -> None:
        """Save cost matrices to a trajectory group (creates a `cost` subgroup)."""
        exclude_fields = exclude_fields or set()
        field_names = list(self.__dataclass_fields__.keys())
        if all(name in exclude_fields for name in field_names):
            return
        cost_grp = grp.create_group("cost")
        if "Vx" not in exclude_fields:
            cost_grp.create_dataset("Vx", data=self.Vx, compression="gzip")
        if "Vu" not in exclude_fields:
            cost_grp.create_dataset("Vu", data=self.Vu, compression="gzip")
        if "W" not in exclude_fields:
            cost_grp.create_dataset("W", data=self.W, compression="gzip")
        if "yref" not in exclude_fields:
            cost_grp.create_dataset("yref", data=self.yref, compression="gzip")
        if "Vx_e" not in exclude_fields:
            cost_grp.create_dataset("Vx_e", data=self.Vx_e, compression="gzip")
        if "W_e" not in exclude_fields:
            cost_grp.create_dataset("W_e", data=self.W_e, compression="gzip")
        if "yref_e" not in exclude_fields:
            cost_grp.create_dataset("yref_e", data=self.yref_e, compression="gzip")
        if "stage_scale" not in exclude_fields:
            cost_grp.attrs["stage_scale"] = float(self.stage_scale)
        if "terminal_scale" not in exclude_fields:
            cost_grp.attrs["terminal_scale"] = float(self.terminal_scale)

@dataclass
class Constraints:
    r"""Definition of constraints for the Optimal Control Problem (OCP).

    Attributes
    ----------
    x0 : np.ndarray
        Initial state vector $x_0 \in \mathbb{R}^{n_x}$.
    lbx : np.ndarray
        Lower bounds on the state vector $\underline{x}$ (stage constraints).
        Shape: (nx,). Condition: $x_k \ge \underline{x}$ for $k=0 \dots N-1$.
    ubx : np.ndarray
        Upper bounds on the state vector $\overline{x}$ (stage constraints).
        Shape: (nx,). Condition: $x_k \le \overline{x}$ for $k=0 \dots N-1$.
    lbu : np.ndarray
        Lower bounds on the input vector $\underline{u}$.
        Shape: (nu,). Condition: $u_k \ge \underline{u}$ for $k=0 \dots N-1$.
    ubu : np.ndarray
        Upper bounds on the input vector $\overline{u}$.
        Shape: (nu,). Condition: $u_k \le \overline{u}$ for $k=0 \dots N-1$.
    lbx_e : np.ndarray
        Lower bounds on the terminal state vector $\underline{x}_f$.
        Shape: (nx,). Condition: $x_N \ge \underline{x}_f$.
    ubx_e : np.ndarray
        Upper bounds on the terminal state vector $\overline{x}_f$.
        Shape: (nx,). Condition: $x_N \le \overline{x}_f$.
    """
    x0: np.ndarray = field(default_factory=lambda: np.array([]))
    lbx: np.ndarray = field(default_factory=lambda: np.array([]))
    ubx: np.ndarray = field(default_factory=lambda: np.array([]))
    lbu: np.ndarray = field(default_factory=lambda: np.array([]))
    ubu: np.ndarray = field(default_factory=lambda: np.array([]))
    lbx_e: np.ndarray = field(default_factory=lambda: np.array([]))
    ubx_e: np.ndarray = field(default_factory=lambda: np.array([]))

    def has_bx(self) -> bool:
        """Check if state bounds are defined."""
        return _is_defined_array(self.lbx) and _is_defined_array(self.ubx)

    def has_bx_e(self) -> bool:
        """Check if terminal state bounds are defined."""
        return _is_defined_array(self.lbx_e) and _is_defined_array(self.ubx_e)

    def has_bu(self) -> bool:
        """Check if input bounds are defined."""
        return _is_defined_array(self.lbu) and _is_defined_array(self.ubu)

    def is_inside_bx(self, x: np.ndarray) -> bool:
        """Check if a state is inside the defined state bounds."""
        if not self.has_bx():
            return True
        return np.all(x >= self.lbx) and np.all(x <= self.ubx)

    def is_inside_bx_e(self, x: np.ndarray) -> bool:
        """Check if a state is inside the defined terminal state bounds."""
        if not self.has_bx_e():
            return True
        return np.all(x >= self.lbx_e) and np.all(x <= self.ubx_e)

    def is_inside_bu(self, u: np.ndarray) -> bool:
        """Check if an input is inside the defined input bounds."""
        if not self.has_bu():
            return True
        return np.all(u >= self.lbu) and np.all(u <= self.ubu)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "Constraints":
        """Load constraints from a trajectory group (expects a `constraints` subgroup)."""
        cons_grp = grp.get("constraints", None)
        if cons_grp is None:
            return cls()

        return cls(
            x0=cons_grp["x0"][:] if "x0" in cons_grp else np.array([]),
            lbx=cons_grp["lbx"][:] if "lbx" in cons_grp else np.array([]),
            ubx=cons_grp["ubx"][:] if "ubx" in cons_grp else np.array([]),
            lbu=cons_grp["lbu"][:] if "lbu" in cons_grp else np.array([]),
            ubu=cons_grp["ubu"][:] if "ubu" in cons_grp else np.array([]),
            lbx_e=cons_grp["lbx_e"][:] if "lbx_e" in cons_grp else np.array([]),
            ubx_e=cons_grp["ubx_e"][:] if "ubx_e" in cons_grp else np.array([]),
        )

    def to_hdf5(self, grp: h5py.Group, exclude_fields: Optional[set] = None) -> None:
        """Save constraints to a trajectory group (creates a `constraints` subgroup)."""
        exclude_fields = exclude_fields or set()
        field_names = list(self.__dataclass_fields__.keys())
        if all(name in exclude_fields for name in field_names):
            return
        cons_grp = grp.create_group("constraints")
        if "x0" not in exclude_fields:
            cons_grp.create_dataset("x0", data=self.x0, compression="gzip")
        if "lbx" not in exclude_fields:
            cons_grp.create_dataset("lbx", data=self.lbx, compression="gzip")
        if "ubx" not in exclude_fields:
            cons_grp.create_dataset("ubx", data=self.ubx, compression="gzip")
        if "lbu" not in exclude_fields:
            cons_grp.create_dataset("lbu", data=self.lbu, compression="gzip")
        if "ubu" not in exclude_fields:
            cons_grp.create_dataset("ubu", data=self.ubu, compression="gzip")
        if "lbx_e" not in exclude_fields:
            cons_grp.create_dataset("lbx_e", data=self.lbx_e, compression="gzip")
        if "ubx_e" not in exclude_fields:
            cons_grp.create_dataset("ubx_e", data=self.ubx_e, compression="gzip")

@dataclass
class MPCConfig:
    r"""Configuration parameters for the Model Predictive Control (MPC) problem.

    Attributes
    ----------
    T_sim : int
        Total number of closed-loop simulation steps to perform ($T_{sim}$).
    N : int
        Prediction horizon length for the OCP.
    nx : int
        Dimension of the state vector $x \in \mathbb{R}^{n_x}$.
    nu : int
        Dimension of the control input vector $u \in \mathbb{R}^{n_u}$.
    dt : float
        Sampling time $\Delta t$ in seconds.
    constraints : Constraints
        Definition of state and input constraints (box constraints) and terminal sets.
    model : LinearSystem
        The linear system dynamics defined by matrices $A$ and $B$ (discrete time).
    cost : LinearLSCost
        The quadratic cost function parameters (weighting matrices $Q, R$, terminal cost $P$, references).
    """
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
        """Load config from a trajectory group, merging optional global config if present."""
        cfg_local = grp.get("config", None)
        cfg_global = grp.file.get("global_config", None) if grp.file is not None else None

        if cfg_local is None and cfg_global is None:
            raise ValueError("No 'config' group found in the provided HDF5 group.")

        base_grp = cfg_global if cfg_global is not None else cfg_local
        cfg = cls(
            T_sim=int(base_grp.attrs.get("T_sim", 0)),
            N=int(base_grp.attrs.get("N", 10)),
            nx=int(base_grp.attrs.get("nx", 2)),
            nu=int(base_grp.attrs.get("nu", 1)),
            dt=float(base_grp.attrs.get("dt", 0.1)),
            constraints=Constraints.from_hdf5(base_grp),
            model=LinearSystem.from_hdf5(base_grp),
            cost=LinearLSCost.from_hdf5(base_grp),
        )

        if cfg_local is None or cfg_local is base_grp:
            return cfg

        # Override scalars if present locally
        if "T_sim" in cfg_local.attrs:
            cfg.T_sim = int(cfg_local.attrs["T_sim"])
        if "N" in cfg_local.attrs:
            cfg.N = int(cfg_local.attrs["N"])
        if "nx" in cfg_local.attrs:
            cfg.nx = int(cfg_local.attrs["nx"])
        if "nu" in cfg_local.attrs:
            cfg.nu = int(cfg_local.attrs["nu"])
        if "dt" in cfg_local.attrs:
            cfg.dt = float(cfg_local.attrs["dt"])

        # Override constraints if present locally
        local_cons = cfg_local.get("constraints", None)
        if local_cons is not None:
            if "x0" in local_cons:
                cfg.constraints.x0 = local_cons["x0"][:]
            if "lbx" in local_cons:
                cfg.constraints.lbx = local_cons["lbx"][:]
            if "ubx" in local_cons:
                cfg.constraints.ubx = local_cons["ubx"][:]
            if "lbu" in local_cons:
                cfg.constraints.lbu = local_cons["lbu"][:]
            if "ubu" in local_cons:
                cfg.constraints.ubu = local_cons["ubu"][:]
            if "lbx_e" in local_cons:
                cfg.constraints.lbx_e = local_cons["lbx_e"][:]
            if "ubx_e" in local_cons:
                cfg.constraints.ubx_e = local_cons["ubx_e"][:]

        # Override model if present locally
        local_model = cfg_local.get("linear_system", None)
        if local_model is not None:
            if "A" in local_model:
                cfg.model.A = local_model["A"][:]
            if "B" in local_model:
                cfg.model.B = local_model["B"][:]
            if "gd" in local_model:
                cfg.model.gd = local_model["gd"][:]

        # Override cost if present locally
        local_cost = cfg_local.get("cost", None)
        if local_cost is not None:
            if "Vx" in local_cost:
                cfg.cost.Vx = local_cost["Vx"][:]
            if "Vu" in local_cost:
                cfg.cost.Vu = local_cost["Vu"][:]
            if "W" in local_cost:
                cfg.cost.W = local_cost["W"][:]
            if "yref" in local_cost:
                cfg.cost.yref = local_cost["yref"][:]
            if "Vx_e" in local_cost:
                cfg.cost.Vx_e = local_cost["Vx_e"][:]
            if "W_e" in local_cost:
                cfg.cost.W_e = local_cost["W_e"][:]
            if "yref_e" in local_cost:
                cfg.cost.yref_e = local_cost["yref_e"][:]
            if "stage_scale" in local_cost.attrs:
                cfg.cost.stage_scale = float(local_cost.attrs["stage_scale"])
            if "terminal_scale" in local_cost.attrs:
                cfg.cost.terminal_scale = float(local_cost.attrs["terminal_scale"])

        return cfg

    def to_hdf5(
        self,
        grp: h5py.Group,
        exclude_attrs: Optional[set] = None,
        exclude_constraints: Optional[set] = None,
        exclude_model: Optional[set] = None,
        exclude_cost: Optional[set] = None,
        group_name: str = "config",
    ) -> None:
        """Save config to a trajectory group (creates a `config` subgroup)."""
        exclude_attrs = exclude_attrs or set()
        attr_fields = set(self.__dataclass_fields__.keys()).difference({"constraints", "model", "cost"})
        constraint_fields = set(self.constraints.__dataclass_fields__.keys())
        model_fields = set(self.model.__dataclass_fields__.keys())
        cost_fields = set(self.cost.__dataclass_fields__.keys())

        if (
            attr_fields.issubset(exclude_attrs)
            and constraint_fields.issubset(exclude_constraints or set())
            and model_fields.issubset(exclude_model or set())
            and cost_fields.issubset(exclude_cost or set())
        ):
            return

        cfg_grp = grp.create_group(group_name)
        if "T_sim" not in exclude_attrs:
            cfg_grp.attrs["T_sim"] = int(self.T_sim)
        if "N" not in exclude_attrs:
            cfg_grp.attrs["N"] = int(self.N)
        if "nx" not in exclude_attrs:
            cfg_grp.attrs["nx"] = int(self.nx)
        if "nu" not in exclude_attrs:
            cfg_grp.attrs["nu"] = int(self.nu)
        if "dt" not in exclude_attrs:
            cfg_grp.attrs["dt"] = float(self.dt)

        self.constraints.to_hdf5(cfg_grp, exclude_fields=exclude_constraints)
        self.model.to_hdf5(cfg_grp, exclude_fields=exclude_model)
        self.cost.to_hdf5(cfg_grp, exclude_fields=exclude_cost)


@dataclass
class MPCTrajectory:
    r"""Data resulting from a single MPC simulation run.

    This container holds both the realized closed-loop trajectories (what actually happened)
    and the open-loop predictions from the solver (what was planned).

    Attributes
    ----------
    states : np.ndarray
        Closed-loop state trajectory history. Shape: (T_sim + 1, nx).
    inputs : np.ndarray
        Closed-loop input trajectory history. Shape: (T_sim, nu).
    times : np.ndarray
        Simulation timestamps. Shape: (T_sim + 1,).
    V_solver : np.ndarray
        Raw objective values returned by the solver (potentially scaled). Shape: (T_sim,).
    V_N : np.ndarray, optional
        Re-calculated Optimal Value Function V_N(x) (Lyapunov candidate). Shape: (T_sim,).
    V_horizon : np.ndarray, optional
        Sequence of stage costs along the prediction horizon. Shape: (T_sim, N+1).
    predicted_states : np.ndarray, optional
        Predicted open-loop state trajectories at each step. Shape: (T_sim, N+1, nx).
    predicted_inputs : np.ndarray, optional
        Predicted open-loop input trajectories at each step. Shape: (T_sim, N, nu).
    feasible : bool
        Flag indicating if the OCP was feasible throughout the trajectory. Default: True.

    Properties
    ----------
    sim_steps : int
        Get the simulation length in steps.
    horizon : int, optional
        Get the prediction horizon length (if available).
    """

    states: np.ndarray
    inputs: np.ndarray
    times: np.ndarray
    V_solver: np.ndarray
    V_N: Optional[np.ndarray] = None
    V_horizon: Optional[np.ndarray] = None
    predicted_states: Optional[np.ndarray] = None
    predicted_inputs: Optional[np.ndarray] = None
    feasible: bool = True
    
    @property
    def sim_steps(self) -> int:
        """Get the simulation length in steps."""
        return self.states.shape[0] - 1
    
    @property
    def horizon(self) -> Optional[int]:
        """Get the prediction horizon length (if available)."""
        if self.predicted_states is not None:
            return self.predicted_states.shape[1] - 1
        if self.predicted_inputs is not None:
            return self.predicted_inputs.shape[1]
        if self.V_horizon is not None:
            return self.V_horizon.shape[1] - 1
        return None

    @property
    def V_pred(self) -> np.ndarray:
        """The cost-to-go for each trajectory at each step. (shape: (T_sim, N+1))"""
        if self.V_horizon is None:
            raise ValueError("No horizon costs available to calculate cost-to-go. Call 'recalculate_costs()' first.")
        return np.flip(np.cumsum(np.flip(self.V_horizon, axis=1), axis=1), axis=1)
    
    def get_scaled_costs(self, stage_scale: float, terminal_scale: float = 1.0) -> np.ndarray:
        """
        Calculate scaled costs for each trajectory.
        
        Parameters
        ----------
        stage_scale, terminal_scale : float
            Scaling factors for stage and terminal costs respectively.

        Returns
        -------
        scaled_costs : np.ndarray
            The scaled total costs for each trajectory. (shape: (T_sim,))
        """
        if self.V_horizon is None:
            raise ValueError("No horizon costs available to scale. Call 'recalculate_costs()' first.")

        scaled_costs = self.V_horizon.copy()
        scaled_costs[:, :-1] *= stage_scale  # Scale stage costs
        scaled_costs[:, -1] *= terminal_scale  # Scale terminal costs
        
        scaled_costs = scaled_costs.sum(axis=1)
        return scaled_costs

    def recalculate_costs(self, cost: LinearLSCost) -> None:
        """
        Recalculates the discrete costs based on the predictions.
        
        Parameters
        ----------
        cost : LinearLSCost
            Cost function parameters.
        """
        # Ensure predictions are present
        if self.predicted_states is None or self.predicted_inputs is None:
            raise ValueError("No prediction data (predicted_states/inputs) available.")
        if cost.Vu.size == 0 or cost.Vx.size == 0 or cost.W.size == 0:
            raise ValueError("Cost matrices Vx, Vu, W must be defined.")
        if cost.yref.size == 0:
            raise ValueError("Cost reference yref must be defined.")
        if cost.yref_e.size == 0 and cost.has_terminal_cost():
            raise ValueError("Terminal cost reference yref_e must be defined if terminal cost is used.")

        # Dimensions
        T_sim = self.predicted_states.shape[0]
        N = self.predicted_inputs.shape[1]

        # Initialize arrays            
        self.V_N = np.full(T_sim, np.nan)
        self.V_horizon = np.full((T_sim, N+1), np.nan)

        # Calculation loop
        for i in range(T_sim):

            # Stage Cost
            for k in range(N):
                x_k = self.predicted_states[i, k, :]
                u_k = self.predicted_inputs[i, k, :]
                self.V_horizon[i, k] = cost.get_stage_cost(x_k, u_k)
            
            # Terminal Cost
            if cost.Vx_e.size != 0 and cost.W_e.size != 0:
                x_N = self.predicted_states[i, N, :]
                self.V_horizon[i, N] = cost.get_terminal_cost(x_N)
            
        # Total unscaled costs
        self.V_N = self.V_horizon.sum(axis=1)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "MPCTrajectory":
        """Load trajectory arrays from a trajectory group."""
        traj_grp = grp.get("trajectory", None)
        if traj_grp is None:
            raise ValueError("No 'trajectory' group found in the provided HDF5 group.")
        
        return cls(
            states=traj_grp["states"][:, :],
            inputs=traj_grp["inputs"][:, :],
            times=traj_grp["times"][:],
            V_solver=traj_grp["V_solver"][:],
            V_N=traj_grp["V_N"][:] if "V_N" in traj_grp else None,
            V_horizon=traj_grp["V_horizon"][:, :] if "V_horizon" in traj_grp else None,
            predicted_states=traj_grp["predicted_states"][:, :, :] if "predicted_states" in traj_grp else None,
            predicted_inputs=traj_grp["predicted_inputs"][:, :, :] if "predicted_inputs" in traj_grp else None,
            feasible=bool(traj_grp.attrs.get("feasible", True)),
        )

    def to_hdf5(self, grp: h5py.Group, save_ocp_trajs: bool = True) -> None:
        """Save trajectory arrays to a trajectory group."""
        traj_grp = grp.create_group("trajectory")
        traj_grp.create_dataset("states", data=self.states, compression="gzip")
        traj_grp.create_dataset("inputs", data=self.inputs, compression="gzip")
        traj_grp.create_dataset("times", data=self.times, compression="gzip")
        traj_grp.create_dataset("V_solver", data=self.V_solver, compression="gzip")

        if self.V_N is not None:
            traj_grp.create_dataset("V_N", data=self.V_N, compression="gzip")
        if save_ocp_trajs and self.V_horizon is not None:
            traj_grp.create_dataset("V_horizon", data=self.V_horizon, compression="gzip")
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
        times = np.arange(T_sim + 1) * dt
        costs = np.full((T_sim,), np.nan)
        predicted_states = np.full((T_sim, N+1, nx), np.nan)
        predicted_inputs = np.full((T_sim, N, nu), np.nan)
        
        return cls(
            states=states,
            inputs=inputs,
            times=times,
            V_solver=costs,
            predicted_states=predicted_states,
            predicted_inputs=predicted_inputs,
            feasible=True
        )


@dataclass
class MPCData:
    r"""Container class representing a single entry in the MPC dataset.

    Attributes
    ----------
    trajectory : MPCTrajectory
        The numerical results of the simulation run, containing both the closed-loop
        realization and the open-loop solver predictions ($V_N$, cost sequences).
    meta : MPCMeta
        Metadata describing the generation context (e.g., unique UUID, timestamps,
        solver status, random seeds).
    config : MPCConfig
        The specific parameter set used to define the OCP for this run
        (including system matrices, constraints, and cost weights).
    """
    trajectory: MPCTrajectory
    meta: MPCMeta = field(default_factory=MPCMeta)
    config: MPCConfig = field(default_factory=MPCConfig)

    def finalize(self, recalculate_costs: bool = False) -> bool:
        """Finalizes the data entry, checking consistency and optionally recalculating costs."""
        T_sim = self.config.T_sim
        traj = self.trajectory
        nx = traj.states.shape[1]
        nu = traj.inputs.shape[1]

        if traj.states.shape != (T_sim + 1, nx):
            raise ValueError(f"Trajectory states shape mismatch: expected {(T_sim + 1, nx)}, got {traj.states.shape}")
        if traj.inputs.shape != (T_sim, nu):
            raise ValueError(f"Trajectory inputs shape mismatch: expected {(T_sim, nu)}, got {traj.inputs.shape}")
        if traj.times.shape != (T_sim + 1,):
            raise ValueError(f"Trajectory times shape mismatch: expected {(T_sim + 1,)}, got {traj.times.shape}")
        if traj.V_solver.shape != (T_sim,):
            raise ValueError(f"Trajectory solver_costs shape mismatch: expected {(T_sim,)}, got {traj.V_solver.shape}")
        if len(self.meta.status_codes) != T_sim:
            raise ValueError(f"Meta status codes length mismatch: expected {T_sim}, got {len(self.meta.status_codes)}")
        if not self.is_feasible() and traj.feasible:
            __logger__.warning(f"Entry marked as feasible, but solver status codes indicate infeasibility.")
            traj.feasible = False
            
        # Optional arrays
        N = self.config.N
        if N != traj.horizon:
            raise ValueError(f"Prediction horizon mismatch: config N={N}, trajectory horizon={traj.horizon}")
        if traj.predicted_states is not None and traj.predicted_inputs is not None:
            if traj.predicted_states.shape != (T_sim, N+1, nx):
                raise ValueError(f"Predicted states shape mismatch: expected {(T_sim, N+1, nx)}, got {traj.predicted_states.shape}")
            if traj.predicted_inputs.shape != (T_sim, N, nu):
                raise ValueError(f"Predicted inputs shape mismatch: expected {(T_sim, N, nu)}, got {traj.predicted_inputs.shape}")
            
        if recalculate_costs:
            traj.recalculate_costs(self.config.cost)
            if traj.V_N is not None:
                if traj.V_N.shape != (T_sim,):
                    raise ValueError(f"Trajectory costs shape mismatch: expected {(T_sim,)}, got {traj.V_N.shape}")
            if traj.V_horizon is not None:
                if traj.V_horizon.shape != (T_sim, N+1):
                    raise ValueError(f"Trajectory horizon_costs shape mismatch: expected {(T_sim, N+1)}, got {traj.V_horizon.shape}")

            scaled_costs = traj.get_scaled_costs(stage_scale=self.config.cost.stage_scale, terminal_scale=self.config.cost.terminal_scale)
            if not np.allclose(scaled_costs, traj.V_solver, rtol=1e-3, atol=1e-6, equal_nan=True):
                __logger__.warning(f"Recalculated scaled costs do not match stored costs.")

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
        if np.isnan(t.states).any() or np.isnan(t.inputs).any() or np.isnan(t.V_solver).any():
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
            self._indices = sorted(
                [k for k in self._h5_file.keys() if k.startswith("traj_")],
                key=lambda x: int(x.split('_')[1])
            )

    def add(self, entry: MPCData):
        """Add to temporary memory buffer (for generation phase)."""
        entry.meta.id = len(self)  # Assign ID based on current length
        self.memory_buffer.append(entry)

    def save(self, path: Path = None, mode: str = 'w', save_ocp_trajs: bool = True) -> None:
        """Flushes memory buffer to HDF5."""
        target_path = Path(path) if path else self.file_path
        if not target_path: raise ValueError("No path provided")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(target_path, mode) as f:
            global_exclusions = self._handle_global_config(f)
            start_idx = len([k for k in f.keys() if k.startswith("traj_")])
            for i, entry in enumerate(self.memory_buffer):
                self._save_entry(f.create_group(f"traj_{start_idx + i}"), entry, global_exclusions, save_ocp_trajs)

        self._refresh_after_save(target_path)

    def _refresh_after_save(self, path: Path) -> None:
        self.memory_buffer = []
        self.file_path = path
        if self._h5_file: self._h5_file.close()
        self._h5_file = h5py.File(self.file_path, 'r')
        self._indices = sorted(
            [k for k in self._h5_file.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split('_')[1])
        )

    def _handle_global_config(self, f: h5py.File) -> Dict[str, Set[str]]:
        if "global_config" in f:
            return self._read_existing_global_config(f["global_config"])

        if self.memory_buffer:
            common = self._find_common_config_fields()
            if any(len(v) > 0 for v in common.values()):
                self._write_global_config(f, self.memory_buffer[0].config, common)
                return common

        return {k: set() for k in ["attrs", "constraints", "model", "cost"]}

    def _save_entry(self, grp: h5py.Group, entry: MPCData, exclusions: Dict[str, Set[str]], save_ocp: bool) -> None:
        entry.config.to_hdf5(
            grp,
            exclude_attrs=exclusions["attrs"],
            exclude_constraints=exclusions["constraints"],
            exclude_model=exclusions["model"],
            exclude_cost=exclusions["cost"],
            group_name="config"
        )
        entry.trajectory.to_hdf5(grp, save_ocp)
        entry.meta.to_hdf5(grp)
        grp.attrs["feasible"] = entry.trajectory.feasible

    def _get_field_map(self) -> Dict[str, List[str]]:
        """Returns a mapping of config category to list of field names."""
        return {
            "attrs": sorted(set(MPCConfig.__dataclass_fields__.keys()) - {"constraints", "model", "cost"}),
            "constraints": sorted(Constraints.__dataclass_fields__.keys()),
            "model": sorted(LinearSystem.__dataclass_fields__.keys()),
            "cost": sorted(LinearLSCost.__dataclass_fields__.keys()),
        }

    def _find_common_config_fields(self) -> Dict[str, Set[str]]:
        """Identifies fields that are constant across the memory buffer."""
        if not self.memory_buffer: return {}
        
        first = self.memory_buffer[0].config
        candidates = {k: set(v) for k, v in self._get_field_map().items()}

        for entry in self.memory_buffer[1:]:
            cfg = entry.config
            for cat, names in candidates.items():
                for name in list(names): 
                    val_cur = self._get_config_value(cfg, cat, name)
                    val_ref = self._get_config_value(first, cat, name)
                    if not _values_equal(val_cur, val_ref):
                        names.discard(name)
        return candidates

    def _get_config_value(self, cfg: MPCConfig, category: str, name: str):
        if category == "attrs": return getattr(cfg, name)
        if category == "constraints": return getattr(cfg.constraints, name)
        if category == "model": return getattr(cfg.model, name)
        if category == "cost": return getattr(cfg.cost, name)
        return None

    def _read_existing_global_config(self, grp: h5py.Group) -> Dict[str, Set[str]]:
        field_map = self._get_field_map()
        exclusions = {}
        
        exclusions["attrs"] = {k for k in field_map["attrs"] if k in grp.attrs}
        
        sub = grp.get("constraints")
        exclusions["constraints"] = {k for k in field_map["constraints"] if k in sub} if sub else set()

        sub = grp.get("linear_system")
        exclusions["model"] = {k for k in field_map["model"] if k in sub} if sub else set()

        sub = grp.get("cost")
        exclusions["cost"] = {k for k in field_map["cost"] if (k in sub or k in sub.attrs)} if sub else set()
        
        return exclusions

    def _write_global_config(self, f: h5py.File, cfg: MPCConfig, fields: Dict[str, Set[str]]) -> None:
        field_map = self._get_field_map()
        exclusions = {k: set(field_map[k]) - fields[k] for k in field_map}

        cfg.to_hdf5(
            f,
            exclude_attrs=exclusions["attrs"],
            exclude_constraints=exclusions["constraints"],
            exclude_model=exclusions["model"],
            exclude_cost=exclusions["cost"],
            group_name="global_config"
        )


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
            global_cfg = self._h5_file.get("global_config", None)
            base_attrs = global_cfg.attrs if global_cfg is not None else {}

            row.update({
                "T_sim": int(base_attrs.get("T_sim", 0)),
                "N": int(base_attrs.get("N", 10)),
                "nx": int(base_attrs.get("nx", 2)),
                "nu": int(base_attrs.get("nu", 1)),
                "dt": float(base_attrs.get("dt", 0.1)),
            })

            if cfg_grp is not None:
                if "T_sim" in cfg_grp.attrs:
                    row["T_sim"] = int(cfg_grp.attrs["T_sim"])
                if "N" in cfg_grp.attrs:
                    row["N"] = int(cfg_grp.attrs["N"])
                if "nx" in cfg_grp.attrs:
                    row["nx"] = int(cfg_grp.attrs["nx"])
                if "nu" in cfg_grp.attrs:
                    row["nu"] = int(cfg_grp.attrs["nu"])
                if "dt" in cfg_grp.attrs:
                    row["dt"] = float(cfg_grp.attrs["dt"])
                
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
                or np.isnan(traj.V_solver).any()
            )

            # Determine effective constraints (Priority: function arg > dataset entry > None)
            # State Constraints
            if _is_defined_array(x_bounds):
                lbx = x_bounds[0]
                ubx = x_bounds[1]
            else:
                lbx = cfg.constraints.lbx
                ubx = cfg.constraints.ubx

            # Input Constraints
            if _is_defined_array(u_bounds):
                lbu = u_bounds[0]
                ubu = u_bounds[1]
            else:
                lbu = cfg.constraints.lbu
                ubu = cfg.constraints.ubu

            # Reference Goal
            if _is_defined_array(x_ref):
                eff_goal = x_ref
            else:
                eff_goal = cfg.cost.yref_e if cfg.cost.has_terminal_cost() else None

            # Check Solver Feasibility
            solver_errors = [code for code in meta.status_codes if code != 0]
            solver_success = len(solver_errors) == 0
            
            # Check State Constraints
            state_violations = np.zeros(traj.states.shape[1], dtype=bool)
            if _is_defined_array(lbx) and _is_defined_array(ubx):
                # Row 0: Lower bounds, Row 1: Upper bounds
                lower_vio = np.any(traj.states[:-1] < (lbx - tol_constraints), axis=0)
                upper_vio = np.any(traj.states[:-1] > (ubx + tol_constraints), axis=0)
                state_violations = lower_vio | upper_vio
            
            # Check Terminal State Constraints
            terminal_state_violations = np.zeros(traj.states.shape[1], dtype=bool)
            if _is_defined_array(cfg.constraints.lbx_e) and _is_defined_array(cfg.constraints.ubx_e):
                # Row 0: Lower bounds, Row 1: Upper bounds
                lower_vio = traj.states[-1] < (cfg.constraints.lbx_e - tol_constraints)
                upper_vio = traj.states[-1] > (cfg.constraints.ubx_e + tol_constraints)
                terminal_state_violations = lower_vio | upper_vio
                
            # Check Input Constraints
            input_violations = np.zeros(traj.inputs.shape[1], dtype=bool)
            if _is_defined_array(lbu) and _is_defined_array(ubu):
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