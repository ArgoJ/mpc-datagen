from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np
from numpy.typing import NDArray


class TrajectoryStatus(str, Enum):
    """Classification status for an empirical MPC trajectory rollout."""
    FEASIBLE_CONVERGED = "FEASIBLE_CONVERGED"
    FEASIBLE_UNCONVERGED = "FEASIBLE_UNCONVERGED"
    CONSTRAINT_VIOLATED = "CONSTRAINT_VIOLATED"
    INFEASIBLE = "INFEASIBLE"
    INVALID_DATA = "INVALID_DATA"


@dataclass
class ConstraintLimit:
    """Details of a single constraint's analytical level set bound."""
    name: str
    bound_value: float
    c_limit: float
    is_active: bool = False


@dataclass
class AnalyticROAReport:
    """Report summarizing the analytical Region of Attraction (ROA) / LQR maximal invariant ellipsoid."""
    method: str = "Analytic LQR ROA"
    is_valid: bool = True
    is_bounded: bool = True

    # Level set and active constraint
    c_min: float = float("nan")
    active_constraint: str = ""
    active_bound_value: float = float("nan")

    # Full breakdown of constraint limits
    constraint_limits: list[ConstraintLimit] = field(default_factory=list)

    # Geometry & Eigenvalues
    ellipsoid_volume: float | None = None
    eigenvalues_P: list[float] = field(default_factory=list)

    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampledPoint:
    """Represents the classification and metrics of a single sampled initial state rollout."""
    index: int
    x0: NDArray
    x_terminal: NDArray
    V_0: float | None
    V_terminal: float | None
    is_feasible: bool
    is_converged: bool
    status: TrajectoryStatus


@dataclass
class EmpiricalROAReport:
    """Comprehensive report summarizing empirical Region of Attraction (ROA) estimation."""
    method: str = "Empirical ROA Estimation"
    is_valid: bool = False

    # Trajectory statistics
    total_trajectories: int = 0
    num_feasible: int = 0
    num_converged: int = 0
    num_failed: int = 0
    convergence_rate: float = float("nan")
    feasibility_rate: float = float("nan")

    # Value function level set (if V_N / costs available)
    c_empirical: float | None = None

    # Spatial / geometric metrics
    state_bounds_empirical: dict[int, tuple[float, float]] = field(default_factory=dict)
    convex_hull_volume: float | None = None
    convex_hull_vertices: NDArray | None = None

    # Details & summary
    sampled_points: list[SampledPoint] = field(default_factory=list)
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
