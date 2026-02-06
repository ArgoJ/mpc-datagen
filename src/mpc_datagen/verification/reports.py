from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class StabilityReport:
    method: str = ""
    is_stable: bool = False
    applicability: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

@dataclass
class LyapunovDescentReport(StabilityReport):
    method: str = "Lyapunov Descent Check"
    max_increase: float = 0.0
    violation_count: int = 0
    total_steps: int = 0
    applicability: bool = True

@dataclass
class AsymptoticStabilityReport(StabilityReport):
    method: str = "Asymptotic Stability Check"
    min_alpha: float = float("nan")
    max_violation: float = float("nan")
    applicability: bool = True # Always applicable

@dataclass
class GrüneHorizonReport(StabilityReport):
    method: str = "Grüne Horizon Condition"
    gamma_estimate: float = float("nan")
    alpha_N_estimate: float = float("nan")
    required_horizon: float = float("nan")

@dataclass
class AlphaViolationStats:
    min_alpha: float = float("nan")
    max_violation: float = float("nan")
    min_residual: float = float("nan")  
    n_used: int = 0

