import numpy as np

from numpy.typing import NDArray
from typing import Literal
from dataclasses import dataclass, field

from ..mpc_data import MPCConfig


@dataclass
class Sampler:
    """Configuration for initial state sampling when generating MPC trajectories."""
    bound_type: Literal["absolute", "percentage"] = "absolute"
    bounds: NDArray = np.array([[-1.0, -1.0], [1.0, 1.0]])
    min_dist: float | NDArray = 0.0
    max_tries: int = 1_000
    seed: int | None = None

    _min_dist_is_array: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            
        if self.max_tries < 1:
            raise ValueError("max_tries must be >= 1")

        # Treat non-scalars as per-state distance thresholds.
        self._min_dist_is_array = not np.isscalar(self.min_dist)
        if self._min_dist_is_array:
            self.min_dist = np.asarray(self.min_dist, dtype=float).reshape(-1)

    def cfg_post_init(self, cfg: MPCConfig) -> None:
        """Post-initialization that depends on the MPCConfig.
        
        Parameters
        ----------
        cfg : MPCConfig
            The MPCConfig extracted from the solver, used to validate and process the sampling bounds.
        """
        nx = cfg.nx
        
        # Minimum distance test
        if self._min_dist_is_array and self.min_dist.shape != (nx,):
            raise ValueError(f"min_dist vector must have shape ({nx},), got {self.min_dist.shape}")
        if not self._min_dist_is_array and float(self.min_dist) < 0.0:
            raise ValueError("min_dist must be non-negative.")
        if self._min_dist_is_array and np.any(self.min_dist < 0.0):
            raise ValueError("min_dist vector must be non-negative component-wise.")
        
        # Bound recalculation based on type
        if self.bound_type == "percentage":
            percentages = self.bounds

            # Basic validation
            if percentages.shape[0] != nx:
                raise ValueError(f"Percentage array must have shape ({nx},). Got {percentages.shape}.")
            if np.any(percentages <= 0) or np.any(percentages > 1):
                raise ValueError("Percentages must be in the interval (0, 1].")

            if np.any(~np.isfinite(cfg.constraints.lbx)) or np.any(~np.isfinite(cfg.constraints.ubx)):
                raise ValueError("Percentage mode requires finite lbx/ubx for all states.")

            self.bounds = self._calculate_percentage_bounds(
                cfg.constraints.lbx, cfg.constraints.ubx, percentages)
        elif self.bound_type == "absolute":
            if self.bounds.shape != (2, nx):
                raise ValueError(f"Bounds must have shape (2, {nx}) for absolute mode. Got {self.bounds.shape}.")
        else:
            raise ValueError(f"Unknown bound_type: {self.bound_type}. Use 'absolute' or 'percentage'.")

    def _x0_is_too_close(self, x0: NDArray, existing_x0: NDArray) -> bool:
        """Return True if `x0` is within the configured minimum distance of `existing_x0`.
        - scalar threshold: max_i |x0_i - existing_i| <= x0_min_dist
        - vector threshold: |x0_i - existing_i| <= x0_min_dist[i] for all i
        """
        if self._min_dist_is_array:
            return bool(np.all(np.abs(x0 - existing_x0) <= self.min_dist))
        return bool(np.max(np.abs(x0 - existing_x0)) <= self.min_dist)

    def sample_unique_x0(self, accepted_x0: list[NDArray]) -> NDArray:
        """Uniformly sample an x0 within bounds, rejecting if too close to any previously accepted x0."""
        uniqueness_disabled = (
            (not self._min_dist_is_array and float(self.min_dist) <= 0.0)
            or (self._min_dist_is_array and bool(np.all(self.min_dist <= 0.0)))
        )
        if uniqueness_disabled:
            return np.random.uniform(self.bounds[0], self.bounds[1])

        for k in range(self.max_tries):
            x0 = np.random.uniform(self.bounds[0], self.bounds[1])
            if not any(self._x0_is_too_close(x0, prev) for prev in accepted_x0):
                return x0

        raise RuntimeError(
            "Failed to sample a unique x0 within max_tries. "
            "Try decreasing min_dist or increasing max_tries."
        )

    @staticmethod
    def _calculate_percentage_bounds(lbx: NDArray, ubx: NDArray, percentages: NDArray) -> tuple[NDArray, NDArray]:
        """Shrink bounds symmetrically around the midpoint using the provided percentages"""
        mid = 0.5 * (lbx + ubx)
        half_range = 0.5 * (ubx - lbx)
        shrink = (1.0 - percentages) * half_range

        sample_lb = mid - (half_range - shrink)
        sample_ub = mid + (half_range - shrink)

        if np.any(sample_lb >= sample_ub):
            raise ValueError("Computed sampling bounds are invalid (lower >= upper). Check percentages and solver bounds.")
        
        return sample_lb, sample_ub
