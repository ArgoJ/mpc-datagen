import numpy as np

from numpy.typing import NDArray
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum

from ..mpc_data import MPCConfig
from ..package_logger import PackageLogger

__logger__ = PackageLogger.get_logger(__name__)


@dataclass
class SamplerBase(ABC):
    """Base class for initial state sampling configurations.

    Parameters
    ----------
    bounds : NDArray
        Sampling bounds.
    seed : int | None
        Random seed for reproducibility. If None, the random generator is os seeded. 

    Notes
    -----
    - The public API is:
        - ``post_init_cfg(cfg)`` called once after solver config extraction.
        - ``sample_x0(accepted_x0)`` called repeatedly during generation.
    - ``bounds`` are interpreted as a 2-by-nx array ``[lbx; ubx]`` for uniform sampling.
    """

    bounds: NDArray = field(default_factory=lambda: np.array([]))
    seed: int | None = None

    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def _set_and_validate_bounds(self, bounds: NDArray, nx: int) -> None:
        bounds_arr = np.asarray(bounds, dtype=float)
        if bounds_arr.shape != (2, nx):
            raise ValueError(f"Bounds must have shape (2, {nx}). Got {bounds_arr.shape}.")
        lb, ub = bounds_arr[0], bounds_arr[1]
        if np.any(~np.isfinite(lb)) or np.any(~np.isfinite(ub)):
            raise ValueError("Sampling bounds must be finite.")
        if np.any(lb >= ub):
            raise ValueError("Sampling bounds are invalid (lower >= upper).")
        self.bounds = bounds_arr

    def post_init_cfg(self, cfg: MPCConfig) -> None:
        """Post-initialization that depends on the MPCConfig.
        
        Parameters
        ----------
        cfg : MPCConfig
            The MPCConfig extracted from the solver, used to validate and process the sampling bounds.
        """
        nx = cfg.nx

        # Default: sample uniformly in [-1, 1]^nx.
        if np.asarray(self.bounds).size == 0:
            default_bounds = np.stack((-np.ones(nx), np.ones(nx)), axis=0)
            self._set_and_validate_bounds(default_bounds, nx)
            return

        self._set_and_validate_bounds(self.bounds, nx)

    def sample_x0(self) -> NDArray:
        """Sample an initial state $x_0$."""
        return self._rng.uniform(self.bounds[0], self.bounds[1])


class BoundType(str, Enum):
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"


@dataclass
class Sampler(SamplerBase):
    """
    Configuration for initial state sampling when generating MPC trajectories.
    
    Parameters
    ----------
    bound_type : BoundType
        Method for interpreting the `bounds` parameter.
        - "absolute": `bounds` are directly used as [lbx; ubx] for uniform sampling.
        - "percentage": `bounds` are interpreted as percentages in (0, 1] to shrink the solver's lbx/ubx around their midpoint.
    bounds : NDArray
        Sampling bounds, interpreted according to `bound_type`.
    min_dist : float | NDArray
        Minimum distance threshold for accepting a new sample relative to previously accepted samples.
    max_tries : int
        Maximum number of attempts to sample a unique initial state before raising an error.
    """
    bound_type: BoundType = BoundType.ABSOLUTE
    min_dist: float | NDArray = 0.0
    max_tries: int = 1_000

    _min_dist_is_array: bool = field(init=False, repr=False, default=False)
    _uniqueness_disabled: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        super().__post_init__()
            
        if self.max_tries < 1:
            raise ValueError("max_tries must be >= 1")

        # Treat non-scalars as per-state distance thresholds.
        self._min_dist_is_array = not np.isscalar(self.min_dist)
        if self._min_dist_is_array:
            self.min_dist = np.asarray(self.min_dist, dtype=float).reshape(-1)

        self._uniqueness_disabled = (
            (not self._min_dist_is_array and float(self.min_dist) <= 0.0)
            or (self._min_dist_is_array and bool(np.all(self.min_dist <= 0.0)))
        )

    def post_init_cfg(self, cfg: MPCConfig) -> None:
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
        match self.bound_type:
            case BoundType.PERCENTAGE:
                percentages = self.bounds

                # Basic validation
                if percentages.shape[0] != nx:
                    raise ValueError(f"Percentage array must have shape ({nx},). Got {percentages.shape}.")
                if np.any(percentages <= 0) or np.any(percentages > 1):
                    raise ValueError("Percentages must be in the interval (0, 1].")

                if np.any(~np.isfinite(cfg.constraints.lbx)) or np.any(~np.isfinite(cfg.constraints.ubx)):
                    raise ValueError("Percentage mode requires finite lbx/ubx for all states.")

                bounds_2xnx = self._calculate_percentage_bounds(
                    cfg.constraints.lbx, cfg.constraints.ubx, percentages)
                self._set_and_validate_bounds(bounds_2xnx, nx)
            case BoundType.ABSOLUTE:
                super().post_init_cfg(cfg)

    def _x0_is_too_close(self, x0: NDArray, existing_x0: NDArray) -> bool:
        """Return True if `x0` is within the configured minimum distance of `existing_x0`.
        - scalar threshold: max_i |x0_i - existing_i| <= x0_min_dist
        - vector threshold: |x0_i - existing_i| <= x0_min_dist[i] for all i
        """
        if self._min_dist_is_array:
            return bool(np.all(np.abs(x0 - existing_x0) <= self.min_dist))
        return bool(np.max(np.abs(x0 - existing_x0)) <= self.min_dist)

    def sample_x0(self, accepted_x0: list[NDArray] | None = None) -> NDArray:
        """Uniformly sample an x0 within bounds, rejecting if too close to any previously accepted x0."""
        if self._uniqueness_disabled:
            return super().sample_x0()

        if accepted_x0 is None:
            accepted_x0 = []

        for k in range(self.max_tries):
            x0 = super().sample_x0()
            if not any(self._x0_is_too_close(x0, prev) for prev in accepted_x0):
                __logger__.debug(f"Accepted x0 ({x0}) after {k+1} tries.")
                return x0

        raise RuntimeError(
            "Failed to sample a unique x0 within max_tries. "
            "Try decreasing min_dist or increasing max_tries."
        )

    @staticmethod
    def _calculate_percentage_bounds(lbx: NDArray, ubx: NDArray, percentages: NDArray) -> NDArray:
        """Shrink bounds symmetrically around the midpoint using the provided percentages.

        Returns
        -------
        bounds : NDArray
            Array with shape (2, nx) storing [sample_lb; sample_ub].
        """
        mid = 0.5 * (lbx + ubx)
        half_range = 0.5 * (ubx - lbx)
        shrink = (1.0 - percentages) * half_range

        sample_lb = mid - (half_range - shrink)
        sample_ub = mid + (half_range - shrink)

        if np.any(sample_lb >= sample_ub):
            raise ValueError("Computed sampling bounds are invalid (lower >= upper). Check percentages and solver bounds.")

        return np.stack((sample_lb, sample_ub), axis=0)
