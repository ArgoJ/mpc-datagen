import numpy as np

from numpy.typing import NDArray
from dataclasses import dataclass, field

from pkg_logger import get_package_logger

__logger__ = get_package_logger(__name__)


@dataclass
class SamplerBase:
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
        - ``sample_x0(accepted_x0)`` called repeatedly during generation.
    - ``bounds`` are interpreted as a 2-by-nx array ``[lbx; ubx]`` for uniform sampling.
    """

    bounds: NDArray = field(default_factory=lambda: np.array([]))
    bias: NDArray = field(default_factory=lambda: np.array([]))
    seed: int | None = None
    
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._validate_bounds()
        self._validate_bias()

    def _validate_bounds(self) -> None:
        bounds_arr = np.asarray(self.bounds, dtype=float)
        if bounds_arr.ndim != 2 or bounds_arr.shape[0] != 2:
            raise ValueError(f"Bounds must have shape (2, nx). Got {bounds_arr.shape}.")
        lb, ub = bounds_arr[0], bounds_arr[1]
        if np.any(~np.isfinite(lb)) or np.any(~np.isfinite(ub)):
            raise ValueError("Sampling bounds must be finite.")
        if np.any(lb >= ub):
            raise ValueError("Sampling bounds are invalid (lower >= upper).")
        self.bounds = bounds_arr

    def _validate_bias(self) -> None:
        if self.bias.size == 0:
            return
        
        bias_arr = np.asarray(self.bias, dtype=float)
        if bias_arr.ndim != 1 or bias_arr.shape[0] != self.bounds.shape[1]:
            raise ValueError(f"Bias must have shape ({self.bounds.shape[1]},). Got {bias_arr.shape}.")
        if np.any(~np.isfinite(bias_arr)):
            raise ValueError("Bias values must be finite.")
        self.bias = bias_arr

    def sample_x0(self, n: int = 1) -> NDArray:
        """Sample one or multiple initial states.

        Parameters
        ----------
        n : int
            Number of samples to draw. Returns shape (n, nx) for n > 1 and shape (nx,)
            for n == 1.
        """
        if n < 1:
            raise ValueError("n must be >= 1.")

        rand_num = self._rng.random((n, self.bounds.shape[1]))
        diff = self.bounds[1, :] - self.bounds[0, :]
        rand_num = self.bounds[0, :] + rand_num * diff

        if self.bias.size > 0:
            rand_num = rand_num + self.bias

        if n == 1:
            return rand_num.reshape(-1)
        return rand_num


@dataclass
class UniqueBoundedSampler(SamplerBase):
    """
    Configuration for initial state sampling when generating MPC trajectories.
    
    Parameters
    ----------
    bounds : NDArray
        Sampling bounds interpreted as [lbx, ubx].
    percentages : NDArray | None
        Optional per-state percentages in (0, 1] used to symmetrically shrink `bounds` around their midpoint.
    min_dist : float | NDArray
        Minimum distance threshold for accepting a new sample relative to previously accepted samples.
    max_tries : int
        Maximum number of attempts to sample a unique initial state before raising an error.
    """
    percentages: NDArray | None = None
    min_dist: float | NDArray = 0.0
    max_tries: int = 1_000

    _min_dist_is_array: bool = field(init=False, repr=False, default=False)
    _uniqueness_disabled: bool = field(init=False, repr=False, default=False)
    _accepted_x0: list[NDArray] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        nx = self.bounds.shape[1]

        if self.max_tries < 1:
            raise ValueError("max_tries must be >= 1")

        # Treat non-scalars as per-state distance thresholds.
        self._min_dist_is_array = not np.isscalar(self.min_dist)
        if self._min_dist_is_array:
            self.min_dist = np.asarray(self.min_dist, dtype=float).reshape(-1)
            if self.min_dist.shape != (nx,):
                raise ValueError(f"min_dist vector must have shape ({nx},), got {self.min_dist.shape}")

        self._uniqueness_disabled = (
            (not self._min_dist_is_array and float(self.min_dist) <= 0.0)
            or (self._min_dist_is_array and bool(np.all(self.min_dist <= 0.0)))
        )

        # Minimum distance test
        if not self._min_dist_is_array and float(self.min_dist) < 0.0:
            raise ValueError("min_dist must be non-negative.")
        if self._min_dist_is_array and np.any(self.min_dist < 0.0):
            raise ValueError("min_dist vector must be non-negative component-wise.")

        if self.percentages is not None:
            percentages_arr = np.asarray(self.percentages, dtype=float).reshape(-1)
            if percentages_arr.shape != (nx,):
                raise ValueError(f"Percentage array must have shape ({nx},). Got {percentages_arr.shape}.")
            if np.any(percentages_arr <= 0) or np.any(percentages_arr > 1):
                raise ValueError("Percentages must be in the interval (0, 1].")
            self.bounds = self._calculate_percentage_bounds(self.bounds[0], self.bounds[1], percentages_arr)
            self.percentages = None

    def _x0_is_too_close(self, x0: NDArray, existing_x0: NDArray) -> bool:
        """Return True if `x0` is within the configured minimum distance of `existing_x0`.
        - scalar threshold: max_i |x0_i - existing_i| <= x0_min_dist
        - vector threshold: |x0_i - existing_i| <= x0_min_dist[i] for all i
        """
        if self._min_dist_is_array:
            return bool(np.all(np.abs(x0 - existing_x0) <= self.min_dist))
        return bool(np.max(np.abs(x0 - existing_x0)) <= self.min_dist)

    def sample_x0(self, n: int = 1) -> NDArray:
        """Sample one or multiple x0 values with optional uniqueness filtering."""
        if n < 1:
            raise ValueError("n must be >= 1.")

        if self._uniqueness_disabled:
            return super().sample_x0(n)

        accepted_batch: list[NDArray] = []
        tries = 0
        while len(accepted_batch) < n and tries < self.max_tries:
            tries += 1
            x0 = np.asarray(super().sample_x0(1), dtype=float).reshape(-1)
            is_close_existing = any(self._x0_is_too_close(x0, prev) for prev in self._accepted_x0)
            is_close_batch = any(self._x0_is_too_close(x0, prev) for prev in accepted_batch)
            if is_close_existing or is_close_batch:
                continue

            __logger__.debug(f"Accepted x0 ({x0}) after {tries} tries.")
            accepted_batch.append(x0)

        if len(accepted_batch) < n:
            raise RuntimeError(
                f"Failed to sample {n} unique x0 values within {self.max_tries} tries. "
                "Try decreasing `min_dist` or increasing `max_tries`."
            )

        self._accepted_x0.extend(accepted_batch)
        x0_arr = np.asarray(accepted_batch, dtype=float)
        if n == 1:
            return x0_arr.reshape(-1)
        return x0_arr

    @staticmethod
    def _calculate_percentage_bounds(lbx: NDArray, ubx: NDArray, percentages: NDArray) -> NDArray:
        """Shrink bounds symmetrically around the midpoint using the provided percentages.

        Returns
        -------
        bounds : NDArray
            Array with shape (2, nx) storing [lbx, ubx].
        """
        mid = 0.5 * (lbx + ubx)
        half_range = 0.5 * (ubx - lbx)
        shrink = (1.0 - percentages) * half_range

        sample_lb = mid - (half_range - shrink)
        sample_ub = mid + (half_range - shrink)

        if np.any(sample_lb >= sample_ub):
            raise ValueError("Computed sampling bounds are invalid (lower >= upper). Check percentages and solver bounds.")

        return np.stack((sample_lb, sample_ub), axis=0)
