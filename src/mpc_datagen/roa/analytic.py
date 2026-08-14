import logging
from math import gamma, pi
import numpy as np
import scipy.linalg as sla
from numpy.typing import NDArray

from ..mpc_data import MPCConfig
from ..adapters.acados import extract_QR
from ..utils.linalg import sym
from .reports import ConstraintLimit, AnalyticROAReport

__logger__ = logging.getLogger(__name__)


class ROAVerifier:
    """Computes the analytical Region of Attraction (ROA) / maximal invariant ellipsoid for unconstrained LQR feedback.

    Parameters
    ----------
    cfg : MPCConfig
        The MPC configuration containing linear system dynamics, quadratic cost weights, and constraints.
    """

    def __init__(self, cfg: MPCConfig):
        """Initializes the certifier with an MPC configuration.
        Automatically computes the LQR controller and the Lyapunov matrix P.
        """
        self.cfg = cfg

        self.P, self.K = self._solve_lqr()
        try:
            self.P_inv = sla.inv(self.P)
        except np.linalg.LinAlgError as e:
            __logger__.critical("Lyapunov matrix P is singular. System might be unobservable or unstable.")
            raise ValueError("Cannot compute ROA: P is singular.") from e

    def _solve_lqr(self) -> tuple[NDArray, NDArray]:
        """Internal method: Solves Riccati equation based on Config costs."""
        # NOTE: cfg.model.A/B are already discrete-time matrices as extracted from acados
        A = np.asarray(self.cfg.model.A, dtype=float)
        B = np.asarray(self.cfg.model.B, dtype=float)

        if A.shape[0] != A.shape[1] or A.shape[0] != B.shape[0]:
            raise ValueError(f"Model dimension mismatch: A={A.shape}, B={B.shape}")

        Q, R = extract_QR(self.cfg.cost.W, self.cfg.cost.Vx, self.cfg.cost.Vu)
        Q, R = sym(Q), sym(R)
        P = sla.solve_discrete_are(A, B, Q, R)

        # K = (R + B^T P B)^-1 (B^T P A)
        R_total = R + B.T @ P @ B
        K = sla.solve(R_total, B.T @ P @ A)

        return P, K

    def _calc_limit_c(self, h_vec: NDArray, k_val: float, name: str) -> float | None:
        """Calculates the max level set c for a single constraint h^T x <= k.
        Returns None if the constraint is not active or invalid.
        """
        if np.isinf(k_val):
            return None

        # If k < 0, the origin (x=0) is violated. c must be 0.
        if k_val < -1e-9:
            __logger__.warning(f"Constraint '{name}' excludes the origin (k={k_val:.4e}). ROA is empty set.")
            return 0.0

        # c = k^2 / ( 2 * h^T P^-1 h)
        denom = 2.0 * h_vec.T @ self.P_inv @ h_vec

        if denom > 1e-12:
            return float((k_val**2) / denom)
        else:
            return None

    def compute_report(self) -> AnalyticROAReport:
        """Evaluates all constraints and generates a comprehensive AnalyticROAReport.

        Returns
        -------
        report : AnalyticROAReport
            Detailed report including the active constraint, all constraint bounds, and ellipsoid volume.
        """
        nx = self.cfg.nx
        nu = self.cfg.nu
        cons = self.cfg.constraints

        limits: list[ConstraintLimit] = []

        def add(h: NDArray, k: float, n: str) -> None:
            abs_k = abs(k)
            val = self._calc_limit_c(h, abs_k, n)
            if val is not None:
                limits.append(ConstraintLimit(name=n, bound_value=float(k), c_limit=float(val)))

        # State Constraints
        if cons.has_bx():
            for i in range(nx):
                ei = np.zeros(nx)
                ei[i] = 1.0
                add(ei, cons.ubx[i], f"x_{i}_max")      # x_i <= ubx
                add(-ei, cons.lbx[i], f"x_{i}_min")     # -x_i <= -lbx

        # Input Constraints
        if cons.has_bu():
            for j in range(nu):
                kj = self.K[j, :]
                # u_j <= ubu  => -kj^T x <= ubu
                add(-kj, cons.ubu[j], f"u_{j}_max")

                # u_j >= lbu  => u_j >= lbu => -Kx >= lbu => Kx <= -lbu
                add(kj, cons.lbu[j], f"u_{j}_min")

        if not limits:
            __logger__.info("No active constraints found. ROA is unbounded.")
            return AnalyticROAReport(
                is_valid=True,
                is_bounded=False,
                c_min=float("inf"),
                active_constraint="None (Unbounded)",
                message="Unbounded: No active state or input constraints.",
            )

        # Find active constraint
        min_limit = min(limits, key=lambda cl: cl.c_limit)
        c_min = min_limit.c_limit

        for cl in limits:
            if np.isclose(cl.c_limit, c_min, rtol=1e-7, atol=1e-9):
                cl.is_active = True

        # Eigenvalues & volume
        p_sym = sym(self.P)
        eigs = np.linalg.eigvalsh(p_sym)
        eigenvalues_list = [float(e) for e in sorted(eigs)]

        # Volume of 0.5 * x^T P x <= c_min  <=>  x^T P x <= 2 * c_min
        volume = None
        if np.all(eigs > 0.0) and np.isfinite(c_min) and c_min > 0.0:
            unit_ball_vol = pi ** (0.5 * nx) / gamma(0.5 * nx + 1.0)
            volume = float(unit_ball_vol * ((2.0 * c_min) ** (0.5 * nx)) / np.sqrt(np.prod(eigs)))

        message = f"PASS: c_analytic = {c_min:.4f} (Active limit: {min_limit.name} = {min_limit.bound_value:.4f})."

        return AnalyticROAReport(
            method="Analytic LQR ROA",
            is_valid=True,
            is_bounded=True,
            c_min=c_min,
            active_constraint=min_limit.name,
            active_bound_value=min_limit.bound_value,
            constraint_limits=limits,
            ellipsoid_volume=volume,
            eigenvalues_P=eigenvalues_list,
            message=message,
        )

    def compute_min_c(self) -> float:
        """Iterates over all constraints in the config and returns the maximum level set value c.

        Returns
        -------
        c : float
            The scalar c such that 0.5 * x^T P x <= c satisfies all constraints.
            Returns infinity if no constraints are active.
        """
        report = self.compute_report()
        return float(report.c_min)

    def verify(self) -> AnalyticROAReport:
        """Convenience method returning the AnalyticROAReport."""
        return self.compute_report()

    def roa_bounds(self, n_points: int = 200) -> tuple[NDArray, float]:
        r"""Generates points on the boundary of the ellipsoid $V(x) = \frac{1}{2} x^T P x = c$.

        Parameters
        ----------
        n_points : int
            Number of boundary points to generate.

        Returns
        -------
        boundary : NDArray
            Matrix of shape (n_points, nx) with the coordinates of the boundary points.
        c_value : float
            The level set value c used.
        """
        c_value = self.compute_min_c()
        nx = self.cfg.nx

        # Random directions on the unit sphere in R^{nx}
        z = np.random.randn(nx, n_points)
        z /= np.linalg.norm(z, axis=0)

        # x = P^{-1/2} * z * sqrt(2 * c)
        try:
            L = np.linalg.cholesky(self.P_inv)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(self.P_inv)
            vals = np.maximum(vals, 0)
            L = vecs @ np.diag(np.sqrt(vals))

        # Factor sqrt(2 * c) due to 0.5 * x^T P x = c
        boundary = (L @ z) * np.sqrt(2.0 * c_value)

        return boundary.T, c_value


# Alias for explicit naming
AnalyticROAVerifier = ROAVerifier
