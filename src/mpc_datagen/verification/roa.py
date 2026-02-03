import numpy as np
import scipy.linalg as sla
from typing import List, Optional

from ..mpc_data import MPCConfig
from ..package_logger import PackageLogger
from ..extractor import extract_QR
from ..linalg import lin_c2d_rk4, sym

__logger__ = PackageLogger.get_logger(__name__)

class ROACertifier:
    def __init__(self, cfg: MPCConfig):
        """
        Initializes the certifier with an MPC configuration.
        Automatically computes the LQR controller and the Lyapunov matrix P.
        """
        self.cfg = cfg

        self.P, self.K = self._solve_lqr()
        try:
            self.P_inv = sla.inv(self.P)
        except np.linalg.LinAlgError as e:
            __logger__.critical("Lyapunov matrix P is singular. System might be unobservable or unstable.")
            raise ValueError("Cannot compute ROA: P is singular.") from e 

    def _solve_lqr(self):
        """Internal method: Solves Riccati equation based on Config costs."""
        A = self.cfg.model.A
        B = self.cfg.model.B
        A, B = lin_c2d_rk4(A, B, self.cfg.dt)

        if A.shape[0] != A.shape[1] or A.shape[0] != B.shape[0]:
            raise ValueError(f"Model dimension mismatch: A={A.shape}, B={B.shape}")
        
        Q, R = extract_QR(self.cfg.cost.W, self.cfg.cost.Vx, self.cfg.cost.Vu)
        Q, R = sym(Q), sym(R)
        P = sla.solve_discrete_are(A, B, Q, R)

        # K = (R + B^T P B)^-1 (B^T P A)
        R_total = R + B.T @ P @ B
        K = sla.solve(R_total, B.T @ P @ A)
        
        return P, K

    def _calc_limit_c(self, h_vec: np.ndarray, k_val: float, name: str) -> Optional[float]:
        """
        Calculates the max level set c for a single constraint h^T x <= k.
        Returns None if the constraint is not active or invalid.
        """
        if np.isinf(k_val):
            return None

        # If k < 0, the origin (x=0) is violated. c must be 0.
        if k_val < -1e-9:
            __logger__.warning(f"Constraint '{name}' excludes the origin (k={k_val:.4e}). ROA is empty set.")
            return 0.0

        # c = k^2 / (h^T P^-1 h)
        denom = h_vec.T @ self.P_inv @ h_vec

        if denom > 1e-12:
            return (k_val**2) / denom
        else:
            return None

    def compute_min_c(self) -> float:
        """
        Iterates over all constraints in the config and returns the maximum level set value c.
        
        Returns:
            float: The scalar c such that x^T P x <= c satisfies all constraints.
                   Returns infinity if no constraints are active.
        """
        nx = self.cfg.nx
        nu = self.cfg.nu
        cons = self.cfg.constraints
        
        candidates: List[float] = []
    
        def add(h, k, n):
            val = self._calc_limit_c(h, k, n)
            if val is not None:
                candidates.append(val)

        # State Constraints
        if cons.has_bx():
            for i in range(nx):
                ei = np.zeros(nx); ei[i] = 1.0
                add(ei, cons.ubx[i], f"x_{i}_max")      # x_i <= ubx
                add(-ei, -cons.lbx[i], f"x_{i}_min")    # -x_i <= -lbx

        # Input Constraints
        if cons.has_bu():
            for j in range(nu):
                kj = self.K[j, :] 
                # u_j <= ubu  => -kj^T x <= ubu
                add(-kj, cons.ubu[j], f"u_{j}_max")
                
                # u_j >= lbu  => u_j >= lbu => -Kx >= lbu => Kx <= -lbu
                # kj^T x <= -lbu
                add(kj, -cons.lbu[j], f"u_{j}_min")

        if not candidates:
            __logger__.info("No active constraints found. ROA is unbounded.")
            return float('inf')
            
        c_min = min(candidates)
        __logger__.info(f"Computed max ROA level set c = {c_min:.4f}")
        return c_min

    def roa_bounds(self, n_points: int = 200) -> np.ndarray:
        """
        Generates points on the boundary of the ellipsoid $V(x) = x^T P x = c$.
        
        Mathematics
        -----------
            We search x such that $x^T P x = c$.  
            We use the Cholesky decomposition of the inverse: $$P^{-1} = L L^T$$.  
            Then for a vector z on the unit sphere ($z^T z = 1$), we have:  
            $$x = \sqrt{c} \cdot L \cdot z$$
        
        Parameters
        ----------
        n_points : int
            Number of points to generate.
            
        Returns
        -------
        boundary : np.ndarray 
            Matrix of shape (n_points, nx) with the coordinates of the boundary points.
        c_value : float
            The level set value c used.
        """
        c_value = self.compute_min_c()
        nx = self.cfg.nx
        
        # Random directions of the unit sphere in $\mathcal{R}^{nx}$
        z = np.random.randn(nx, n_points)
        z /= np.linalg.norm(z, axis=0) # Norm on 1

        # $x = P^{-1/2} * z * \sqrt{c}$
        try:
            # $P_{inv} = L @ L^T$
            L = np.linalg.cholesky(self.P_inv)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(self.P_inv)
            # Clip negative Eigenvalues (should not exist for P > 0, but possible numerically)
            vals = np.maximum(vals, 0)
            L = vecs @ np.diag(np.sqrt(vals))

        # $x = L * z * \sqrt{c}$
        boundary = (L @ z) * np.sqrt(c_value)
        
        return boundary.T, c_value