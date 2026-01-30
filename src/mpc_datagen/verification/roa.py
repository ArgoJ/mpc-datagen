import numpy as np
import scipy.linalg as sla

from ..mpc_data import MPCConfig
from ..package_logger import PackageLogger
from ..extractor import extract_QR

__logger__ = PackageLogger.get_logger(__name__)

class ROACertifier:
    def __init__(self, cfg: MPCConfig):
        """
        Initializes the certifier with an MPC configuration.
        Automatically computes the LQR controller and the Lyapunov matrix P.
        """
        self.cfg = cfg
        self.c_candidates = []

        self.P, self.K = self._solve_lqr()
        try:
            self.P_inv = sla.inv(self.P)
        except np.linalg.LinAlgError:
            __logger__.error("P is singular, cannot compute its inverse. ROA calculation may fail.")
            self.P_inv = np.eye(len(self.P)) 

    def _solve_lqr(self):
        """Internal method: Solves Riccati equation based on Config costs."""
        A = self.cfg.model.A
        B = self.cfg.model.B
        # TODO: discretize A and B
        
        Q, R = extract_QR(self.cfg.cost.W, self.cfg.cost.Vx, self.cfg.cost.Vu)
        P = sla.solve_discrete_are(A, B, Q, R)

        # Calculate K: u = -K * x
        # K = (R + B^T P B)^-1 (B^T P A)
        R_total = R + B.T @ P @ B
        K = sla.solve(R_total, B.T @ P @ A)
        
        return P, K

    def _add_limit(self, h_vec, k_val, name="Constraint"):
        """Adds a candidate c for a constraint h^T x <= k."""
        if np.isinf(k_val):
            return

        if k_val < -1e-9:
            __logger__.warning(f"Constraint '{name}' excludes the origin (k={k_val}). Setting c=0.")
            self.c_candidates.append(0.0)
            return

        # c = k^2 / (h^T P^-1 h)
        denom = h_vec.T @ self.P_inv @ h_vec
        
        if denom > 1e-12:
            c = (k_val**2) / denom
            self.c_candidates.append(c)

    @classmethod
    def max_roa(cls, cfg: MPCConfig) -> float:
        """Searches all constraints from the config and returns the maximum level set c."""
        roa = cls(cfg)
        
        nx = roa.cfg.nx
        nu = roa.cfg.nu
        cons = roa.cfg.constraints

        # State Constraints
        if cons.has_bx():
            for i in range(nx):
                ei = np.zeros(nx); ei[i] = 1.0
                # x_i <= ubx -> e_i^T x <= ubx
                roa._add_limit(ei, cons.ubx[i], f"x_{i}_max")
                # x_i >= lbx -> -e_i^T x <= -lbx
                roa._add_limit(-ei, -cons.lbx[i], f"x_{i}_min")
        # Input Constraints
        if cons.has_bu():
            for j in range(nu):
                kj = roa.K[j, :] 
                # u_j <= ubu -> -kj^T x <= ubu
                roa._add_limit(-kj, cons.ubu[j], f"u_{j}_max")
                # u_j >= lbu -> u_j >= lbu -> -Kx >= lbu -> Kx <= -lbu
                # kj^T x <= -lbu
                roa._add_limit(kj, -cons.lbu[j], f"u_{j}_min")
        if not roa.c_candidates:
            return float('inf')
            
        return min(roa.c_candidates)