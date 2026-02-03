import numpy as np

from typing import Any, Literal, Optional, Tuple
from acados_template import AcadosOcpSolver

from .mpc_data import MPCConfig, LinearSystem
from .linalg import discretize_and_linearize_rk4
from .package_logger import PackageLogger

__logger__ = PackageLogger.get_logger(__name__)


# --- Helpers ---
def ensure_linear_ls_cost_type(cost_type: Literal['LINEAR_LS', 'NONLINEAR_LS']) -> None:
    """Ensure that only LINEAR_LS cost type is used."""
    if cost_type != 'LINEAR_LS':
        __logger__.warning("Only LINEAR_LS cost type is supported.")
        return False
    return True

def _is_none(*values: Optional[Any]) -> Any:
    """Ensure that a value is not None, otherwise raise an error."""
    for val in values:
        if val is None:
            return True
        
    return False

# --- Extracts ---
def extract_stage_reference(
    yref: Optional[np.ndarray],
    nx: int, 
    nu: int
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Extraction of (x*, u*) from yref."""
    if _is_none(yref, nx, nu):
        return None
    x_ref = np.zeros(nx)
    u_ref = np.zeros(nu)
    
    yref = np.asarray(yref).reshape(-1)
    if yref.size == (nx + nu):
        x_ref = yref[: nx].copy()
        u_ref = yref[nx : nx + nu].copy()
    else:
        raise NotImplementedError(
            "Cannot extract (x*, u*) from yref with unexpected size. Require size "
            f"nx + nu = {nx + nu}, got {yref.size}."
        )

    return x_ref, u_ref


def extract_terminal_reference(yref_e: Optional[np.ndarray], nx: int) -> Optional[np.ndarray]:
    """Extraction of x_e* from yref_e."""
    if _is_none(yref_e, nx):
        return None

    yref_e = np.asarray(yref_e).reshape(-1)
    if yref_e.shape[0] != nx:
        raise NotImplementedError(
            "Cannot extract x_e* from yref_e with unexpected size. Require size "
            f"nx = {nx}, got {yref_e.size}."
        )

    return yref_e.copy()


def indexed_bounds(
    lb: np.ndarray,
    ub: np.ndarray,
    idx: np.ndarray,
    dim: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Reconstruct full bounds vectors from acados indexed bounds."""
    if _is_none(lb, ub, idx):
        return None

    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    idx = np.asarray(idx, dtype=int).reshape(-1)

    if lb.size == 0 or ub.size == 0 or idx.size == 0:
        return None
    if lb.shape != ub.shape or lb.shape != idx.shape:
        return None
    if dim <= 0:
        return None

    full_lb = -np.inf * np.ones((dim,), dtype=float)
    full_ub = np.inf * np.ones((dim,), dtype=float)

    # Guard against out-of-range indices.
    if np.any(idx < 0) or np.any(idx >= dim):
        return None

    full_lb[idx] = lb
    full_ub[idx] = ub
    return full_lb, full_ub

def extract_QR(
    W: np.ndarray, 
    Vx: np.ndarray, 
    Vu: np.ndarray
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Extracts Q, R from the cost configuration."""
    if _is_none(W, Vx, Vu):
        return None

    W = np.asarray(W)
    Vx = np.asarray(Vx)
    Vu = np.asarray(Vu)

    Q = Vx.T @ W @ Vx
    R = Vu.T @ W @ Vu

    return Q, R

def extract_Qf(
    W_e: np.ndarray,
    Vx_e: np.ndarray
) -> Optional[np.ndarray]:
    """Extracts Qf from the terminal cost configuration."""
    if _is_none(W_e, Vx_e):
        return None

    W_e = np.asarray(W_e)
    Vx_e = np.asarray(Vx_e)

    Qf = Vx_e.T @ W_e @ Vx_e
    if np.allclose(Qf, 0.0, atol=0.0, rtol=0.0):
        return None
    return Qf


# --- Extractor class for AcadosOcpSolver ---
class MPCConfigExtractor():
    """Extractor for AcadosOcpSolver objects.

    This class extracts relevant matrices and parameters from an `AcadosOcpSolver`
    instance for use in Lyapunov verification routines.
    """

    def __init__(self, solver: AcadosOcpSolver) -> None:
        self.ocp = solver.acados_ocp
        self.cfg = MPCConfig(
            T_sim=0,  # not extractable here
            N=self.ocp.solver_options.N_horizon,
            nx=self.ocp.dims.nx,
            nu=self.ocp.dims.nu,
            dt=float(self.ocp.solver_options.tf) / float(self.ocp.solver_options.N_horizon),
        )
        self._extract_constraints()
        self._extract_cost()
        self._extract_model()
    
    @classmethod
    def get_cfg(cls, solver: AcadosOcpSolver) -> MPCConfig:
        """Get the extracted MPCConfig.
        
        Note
        ----
        T_sim needs to be set separately, as it is not extractable from AcadosOcpSolver.
        """
        extractor = cls(solver)
        return extractor.cfg

    def _extract_cost(self) -> None:
        """Extract the initial state x0 from acados constraints."""
        self.cfg.cost.Vx = self.ocp.cost.Vx
        self.cfg.cost.Vu = self.ocp.cost.Vu
        self.cfg.cost.W = self.ocp.cost.W
        self.cfg.cost.yref = self.ocp.cost.yref
        self.cfg.cost.Vx_e = self.ocp.cost.Vx_e
        self.cfg.cost.W_e = self.ocp.cost.W_e
        self.cfg.cost.yref_e = self.ocp.cost.yref_e
        
        if self.ocp.solver_options.cost_scaling is not None \
            and (not np.allclose(self.ocp.solver_options.cost_scaling[:-1], self.cfg.dt) \
            or self.ocp.solver_options.cost_scaling[-1] != 1.0):
            __logger__.warning(
                "Cost scaling is not supported in this extractor. "
                f"Using default stage_scale = {self.cfg.dt}, terminal_scale = 1.0. \n {self.ocp.solver_options.cost_scaling}")
        
        self.cfg.cost.stage_scale = self.cfg.dt
        self.cfg.cost.terminal_scale = 1.0

    def _extract_constraints(self) -> None:
        """Extract full input bounds from acados indexed bounds."""
        constr = self.ocp.constraints

        # Initial state
        self.cfg.constraints.x0 = constr.x0 if constr.x0 is not None else np.array([])

        # State bounds
        x_bounds = indexed_bounds(
            constr.lbx,
            constr.ubx,
            constr.idxbx,
            self.cfg.nx
        )
        if x_bounds is not None:
            self.cfg.constraints.lbx = x_bounds[0]
            self.cfg.constraints.ubx = x_bounds[1]

        # Input bounds
        u_bounds = indexed_bounds(
            constr.lbu,
            constr.ubu,
            constr.idxbu,
            self.cfg.nu
        )
        if u_bounds is not None:
            self.cfg.constraints.lbu = u_bounds[0]
            self.cfg.constraints.ubu = u_bounds[1]

        # Terminal state bounds
        x_e_bounds = indexed_bounds(
            constr.lbx_e,
            constr.ubx_e,
            np.arange(self.cfg.nx),
            self.cfg.nx
        )
        if x_e_bounds is not None:
            self.cfg.constraints.lbx_e = x_e_bounds[0]
            self.cfg.constraints.ubx_e = x_e_bounds[1]

    def _extract_model(self) -> None:
        """Extract the discretized model dynamics."""
        x_lin, u_lin = self._extract_x_and_u_lin()
        self.cfg.model = self._extract_discretized_dynamics(x_lin, u_lin, self.cfg.dt)

    def _extract_x_and_u_lin(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get linearization points for state and input."""
        if self.cfg.cost.yref.shape[0] != (self.cfg.nx + self.cfg.nu):
            raise ValueError(
                "Cannot extract linearization points from yref with unexpected size. "
                f"Expected size nx + nu = {self.cfg.nx + self.cfg.nu}, got {self.cfg.cost.yref.shape[0]}."
            )

        x_lin = self.cfg.cost.yref[:self.cfg.nx]
        u_lin = self.cfg.cost.yref[self.cfg.nx:]

        if x_lin is None:
            raise ValueError("Cannot extract linearization point for state: x_ref is None.")
        if u_lin is None:
            raise ValueError("Cannot extract linearization point for input: u_ref is None.")

        return x_lin, u_lin

    def _extract_discretized_dynamics(self, x_lin: np.ndarray, u_lin: np.ndarray, dt: float) -> LinearSystem:
        """Compute the discrete-time linearization (A, B, g).
        
        Parameters
        ----------
        x_lin, u_lin : np.ndarray
            Linearization points for state and input.
        dt : float
            Sampling time.

        Returns
        -------
        Ad, Bd : np.ndarray
            Discrete-time state and input matrices.
        gd : np.ndarray
            Affine offset term so that $x^+ \\approx Ad x + Bd u + gd$.
        """
        if self.ocp.solver_options.integrator_type != "ERK" or self.ocp.model.f_expl_expr is None:
            raise NotImplementedError("Only explicit ODE models are supported in this verifier.")

        if self.ocp.solver_options.sim_method_num_stages is not None and np.any(self.ocp.solver_options.sim_method_num_stages != 4):
            raise NotImplementedError("Only RK4 integration is supported in this verifier.")
        
        if self.ocp.solver_options.sim_method_num_steps is not None and np.any(self.ocp.solver_options.sim_method_num_steps < 1):
            raise NotImplementedError("Number of integration steps must be at least 1.")

        x = self.ocp.model.x
        u = self.ocp.model.u
        f_expr = self.ocp.model.f_expl_expr

        return LinearSystem(*discretize_and_linearize_rk4(
            x, u, f_expr, dt, x_lin, u_lin
        ))

    
class LinearSystemExtractor:
    """Extractor for AcadosOcpSolver model dynamics.
    
    This class extracts the nonlinear dynamics function from an `AcadosOcpSolver`
    instance for use in Lyapunov verification routines.
    """

    def __init__(self, solver: AcadosOcpSolver) -> None:
        self.ocp = solver.acados_ocp
        self.cfg = MPCConfigExtractor.get_cfg(solver)
        self.linear_system = self.cfg.model

    @classmethod
    def get_system(cls, solver: AcadosOcpSolver) -> LinearSystem:
        """Get the linearized system matrices from the solver."""
        extractor = cls(solver)
        return extractor.linear_system
