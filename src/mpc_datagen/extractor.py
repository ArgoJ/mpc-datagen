import numpy as np
import logging

from numpy.typing import NDArray
from typing import Any, Literal
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpBatchSolver

from .mpc_data import MPCConfig, LinearSystem, LinearLSCost, Constraints
from .linalg import discretize_and_linearize_rk4

__logger__ = logging.getLogger(__name__)


# --- Helpers ---
def ensure_linear_ls_cost_type(cost_type: Literal['LINEAR_LS', 'NONLINEAR_LS']) -> None:
    """Ensure that only LINEAR_LS cost type is used."""
    if cost_type != 'LINEAR_LS':
        __logger__.warning("Only LINEAR_LS cost type is supported.")
        return False
    return True

def _is_none(*values: Any) -> Any:
    """Ensure that a value is not None."""
    for val in values:
        if val is None:
            return True
        
    return False

def is_batch_solver(solver: AcadosOcpSolver | AcadosOcpBatchSolver) -> bool:
    return isinstance(solver, AcadosOcpBatchSolver)

def get_primary_solver(solver: AcadosOcpSolver | AcadosOcpBatchSolver) -> AcadosOcpSolver:
    if is_batch_solver(solver):
        return solver.ocp_solvers[0]
    return solver

def validate_solver(solver: AcadosOcpSolver | AcadosOcpBatchSolver) -> None:
    if not isinstance(solver, (AcadosOcpSolver, AcadosOcpBatchSolver)):
        raise ValueError("Solver must be an instance of AcadosOcpSolver or AcadosOcpBatchSolver.")

def resolve_solver(solver: AcadosOcpSolver | AcadosOcpBatchSolver) -> AcadosOcpSolver | AcadosOcpBatchSolver:
    validate_solver(solver)
    return get_primary_solver(solver)


# --- Extracts ---
def extract_stage_reference(
    yref: NDArray | None,
    nx: int, 
    nu: int
) -> tuple[NDArray, NDArray] | None:
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


def extract_terminal_reference(yref_e: NDArray | None, nx: int) -> NDArray | None:
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
    lb: NDArray | None,
    ub: NDArray | None,
    idx: NDArray | None,
    dim: int,
) -> tuple[NDArray, NDArray] | None:
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
    W: NDArray, 
    Vx: NDArray, 
    Vu: NDArray
) -> tuple[NDArray, NDArray] | None:
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
    W_e: NDArray,
    Vx_e: NDArray
) -> NDArray | None:
    """Extracts Qf from the terminal cost configuration."""
    if _is_none(W_e, Vx_e):
        return None

    W_e = np.asarray(W_e)
    Vx_e = np.asarray(Vx_e)

    Qf = Vx_e.T @ W_e @ Vx_e
    if np.allclose(Qf, 0.0, atol=0.0, rtol=0.0):
        return None
    return Qf


def extract_cost(ocp: AcadosOcp, dt: float) -> LinearLSCost:
    """Extract the initial state x0 from acados constraints."""
    cost = LinearLSCost()
    cost.Vx = ocp.cost.Vx
    cost.Vu = ocp.cost.Vu
    cost.W = ocp.cost.W
    cost.yref = ocp.cost.yref
    cost.Vx_e = ocp.cost.Vx_e
    cost.W_e = ocp.cost.W_e
    cost.yref_e = ocp.cost.yref_e

    if ocp.solver_options.cost_scaling is not None \
        and (not np.allclose(ocp.solver_options.cost_scaling[:-1], dt) \
        or ocp.solver_options.cost_scaling[-1] != 1.0):
        __logger__.warning(
            "Cost scaling is not supported in this extractor. "
            f"Using default stage_scale = {dt}, terminal_scale = 1.0. \n {ocp.solver_options.cost_scaling}")
    
    cost.stage_scale = dt
    cost.terminal_scale = 1.0
    return cost


def extract_constraints(ocp: AcadosOcp, nx: int, nu: int) -> Constraints:
    """Extract full input bounds from acados indexed bounds."""
    constr = ocp.constraints
    constraints = Constraints()

    # Initial state
    constraints.x0 = constr.x0 if constr.x0 is not None else np.array([])

    # State bounds
    x_bounds = indexed_bounds(
        constr.lbx,
        constr.ubx,
        constr.idxbx,
        nx
    )
    if x_bounds is not None:
        constraints.lbx = x_bounds[0]
        constraints.ubx = x_bounds[1]

    # Input bounds
    u_bounds = indexed_bounds(
        constr.lbu,
        constr.ubu,
        constr.idxbu,
        nu
    )
    if u_bounds is not None:
        constraints.lbu = u_bounds[0]
        constraints.ubu = u_bounds[1]

    # Terminal state bounds
    x_e_bounds = indexed_bounds(
        constr.lbx_e,
        constr.ubx_e,
        np.arange(nx),
        nx
    )
    if x_e_bounds is not None:
        constraints.lbx_e = x_e_bounds[0]
        constraints.ubx_e = x_e_bounds[1]

    return constraints


def extract_x_and_u_lin(cost: LinearLSCost, nx: int, nu: int) -> tuple[NDArray, NDArray]:
    """Get linearization points for state and input."""
    if cost.yref.shape[0] != (nx + nu):
        raise ValueError(
            "Cannot extract linearization points from yref with unexpected size. "
            f"Expected size nx + nu = {nx + nu}, got {cost.yref.shape[0]}."
        )

    x_lin = cost.yref[:nx]
    u_lin = cost.yref[nx:]

    if x_lin is None:
        raise ValueError("Cannot extract linearization point for state: x_ref is None.")
    if u_lin is None:
        raise ValueError("Cannot extract linearization point for input: u_ref is None.")

    return x_lin, u_lin


def extract_discretized_dynamics(
    ocp: AcadosOcp,
    x_lin: NDArray,
    u_lin: NDArray,
    dt: float,
) -> LinearSystem:
    """Compute the discrete-time linearization (A, B, g).
    
    Parameters
    ----------
    x_lin, u_lin : NDArray
        Linearization points for state and input.
    dt : float
        Sampling time.

    Returns
    -------
    Ad, Bd : NDArray
        Discrete-time state and input matrices.
    gd : NDArray
        Affine offset term so that $x^+ \\approx Ad x + Bd u + gd$.
    """
    if ocp.solver_options.integrator_type != "ERK" or ocp.model.f_expl_expr is None:
        raise NotImplementedError("Only explicit ODE models are supported in this verifier.")

    if ocp.solver_options.sim_method_num_stages is not None and np.any(ocp.solver_options.sim_method_num_stages != 4):
        raise NotImplementedError("Only RK4 integration is supported in this verifier.")
    
    if ocp.solver_options.sim_method_num_steps is not None and np.any(ocp.solver_options.sim_method_num_steps < 1):
        raise NotImplementedError("Number of integration steps must be at least 1.")

    x = ocp.model.x
    u = ocp.model.u
    f_expr = ocp.model.f_expl_expr

    return LinearSystem(*discretize_and_linearize_rk4(
        x, u, f_expr, dt, x_lin, u_lin
    ))


def extract_model(
    ocp: AcadosOcp,
    cost: LinearLSCost,
    nx: int,
    nu: int,
    dt: float,
) -> LinearSystem:
    """Extract the discretized model dynamics."""
    x_lin, u_lin = extract_x_and_u_lin(cost, nx, nu)
    return extract_discretized_dynamics(ocp, x_lin, u_lin, dt)


def _extract_single_cfg(solver: AcadosOcpSolver) -> MPCConfig:
    ocp = solver.acados_ocp
    cfg = MPCConfig(
        T_sim=0,
        N=ocp.solver_options.N_horizon,
        nx=ocp.dims.nx,
        nu=ocp.dims.nu,
        dt=float(ocp.solver_options.tf) / float(ocp.solver_options.N_horizon),
    )
    cfg.constraints = extract_constraints(ocp, cfg.nx, cfg.nu)
    cfg.cost = extract_cost(ocp, cfg.dt)
    cfg.model = extract_model(ocp, cfg.cost, cfg.nx, cfg.nu, cfg.dt)
    return cfg


def extract_cfg(solver: AcadosOcpSolver | AcadosOcpBatchSolver) -> MPCConfig | list[MPCConfig]:
    """Extract MPC configuration from the given solver or solvers."""
    if is_batch_solver(solver):
        return [_extract_single_cfg(ocp_solver) for ocp_solver in solver.ocp_solvers]
    else:
        return _extract_single_cfg(solver)