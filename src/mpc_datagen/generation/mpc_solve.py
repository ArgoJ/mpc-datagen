import time
import logging
import numpy as np

from numpy.typing import NDArray
from acados_template import AcadosOcpSolver, AcadosSimSolver
from enum import IntEnum
from dataclasses import dataclass

from ..mpc_data import MPCData, MPCTrajectory, MPCMeta, MPCConfig
from ..extractor import MPCConfigExtractor

__logger__ = logging.getLogger(__name__)

class BreakOn(IntEnum):
    """Enumeration for conditions to break the closed-loop simulation early.
    
    Attributes
    ----------
    NONE
        No early stopping, run for the full T_sim.
    INFEASIBLE
        Stop if the solver returns an infeasibility status code.
    IN_EPS
        Stop if the state remains within an epsilon band around the reference for a specified number of consecutive steps.
    ALL
        Stop if all conditions (infeasibility and epsilon band) are met.
    """
    NONE = 0
    INFEASIBLE = 1
    IN_EPS = 3
    ALL = 4

@dataclass
class EpsBandConfig:
    """
    Configuration for epsilon band checks in closed-loop simulation.
    
    Parameters
    ----------
    eps_band : float | NDArray
        Epsilon band around the reference output `yref` used. 
        Can be a scalar (same band for all states) or a vector of shape (nx,) for per-state bands.  
        Default is 1e-2.
    eps_consecutive : int
        Number of consecutive steps within the eps_band required to trigger a break. Must be >= 1.  
        Default is 5.
    """
    eps_band: float | NDArray = 1e-2
    eps_consecutive: int = 5

    def __post_init__(self):
        if self.eps_consecutive < 1:
            raise ValueError("eps_consecutive must be >= 1.")


def _resolve_eps_band(nx: int, eps_band: float | NDArray) -> NDArray:
    """Normalize `eps_band` to a per-state vector of shape (nx,)."""
    if np.isscalar(eps_band):
        eps_vec = np.full(int(nx), float(eps_band), dtype=float)
    else:
        eps_vec = np.asarray(eps_band, dtype=float).reshape(-1)
        if eps_vec.shape != (int(nx),):
            raise ValueError(f"eps_band must be a scalar or shape ({int(nx)},), got {eps_vec.shape}")

    if not np.all(np.isfinite(eps_vec)):
        raise ValueError("eps_band must contain only finite values")
    if np.any(eps_vec < 0.0):
        raise ValueError("eps_band must be >= 0 component-wise")
    return eps_vec


def _in_state_band(x: NDArray, cfg: MPCConfig, eps_band: float | NDArray) -> bool:
    """Return True if |x - x_ref| <= eps_band component-wise.

    `eps_band` may be a scalar or a vector of shape (nx,) to account for different state scales.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    eps_vec = _resolve_eps_band(cfg.nx, eps_band)
    x_ref = cfg.cost.yref @ cfg.cost.Vx
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
    if x_ref.shape != x.shape:
        raise ValueError(f"x_ref shape mismatch: expected {x.shape}, got {x_ref.shape}")
    return bool(np.all(np.abs(x - x_ref) <= eps_vec))


def solve_mpc_closed_loop(
    solver: AcadosOcpSolver,
    integrator: AcadosSimSolver | None = None,
    cfg: MPCConfig | None = None,
    T_sim: int | None = None,
    break_on: BreakOn = BreakOn.INFEASIBLE,
    xeps_cfg: EpsBandConfig | None = None,
) -> MPCData:
    """
    Simulates a closed-loop MPC run using an Acados solver.

    Parameters
    ----------
    solver : AcadosOcpSolver
        The initialized Acados OCP solver.
    integrator : AcadosSimSolver, optional
        Acados integrator for accurate simulation steps.
    cfg : MPCConfig, optional
        Configuration dictionary to store in MPCData.
    T_sim : int, optional
        Number of simulation steps. If None, uses cfg.T_sim.
    break_on : BreakOn
        Condition to break the simulation loop.
    xeps_cfg : EpsBandConfig | None
        Configuration for epsilon band checks used when `break_on` is
        `BreakOn.IN_EPS` or `BreakOn.ALL`.

    Returns
    -------
    MPCData
        The collected data from the closed-loop run.
    """
    if cfg is None and T_sim is None:
        raise ValueError("Either cfg or T_sim must be provided.")
        
    if cfg is None:
        cfg = MPCConfigExtractor.get_cfg(solver)
        
    if T_sim is not None:
        cfg.T_sim = T_sim

    if break_on in (BreakOn.IN_EPS, BreakOn.ALL) and xeps_cfg is None:
        xeps_cfg = EpsBandConfig()
        
    # Initialize Trajectory container with NaNs
    traj = MPCTrajectory.empty_from_cfg(cfg)
    
    # Set initial state
    traj.states[0, :] = cfg.constraints.x0.copy()
    
    solve_times = []
    status_codes = []
    
    current_x = cfg.constraints.x0.copy()
    is_feasible_run = True
    in_eps_streak = 0

    sim_start_time = time.time()

    T_eff = 0
    for i in range(cfg.T_sim):
        solver.set(0, "lbx", current_x)
        solver.set(0, "ubx", current_x)
        
        status = solver.solve()
        if status != 0:
            if break_on in (BreakOn.INFEASIBLE, BreakOn.ALL) and status == 1:
                __logger__.warning(f"Solver failed at step {i} with status {status}. Stopping.")
                is_feasible_run = False
                break

        status_codes.append(status)
        solve_times.append(solver.get_stats("time_tot"))
                
        # Retrieve Predictions
        pred_x = np.zeros((cfg.N + 1, cfg.nx))
        for k in range(cfg.N + 1):
            pred_x[k, :] = solver.get(k, "x")
            
        pred_u = np.zeros((cfg.N, cfg.nu))
        for k in range(cfg.N):
            pred_u[k, :] = solver.get(k, "u")
            
        # Store predictions
        traj.predicted_states[i, :, :] = pred_x
        traj.predicted_inputs[i, :, :] = pred_u
        traj.V_solver[i] = solver.get_cost()
        
        # Apply Control
        u_applied = pred_u[0, :].flatten()
        traj.inputs[i, :] = u_applied

        # Optional early stop if within eps band
        should_break_eps = False
        if break_on in (BreakOn.IN_EPS, BreakOn.ALL):
            # Only count successful solves towards the streak.
            if status == 0 and _in_state_band(current_x, cfg, xeps_cfg.eps_band):
                in_eps_streak += 1
            else:
                in_eps_streak = 0
            should_break_eps = (in_eps_streak >= xeps_cfg.eps_consecutive)
        
        # Simulate System
        if integrator is not None:
            integrator.set("x", current_x)
            integrator.set("u", u_applied)
            
            status_sim = integrator.solve()
            if status_sim != 0:
                __logger__.error(f"Integrator failed at step {i} with status {status_sim}")
            
            current_x = integrator.get("x")
        else:
            current_x = pred_x[1, :].flatten()
        
        traj.states[i+1, :] = current_x

        T_eff += 1

        if should_break_eps:
            __logger__.debug(
                f"Breaking after {in_eps_streak} consecutive solves within eps_band={xeps_cfg.eps_band} around x_ref. "
                f"(step={i})"
            )
            break

    sim_end_time = time.time()
    
    # Update feasibility
    traj.feasible = is_feasible_run

    # Construct Meta Information
    meta = MPCMeta(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sim_start_time)),
        solve_time_mean=float(np.mean(solve_times)) if solve_times else 0.0,
        solve_time_max=float(np.max(solve_times)) if solve_times else 0.0,
        solve_time_total=float(np.sum(solve_times)) if solve_times else 0.0,
        sim_duration_wall=sim_end_time - sim_start_time,
        steps_simulated=T_eff,
        status_codes=status_codes
    )
    
    data = MPCData(
        config=cfg,
        trajectory=traj,
        meta=meta
    )
    data.finalize(recalculate_costs=True, truncate=True)
    return data
