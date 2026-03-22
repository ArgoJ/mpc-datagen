import time
import numpy as np

from numpy.typing import NDArray
from acados_template import AcadosOcpSolver, AcadosOcpBatchSolver

from pkg_logger import suppress_native_output, get_package_logger
from ..mpc_data import MPCData, MPCTrajectory, MPCMeta, MPCConfig
from ..extractor import extract_cfg, get_primary_solver, is_batch_solver

__logger__ = get_package_logger(__name__)








def solve_mpc_closed_loop(
    solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    x0: NDArray,
    reset_solver: bool = False,
    xeps_cfg: EpsBandConfig | None = None,
) -> MPCData | list[MPCData]:
    """
    Simulates a closed-loop MPC run using an Acados solver.

    Parameters
    ----------
    solver : AcadosOcpSolver | AcadosOcpBatchSolver
        The initialized Acados OCP solver (single or batch).
    x0 : NDArray
        Initial state for the closed-loop simulation. Should have shape (batch_size, nx).
    reset_solver : bool
        If True, resets the solver states to zero.
    xeps_cfg : EpsBandConfig | None
        Configuration for epsilon band checks.

    Returns
    -------
    MPCData | list[MPCData]
        The collected data from the closed-loop run.
    """
    cfg = extract_cfg(solver)
    n_batch = x0.shape[0] if x0.ndim == 2 else 1
    if (isinstance(cfg, list) and len(cfg) != n_batch) or (isinstance(cfg, MPCConfig) and n_batch != 1):
        raise ValueError(f"Length of cfg list ({len(cfg)}) must match batch size ({n_batch}).")

    data = []
    for cfg_i, x0_i in zip(cfg, x0) if isinstance(cfg, list) else [(cfg, x0)]:
        traj_i = MPCTrajectory.empty_from_cfg(cfg_i)
        traj_i.states[0, :] = x0_i.flatten()
        data_i = MPCData(config=cfg_i, trajectory=traj_i, meta=MPCMeta())
        data.append(data_i)

    current_x = x0.copy()
    in_eps_streak = 0
    is_sqp_solver = _is_sqp_solver(solver)
    nx = cfg[0].nx
    nu = cfg[0].nu

    __logger__.debug(f"Starting closed-loop simulation for T_sim={cfg.T_sim} steps. SQP solver: {is_sqp_solver}")

    sim_start_time = time.time()
    T_eff = 0
    for i in range(cfg.T_sim):

        if is_sqp_solver:
            x_guess = np.tile(current_x, cfg.N + 1)
            _set_sqp_x_guess(solver, x_guess)

        _set_initial_state_constraints(solver, current_x)

        with suppress_native_output(suppress_stdout=True, suppress_stderr=True):
            status = _solve_once(solver)

        __failed = False
        for data_i, status_i in zip(data, status):
            if status_i in (0, 5):
                data_i.meta.solver_status[i] = status_i
                __logger__.debug(f"Solver failed at step {i} with status {status}. Stopping.")
                __failed = True
            else:
                data_i.meta
            if __failed:
                break

        # Retrieve Predictions
        pred_x = solver.get_flat("x", n_batch=n_batch).reshape(-1, nx)
        pred_u = solver.get_flat("u", n_batch=n_batch).reshape(-1, nu)

        # Store predictions
        olve_times.append(solver.get_stats("time_tot"))
        traj.predicted_states[i, :, :] = pred_x
        traj.predicted_inputs[i, :, :] = pred_u
        traj.V_solver[i] = solver.get_cost()

        # Apply Control
        
        traj.inputs[i, :] = u_applied

        u_applied = pred_u[0:nx, :].flatten()

        # Optional early stop if within eps band
        should_break_eps = False
        if xeps_cfg is not None:
            # Only count successful solves towards the streak.
            if status == 0 and _in_state_band(current_x, cfg, xeps_cfg.eps_band):
                in_eps_streak += 1
            else:
                in_eps_streak = 0
            should_break_eps = (in_eps_streak >= xeps_cfg.eps_consecutive)

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

    # Construct Meta Information
    meta = MPCMeta(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sim_start_time)),
        solve_time_mean=float(np.mean(solve_times)) if solve_times else 0.0,
        solve_time_max=float(np.max(solve_times)) if solve_times else 0.0,
        solve_time_total=float(np.sum(solve_times)) if solve_times else 0.0,
        sim_duration_wall=sim_end_time - sim_start_time,
        steps_simulated=T_eff,
        status_codes=status_codes,
        feasible=is_feasible_run,
    )
    
    data = MPCData(
        config=cfg,
        trajectory=traj,
        meta=meta
    )
    data.finalize(recalculate_costs=True, truncate=True)
    return data
