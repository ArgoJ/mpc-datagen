# %% [markdown] 
# # Double Integrator Example


# %% General Imports
import numpy as np
from scipy.linalg import solve_discrete_are, block_diag
from casadi import SX
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from typing import Optional, Tuple

import mpc_datagen.linalg as mdg_linalg
import mpc_datagen.plots as mdg_plots
from mpc_datagen import *
from mpc_datagen.verification import (
    StabilityCertifier,
    StabilityVerifier,
    VerificationRender,
    ROACertifier,
)



# %% Model Definition
def get_model(A: np.ndarray, B: np.ndarray) -> AcadosModel:
    """Create and return an `AcadosModel` for the linear system.
    
    Parameters
    ----------
    A : np.ndarray
        State matrix of the discrete-time double integrator.
    B : np.ndarray
        Input matrix of the discrete-time double integrator.

    Returns
    -------
    model : AcadosModel
        Configured AcadosModel object.
    """
    nx = A.shape[0]
    nu = B.shape[1]

    # states
    x = SX.sym("x", nx)

    # control
    u = SX.sym("u", nu)
    
    A_sx = SX(A)
    B_sx = SX(B)
    
    # Continuous dynamics
    x_dot = A_sx @ x + B_sx @ u

    model = AcadosModel()
    model.name = "double_integrator"
    model.x = x
    model.u = u
    model.f_expl_expr = x_dot
    model.disc_dyn_expr = None

    return model


# %% OCP Solver Definition
def get_ocp_solver(
    A_c: np.ndarray, 
    B_c: np.ndarray, 
    Q: np.ndarray, 
    R: np.ndarray,
    P: Optional[np.ndarray] = None,
    dt: float = 0.1, 
    N: int = 20,
    tol: float = 1e-8,
    terminal_mode: str = "regional",
    bounds_scale: float = 10.0,
    terminal_box_halfwidth: float = 1.0,
) -> Tuple[AcadosOcpSolver, dict]:
    """Create an acados OCP solver for a continuous-time linear system.

    Parameters
    ----------
    A_c, B_c : np.ndarray
        Continuous system matrices (dot(x) = Ax + Bu).
    Q, R : np.ndarray
        Stage cost matrices (x'Qx + u'Ru).
    P : np.ndarray, optional
        Terminal cost matrix (x_N' P x_N). If None, calculated via DARE on discretized system.
    dt : float
        Sampling time in seconds.
    N : int
        Number of control intervals.
    tol : float
        Solver tolerances for the QP solver.

    Returns
    -------
    solver : AcadosOcpSolver
        Constructed acados OCP solver.
    info : dict
        Useful information about the problem (A_d, B_d, P used).
    """
    nx = A_c.shape[0]
    nu = B_c.shape[1]

    ocp = AcadosOcp()
    ocp.model = get_model(A_c, B_c)

    # Calculate DARE
    A_d, B_d = mdg_linalg.lin_c2d_rk4(A_c, B_c, dt, num_steps=1)

    if P is None and terminal_mode == "regional":
        P = solve_discrete_are(A_d, B_d, Q, R)

    # Solver options
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = dt * N
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.qp_solver_tol_stat = tol  # Gradienten-Check
    ocp.solver_options.qp_solver_tol_eq   = tol  # Equality constraints
    ocp.solver_options.qp_solver_tol_ineq = tol  # Inequality constraints
    ocp.solver_options.qp_solver_tol_comp = tol  # Complementarity

    # Erhöhe zur Sicherheit die maximalen Iterationen, falls er länger braucht
    ocp.solver_options.qp_solver_iter_max = 100

    # Cost setup
    ocp.cost.cost_type = "LINEAR_LS"

    W = block_diag(Q, R)
    ocp.cost.W = W
    ocp.cost.Vx = np.vstack((np.eye(nx), np.zeros((nu, nx))))
    ocp.cost.Vu = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
    ocp.cost.yref = np.zeros((nx + nu,))

    # Terminal cost / ingredients
    ocp.cost.cost_type_e = "LINEAR_LS"
    if terminal_mode == "regional":
        ocp.cost.W_e = P
    else:
        # For "no terminal" scheme and for equilibrium terminal constraints,
        ocp.cost.W_e = np.zeros((nx, nx))
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.zeros((nx,))

    # Constraints
    ocp.constraints.x0 = np.zeros((nx,))

    # (Large) box constraints, used for feasibility and for the LQR terminal certificate sizing.
    ocp.constraints.lbu = -bounds_scale * np.ones((nu,))
    ocp.constraints.ubu = bounds_scale * np.ones((nu,))
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.lbx = -bounds_scale * np.ones((nx,))
    ocp.constraints.ubx = bounds_scale * np.ones((nx,))
    ocp.constraints.idxbx = np.arange(nx)

    if terminal_mode == "regional":
        # A small terminal box around the equilibrium (proxy for a local terminal set X_f).
        hw = float(terminal_box_halfwidth)
        ocp.constraints.lbx_e = -hw * np.ones((nx,))
        ocp.constraints.ubx_e = hw * np.ones((nx,))
        ocp.constraints.idxbx_e = np.arange(nx)
    if terminal_mode == "equilibrium":
        # Exact equilibrium terminal constraint x(N) = 0.
        ocp.constraints.lbx_e = np.zeros((nx,))
        ocp.constraints.ubx_e = np.zeros((nx,))
        ocp.constraints.idxbx_e = np.arange(nx)

    solver = AcadosOcpSolver(ocp, json_file=f"{ocp.model.name}_ocp.json")

    info = {
        "A_d": A_d,
        "B_d": B_d,
        "P": P,
        "terminal_mode": terminal_mode,
        "bounds_scale": bounds_scale,
    }

    return solver, info


# %%  
if __name__ == "__main__":
    # Continuous-time double integrator matrices (standard)
    A_c = np.array([[0, 1],
                    [0, 0]])
    B_c = np.array([[0],
                    [1]])

    # Cost matrices
    Q = np.diag([15.0, 1.0])
    R = np.diag([0.1])

    def run_case(
        name: str,
        terminal_mode: str,
        N: int,
        x0_bounds: np.ndarray,
        T_sim: int = 30,
        n_samples: int = 10,
        bounds_scale: float = 10.0,
        terminal_box_halfwidth: float = 1.0,
    ) -> None:
        print("\n" + "=" * 80)
        print(f"CASE: {name} (terminal_mode={terminal_mode}, N={N})")
        print("=" * 80)

        dt = 0.1
        solver, info = get_ocp_solver(
            A_c, B_c,
            Q, R,
            dt=dt,
            N=N,
            tol=1e-8,
            terminal_mode=terminal_mode,
            bounds_scale=bounds_scale,
            terminal_box_halfwidth=terminal_box_halfwidth,
        )

        generator = MPCDataGenerator(
            solver=solver,
            x0_bounds=x0_bounds,
            T_sim=T_sim,
            verbose=True,
            reset_solver=True,
            break_on=BreakOn.INFEASIBLE,
        )
        dataset = generator.generate(n_samples=n_samples)
        dataset.validate()
        dataset.save(f"data/double_integrator_{terminal_mode}_N{N}_data")

        # 1) Empirical verifier aggregated over the dataset
        cert_stats = StabilityCertifier.certify(solver)
        veri_stats = StabilityVerifier.verify(dataset, solver, alpha_required=1e-4)
        VerificationRender(cert_stats).render()
        VerificationRender(veri_stats).render()
        
        
        subdataset = dataset[:min(100, n_samples)]
        mdg_plots.relaxed_dp_residual(
            dataset=subdataset,
            html_path=f"plots/double_integrator_{terminal_mode}_N{N}_relaxed_dp_res.html",)
        mdg_plots.cost_descent(
            dataset=subdataset,
            html_path=f"plots/double_integrator_{terminal_mode}_N{N}_cost_descent.html",)
        mdg_plots.mpc_trajectories(
            dataset=subdataset,
            state_labels=["Position", "Velocity"],
            control_labels=["Acceleration"],
            time_bound=T_sim * dt,
            html_path=f"plots/double_integrator_{terminal_mode}_N{N}_trajectories.html",)
        
        P = info["P"]
        if P is not None:
            lyap_fun = lambda x: 0.5 * mdg_linalg.weighted_quadratic_norm(x, P)
            mdg_plots.lyapunov(
                dataset=subdataset,
                lyapunov_func=lyap_fun,
                state_labels=["x", "v"],
                plot_3d=True,
                use_optimal_v=False,
                html_path=f"plots/double_integrator_{terminal_mode}_N{N}_lyapunov.html",)
            
            roa_cert = ROACertifier(subdataset[0].config)
            roa_bounds, c_min = roa_cert.roa_bounds()
            mdg_plots.roa(
                lyapunov_func=lyap_fun,
                c_level=c_min,
                roa_bounds=roa_bounds,
                state_labels=["x", "v"],
                html_path=f"plots/double_integrator_{terminal_mode}_N{N}_roa.html",
            )
        

    # Case 1: regional terminal cost + small terminal set (should pass regional proof + empirical)
    run_case(
        name="Regional terminal ingredients",
        terminal_mode="regional",
        N=20,
        x0_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        T_sim=25,
        n_samples=50,
        bounds_scale=10.0,
        terminal_box_halfwidth=1.0,
    )

    # Case 2: equilibrium terminal constraint x(N)=0 (sample close so feasibility is easy)
    run_case(
        name="Equilibrium terminal constraint",
        terminal_mode="equilibrium",
        N=25,
        x0_bounds=np.array([[-0.5, -0.5], [0.5, 0.5]]),
        T_sim=25,
        n_samples=50,
        bounds_scale=10.0,
        terminal_box_halfwidth=1.0,
    )

    # Case 3: no terminal ingredients (zero terminal weight, no terminal bounds)
    run_case(
        name="No terminal ingredients (Grüne horizon condition)",
        terminal_mode="none",
        N=40,
        x0_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        T_sim=25,
        n_samples=50,
        bounds_scale=50.0,
        terminal_box_halfwidth=1.0,
    )