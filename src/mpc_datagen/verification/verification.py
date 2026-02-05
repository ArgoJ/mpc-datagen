import numpy as np

from collections.abc import Generator
from acados_template import AcadosOcpSolver

from .reports import *
from .gruene import grune_required_horizon_and_alpha
from ..extractor import MPCConfigExtractor, LinearSystemExtractor
from ..mpc_data import MPCData, MPCDataset, MPCConfig, MPCMeta, MPCTrajectory, LinearSystem
from ..package_logger import PackageLogger


__logger__ = PackageLogger.get_logger(__name__)
    

class StabilityVerifier:
    """
    A tool to rigorously check stability and performance properties of NMPC trajectories
    based on the optimal value function V_N as a Lyapunov function.
    """

    def __init__(self, dataset: MPCDataset, solver: AcadosOcpSolver | None = None):
        """Create a linear stability verifier over an entire dataset.

        Parameters
        ----------
        dataset : MPCDataset
            The dataset containing MPC trajectories and configurations.
        solver : AcadosOcpSolver, optional
            The Acados OCP solver instance used for linear stability verification.
        """
        self.dataset = dataset
        self.eps = 1e-6

        # Bindable entry 
        self._active_entry: MPCData | None = None
        self.traj : MPCTrajectory | None = None
        self.meta : MPCMeta | None = None
        
        self.cfg: MPCConfig | None = None
        self.sys: LinearSystem | None = None
        if isinstance(solver, AcadosOcpSolver):
            self.cfg = MPCConfigExtractor.get_cfg(solver)
            self.cfg.T_sim = dataset[0].config.T_sim
            self.sys = LinearSystemExtractor.get_system(solver)
        else:
            raise NotImplementedError(
                "StabilityVerifier currently only supports verification with AcadosOcpSolver instances.")

    def __getitem__(self, index: int) -> MPCData:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Generator[MPCData, None, None]:
        for entry in self.dataset:
            yield entry


    # --- INTERNAL HELPERS ---
    def _bind_entry(self, entry: MPCData) -> None:
        """Bind internal state to a specific trajectory entry."""
        self._active_entry = entry

        self.traj = entry.trajectory
        self.meta = entry.meta
        local_cfg = entry.config
        
        if self.cfg is None:
            self.cfg = local_cfg
        
        self._require_bound_entry()

        if local_cfg.dt != self.cfg.dt:
            raise ValueError(
                "Entry dt does not match solver dt. "
                "This verifier assumes a single dt for the entire dataset."
            )
        if local_cfg.N != self.cfg.N:
            raise ValueError(
                "Entry horizon N does not match solver N. "
                "This verifier assumes a single horizon N for the entire dataset."
            )
        if local_cfg.cost.yref.shape != self.cfg.cost.yref.shape \
            or not np.allclose(local_cfg.cost.yref, self.cfg.cost.yref, rtol=0.0, atol=0.0):
            raise ValueError(
                "Entry yref does not match solver yref. "
                "This verifier assumes a single yref for the entire dataset."
            )
        if self.cfg.T_sim != local_cfg.T_sim:
            __logger__.warning(
                f"Entry T_sim ({local_cfg.T_sim}) does not match configuration T_sim ({self.cfg.T_sim}). "
                "Using dataset T_sim for verification."
            )
            self.cfg.T_sim = local_cfg.T_sim

        if self.cfg.T_sim > self.traj.sim_steps:
            raise ValueError(
                "Entry T_sim exceeds trajectory length. T_sim must be <= trajectory length."
                f"{self.cfg.T_sim} > {self.traj.sim_steps}"
            )
        if self.cfg.constraints.has_bx_e() != local_cfg.constraints.has_bx_e():
            raise ValueError(
                "Entry and solver terminal state_bounds do not match. "
                "This verifier assumes terminal state_bounds for the entire dataset."
            )
        if self.cfg.cost.has_terminal_cost() != local_cfg.cost.has_terminal_cost():
            raise ValueError(
                "Entry and solver terminal_cost do not match. "
                "This verifier assumes terminal_cost for the entire dataset."
            )

    def _require_bound_entry(self) -> None:
        if self._active_entry is None or self.traj is None or self.cfg is None:
            raise ValueError(
                "No active entry is bound (internal error). Dataset-level methods must bind an entry before calling per-trajectory helpers."
            )


    # --- COST CALCULATIONS ---
    def _V(self, step_index: int) -> float:
        """Get V_N(x_step) using the selected value-function source."""
        self._require_bound_entry()
        if step_index < 0 or step_index >= len(self.traj.V_N):
            raise IndexError(f"Step index {step_index} out of range for trajectory costs.")
        return float(self.traj.V_N[step_index])

    def _l_star(self, x: np.ndarray) -> float:
        """Lower bound on l*(x) := min_u l(x,u) via unconstrained minimization.

        This ignores input constraints, yielding a conservative (small) lower bound on l*(x)
        for constrained problems. In Grüne's gamma estimate, this makes V/l* larger (safer).
        """
        x = np.asarray(x, dtype=float).reshape(-1)

        a = (self.cfg.cost.Vx @ x - self.cfg.cost.yref).reshape(-1)  # y = a + Vu u
        H = self.cfg.cost.Vu.T @ self.cfg.cost.W @ self.cfg.cost.Vu
        H = 0.5 * (H + H.T)
        g = (self.cfg.cost.Vu.T @ self.cfg.cost.W @ a).reshape(-1)

        if H.size == 0:
            return float(a.T @ self.cfg.cost.W @ a)

        try:
            u_star = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            u_star = -np.linalg.lstsq(H, g, rcond=None)[0]

        if not self.cfg.constraints.is_inside_bu(u_star):
            __logger__.error(
                "Unconstrained optimal control input u* is outside input bounds; "
                "l*(x) lower bound may be conservative."
            )
            return 0.0
        if not self.cfg.constraints.is_inside_bx(x):
            __logger__.error(
                "State x is outside state bounds; l*(x) lower bound may be invalid."
            )
            return 0.0

        return float(self.cfg.cost.get_stage_cost(x, u_star))


    # --- DATASET ITERATORS ---
    def iter_binded_entries(self, require_feasibility: bool = True) -> Generator[MPCData, None, None]:
        """Iterator over dataset entries with optional filtering/binding.
        
        Parameters
        ----------
        require_feasibility : bool
            Whether to only yield feasible entries.
            
        Yields
        -------
        entry : MPCData
            The next entry in the dataset.
        """
        for entry in self:
            if require_feasibility and (not entry.is_feasible()):
                continue
            
            self._bind_entry(entry)
            yield entry

    def get_feasible_dataset(self) -> MPCDataset:
        """Extract a feasible-only subset of the dataset."""
        feasible_entries: list[MPCData] = []
        for entry in self.dataset:
            if entry.is_feasible():
                feasible_entries.append(entry)
        return MPCDataset(data_buffer=feasible_entries)


    # --- LYAPUNOV DESCENT CHECK ---
    def lyapunov_descent(self) -> np.ndarray:
        """Calculate $V_N(x_{k+1}) - V_N(x_k)$ for the current trajectory.

        Returns
        -------
        diffs : np.ndarray
            The sequence of Lyapunov differences V_N(x_{k+1}) - V_N(x_k).
            Values <= 0 satisfy the decrease condition.
        """
        limit = min(self.cfg.T_sim, len(self.traj.V_N))
        if limit < 2:
            return np.array([])

        V = np.asarray(self.traj.V_N[:limit], dtype=float)
        diffs = V[1:] - V[:-1]
        valid_mask = np.isfinite(V[:-1]) & np.isfinite(V[1:])

        return diffs[valid_mask]

    def check_lyapunov_descent(self) -> LyapunovDescentReport:
        """
        Check for global non-increase (monotonicity) of the Lyapunov function V_N(x).
        
        Verifies $V_N(x_{k+1}) - V_N(x_k) \le 0$ for all trajectory steps, 
        ensuring stability even in regions with small stage cost (near equilibrium).

        Returns
        -------
        report : LyapunovDescentReport
            Report summarizing the descent check results.
        """
        max_increase = 0.0
        violation_count = 0
        total_steps = 0
        
        for _ in self.iter_binded_entries():
            diffs = self.lyapunov_descent()
            total_steps += len(diffs)
            
            # V_next - V_curr <= 0 (allowing for tolerance)
            # Violation if: V_next - V_curr > self.eps
            violation_mask = diffs > self.eps
            n_violations = np.sum(violation_mask)
            
            if n_violations > 0:
                violation_count += n_violations
                max_increase = max(max_increase, float(np.max(diffs[violation_mask])))
        
        is_stable = (violation_count == 0)
        msg = (f"{'Passed' if is_stable else 'Failed'}: "
               f"{violation_count}/{total_steps} violations, "
               f"max_increase={max_increase:.4e}")

        return LyapunovDescentReport(
            is_stable=is_stable,
            message=msg,
            max_increase=max_increase,
            violation_count=int(violation_count),
            total_steps=total_steps
        )


    # --- ALPHA-DECAY CHECK ---
    def alpha_and_max_violation(self, alpha_required: float = 1e-3) -> AlphaViolationStats:
        r"""
        Estimates the *observed* alpha and maximum violation for steps with significant stage cost.
        Verifies $V_N(x_{k+1}) - V_N(x_k) \le -\alpha \ell(x_k, u_k)$.

        Parameters
        ----------
        alpha_required : float
            Minimum empirical threshold required for verification.

        Returns
        -------
        stats : AlphaViolationStats
            The observed minimum alpha and maximum violation statistics.
        """
        T_limit = min(self.cfg.T_sim, len(self.traj.inputs), len(self.traj.states) - 1, len(self.traj.V_N) - 1)
        if T_limit < 1:
            return AlphaViolationStats()

        states = np.asarray(self.traj.states[:T_limit], dtype=float)
        inputs = np.asarray(self.traj.inputs[:T_limit], dtype=float)
        
        # Value function sequence: V_k, V_{k+1}
        V = np.asarray(self.traj.V_N, dtype=float)
        V_curr = V[:T_limit]
        V_next = V[1:T_limit+1]

        l_curr = self.cfg.cost.get_stage_cost(states, inputs)

        # --- Verification Logic ---
        v_finite_mask = np.isfinite(V_curr) & np.isfinite(V_next)            
        mask = v_finite_mask & (np.isfinite(l_curr) & (l_curr > self.eps))
        if not np.any(mask):
            return AlphaViolationStats()

        vc = V_curr[mask]
        vn = V_next[mask]
        lc = l_curr[mask]
        
        alpha_obs = (vc - vn) / lc
        
        # Filter non-finite alpha
        valid_alpha_mask = np.isfinite(alpha_obs)
        if np.any(valid_alpha_mask):
            min_alpha = float(np.min(alpha_obs[valid_alpha_mask]))
        else:
            min_alpha = float("nan")

        # V_next <= V_curr - alpha_req * l_curr
        rhs = vc - alpha_required * lc
        residual = vn - rhs
        violation = np.maximum(0.0, residual)
        
        max_violation = float(np.max(violation))
        min_residual = float(np.min(residual))

        return AlphaViolationStats(
            min_alpha=float(min_alpha), 
            max_violation=float(max_violation), 
            min_residual=float(min_residual), 
            n_used=int(np.sum(mask)))


    def asymptotic_stability(self, alpha_required: float = 1e-3) -> AsymptoticStabilityReport:
        """
        Checks for potential decrease (alpha-decay) where stage cost is significant.
        Using the minimum observed alpha and maximum violation over steps with l(x,u) > eps.

        Parameters
        ----------
        alpha_required : float
            Minimum empirical threshold required for verification.

        Returns
        -------
        report : AsymptoticStabilityReport
            A report indicating the empirical alpha and whether the decrease condition is satisfied.
        """
        alphas = []
        violations = []
        used = 0

        for _ in self.iter_binded_entries(require_feasibility=True):
            stats = self.alpha_and_max_violation(alpha_required=alpha_required)
            if stats.n_used == 0:
                continue
            if np.isfinite(stats.min_alpha):
                alphas.append(float(stats.min_alpha))
            violations.append(float(stats.max_violation))
            used += 1

        if used == 0:
            min_alpha = 0.0
            max_violation = float("inf")
            empirical_ok = False
        else:
            min_alpha = float(np.min(alphas)) if alphas else 0.0
            max_violation = float(np.max(violations))

        empirical_ok = bool((min_alpha >= alpha_required) and (max_violation <= 0.0))

        return AsymptoticStabilityReport(
            is_stable=empirical_ok,
            message=(
                f"{'Satisfied' if empirical_ok else 'Not satisfied'}: "
                f"min_alpha={min_alpha:.4e}, "
                f"max_violation={max_violation:.4e}, "
                f"alpha_required={alpha_required:.4e}."
            ),
            min_alpha=min_alpha,
            max_violation=max_violation
        )


    # --- GRÜNE CONDITION CHECK ---
    def gamma_estimates(self) -> list[float]:
        """Estimate the maximum gamma value over the dataset."""
        gamma_values: list[float] = []

        T_sim = min(self.cfg.T_sim, len(self.traj.states), len(self.traj.V_N))
        for n in range(T_sim):
            x = np.asarray(self.traj.states[n], dtype=float)
            if not np.all(np.isfinite(x)):
                __logger__.debug(f"Skipping step {n} due to non-finite state x={x}")
                continue

            Vn = float(self._V(n))
            if not np.isfinite(Vn):
                __logger__.debug(f"Skipping step {n} due to non-finite cost Vn={Vn:.4e}")
                continue

            # l*(x) := min_u l(x,u)
            lstar = float(self._l_star(x))

            if (not np.isfinite(lstar)) or (lstar <= self.eps):
                __logger__.debug(f"Skipping step {n} due to small stage cost l(x,u)={lstar:.4e} <= tol={self.eps:.4e}")
                continue

            gamma_values.append(float(Vn / lstar))
        return gamma_values

    def grüne_horizon_condition(self) -> GrüneHorizonReport:
        """Dataset-level Grüne horizon condition certification.

        Returns
        -------
        report : GrüneHorizonReport
            A report indicating applicability and estimates of gamma, alpha_N, and required horizon.
        """
        cfg0 = self.dataset[0].config
        has_terminal_cost = cfg0.cost.has_terminal_cost()
        has_terminal_bounds = cfg0.constraints.has_bx_e()

        if has_terminal_cost or has_terminal_bounds:
            return GrüneHorizonReport(
                applicability=False,
                message="Not applicable: Dataset includes terminal cost/bounds; no-terminal theorem does not directly apply")

        N = int(cfg0.N)
        gamma_values: list[float] = []
        
        for _ in self.iter_binded_entries():
            gamma_values.extend(
                self.gamma_estimates())
                

        if not gamma_values:
            return GrüneHorizonReport(
                applicability=False,
                message="Not applicable: insufficient data")
        
        gamma = float(np.max(gamma_values))
        N_required, alpha_N = grune_required_horizon_and_alpha(gamma=gamma, N=N)
        stable_flag = bool((N >= N_required) and (alpha_N > 0.0))
        
        return GrüneHorizonReport(
            applicability=True,
            gamma_estimate=gamma,
            alpha_N_estimate=alpha_N,
            required_horizon=N_required,
            is_stable=stable_flag,
            message=f"Grüne condition estimated with gamma={gamma:.4f}, alpha_N={alpha_N:.4f}, required_horizon={N_required}<={N}.")


    # --- Certification Interface ---
    @classmethod
    def verify(
        cls,
        dataset: MPCDataset,
        solver: AcadosOcpSolver | None = None,
        alpha_required: float = 1e-4,
    ) -> StabilityReport:
        """Dataset-level verification using the optimal value function as a Lyapunov candidate.
        
        Parameters
        ----------
        dataset : MPCDataset
            The dataset containing MPC trajectories and configurations.
        solver : AcadosOcpSolver, optional
            The Acados OCP solver instance used for linear stability verification.
        alpha_required : float
            Minimum empirical alpha required for verification.

        Returns
        -------
        StabilityReport
            Stability report indicating whether the dataset passes empirical checks.
        """
        verifier = StabilityVerifier(dataset, solver)

        # 1. Global Descent Check (Monotonicity)
        descent_report = verifier.check_lyapunov_descent()
        
        # 2. Asymptotic Stability (Alpha-Decay)
        asym_stab_report = verifier.asymptotic_stability(alpha_required=alpha_required)
        
        # 3. Grüne Condition
        grune_report = verifier.grüne_horizon_condition()
        gruene_pass = bool(grune_report.applicability and grune_report.is_stable)

        if asym_stab_report.is_stable and descent_report.is_stable:
            msg = (
                f"PASS. Lyapunov descent observed with min_alpha={asym_stab_report.min_alpha:.3e}, "
                f"alpha_required={alpha_required:.3e}, and no descent violations.")
        elif not asym_stab_report.is_stable and descent_report.is_stable:
            msg = (
                f"PASS. Asymptotic stability estimated with alpha={asym_stab_report.min_alpha:.3e} "
                f"and alpha_required={alpha_required:.3e}.")
        elif gruene_pass:
            msg = (
                f"PASS. Grüne horizon condition estimated with gamma={grune_report.gamma_estimate:.3e} "
                f"and required_horizon={grune_report.required_horizon}.")
        else:
            msg = (
                f"FAIL. lyapunov='{asym_stab_report.message}', "
                f"descent='{descent_report.message}', "
                f"grune='{grune_report.message}'.")

        return StabilityReport(
            method="Empirical Verification",
            is_stable=bool(gruene_pass or (asym_stab_report.is_stable and descent_report.is_stable)),
            details={
                "lyapunov_alpha_report": asym_stab_report,
                "lyapunov_descent_report": descent_report,
                "grune_report": grune_report,
            },
            message=msg,
        )