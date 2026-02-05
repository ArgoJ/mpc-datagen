import numpy as np

from typing import Generator, Optional, List
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

    def __init__(self, dataset: MPCDataset, solver: Optional[AcadosOcpSolver] = None):
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
        self._active_entry: Optional[MPCData] = None
        self.traj : Optional[MPCTrajectory] = None
        self.meta : Optional[MPCMeta] = None
        
        self.cfg: Optional[MPCConfig] = None
        self.sys: Optional[LinearSystem] = None
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
        Generator[MPCData, None, None]
            The next entry in the dataset.
        """
        for entry in self:
            if require_feasibility and (not entry.is_feasible()):
                continue
            
            self._bind_entry(entry)
            yield entry

    def get_feasible_dataset(self) -> MPCDataset:
        """Extract a feasible-only subset of the dataset."""
        feasible_entries: List[MPCData] = []
        for entry in self.dataset:
            if entry.is_feasible():
                feasible_entries.append(entry)
        return MPCDataset(data_buffer=feasible_entries)


    # --- LYAPUNOV STABILITY CHECKS ---
    def lyapunov_decrease(self) -> List[float]:
        """Calculate $V_N(x_{k+1}) - V_N(x_k)$ for the current trajectory.

        Returns
        -------
        List[float]
            The sequence of Lyapunov differences V_N(x_{k+1}) - V_N(x_k).
            Values <= 0 satisfy the decrease condition.
        """
        diffs: List[float] = []

        for n in range(self.cfg.T_sim - 1):
            V_curr = self._V(n)
            V_next = self._V(n + 1)

            if not (np.isfinite(V_curr) and np.isfinite(V_next)):
                __logger__.debug(f"Skipping step {n} due to non-finite value function V_N(x): V_curr={V_curr:.4e}, V_next={V_next:.4e}")
                continue

            diffs.append(float(V_next - V_curr))

        return diffs

    def alpha_and_max_violation(self, alpha_required: float = 1e-3) -> AlphaViolationStats:
        """
        This implementation estimates the *observed* alpha and maximum violation at each step as
        
        $$\alpha_{\text{obs}}(n) = min((V_N(x_k) - V_N(x_{k+1})) / \ell(x_k,u_k))$$  
        $$viol(n)      = max(0, V_N(x_{k+1}) - V_N(x_k) + \alpha_{\text{req}}*\ell(x_k,u_k))$$  
        
        where \ell(x_k,u_k) is the stage cost at step n.

        Parameters
        ----------
        alpha_required : float
            Minimum empirical threshold required for verification.

        Returns
        -------
        stats : AlphaViolationStats
            The observed minimum alpha and maximum violation statistics.
        """
        min_alpha: float = float("inf")
        max_violation = 0.0
        min_residual = float("inf")
        n_used = 0

        T_sim = min(self.cfg.T_sim, len(self.traj.states), len(self.traj.V_N))
        for n in range(T_sim - 1):
            V_curr = self._V(n)
            V_next = self._V(n + 1)
            if not (np.isfinite(V_curr) and np.isfinite(V_next)):
                __logger__.debug(f"Skipping step {n} due to non-finite value function V_N(x): V_curr={V_curr:.4e}, V_next={V_next:.4e}")
                continue

            l_curr = self.cfg.cost.get_stage_cost(self.traj.states[n], self.traj.inputs[n])

            if not np.isfinite(l_curr) or l_curr <= self.eps:
                # Still want to detect viaolations in decrease condition.
                __logger__.debug(f"Skipping step {n} due to non-finite or too small stage cost l_curr={l_curr:.4e}")
                eps = 1e-10 + 1e-8 * max(1.0, abs(V_curr))
                residual = float(V_next - V_curr)
                violation = max(0.0, residual - eps)
                max_violation = max(max_violation, violation)
                min_residual = min(min_residual, residual)
                n_used += 1
                continue

            alpha_obs = (V_curr - V_next) / l_curr
            if not np.isfinite(alpha_obs):
                __logger__.debug(f"Skipping step {n} due to non-finite observed alpha_obs={alpha_obs:.4e}")
                continue

            min_alpha = min(min_alpha, float(alpha_obs))
            rhs = V_curr - float(alpha_required) * l_curr
            residual = float(V_next - rhs)               # <0 means satisfied with margin
            violation = max(0.0, residual)               # >=0 by definition
            max_violation = max(max_violation, violation)
            min_residual = min(min_residual, residual)

            n_used += 1

        if n_used == 0:
            return AlphaViolationStats()

        if min_alpha == float("inf"):
            min_alpha = float("nan")

        return AlphaViolationStats(
            min_alpha=float(min_alpha), 
            max_violation=float(max_violation), 
            min_residual=float(min_residual), 
            n_used=int(n_used))

    def strict_asymptotic_stability(self, alpha_required: float = 1e-6) -> AsymptoticStabilityReport:
        """Empiricaly strict asymptotic stability verification for the dataset.
        Using the minimum observed alpha and maximum violation over all feasible trajectories.

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

        for _ in self.iter_binded_entries():
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
    def gamma_estimates(self) -> List[float]:
        """Estimate the maximum gamma value over the dataset."""
        gamma_values: List[float] = []

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

    def grüne_horizon_condition(self, min_cost_threshold: float = 1e-6) -> GrüneHorizonReport:
        """Dataset-level Grüne horizon condition certification.

        Parameters
        ----------
        min_cost_threshold : float
            Minimum stage cost l*(x0) to consider a data point valid for gamma estimation.

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
        gamma_values: List[float] = []
        
        for _ in self.iter_binded_entries():
            gamma_values.extend(
                self.gamma_estimates(min_cost_threshold=min_cost_threshold))
                

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
        solver: Optional[AcadosOcpSolver] = None,
        alpha_required: float = 1e-4,
        min_cost_threshold: float = 1e-3,
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
        min_cost_threshold : float
            Minimum stage cost l*(x0) to consider a data point valid for gamma estimation.

        Returns
        -------
        StabilityReport
            Stability report indicating whether the dataset passes empirical checks.
        """
        verifier = StabilityVerifier(dataset, solver)
        lyap_report = verifier.strict_asymptotic_stability(alpha_required=alpha_required, min_cost_threshold=min_cost_threshold)
        grune_report = verifier.grüne_horizon_condition(min_cost_threshold=min_cost_threshold)

        gruene_pass = bool(grune_report.applicability and grune_report.is_stable)
        lyap_pass = bool(lyap_report.is_stable)

        if lyap_pass:
            msg = (
                f"PASS. Lyapunov decrease observed with min_alpha={lyap_report.min_alpha:.3e} "
                f"and alpha_required={alpha_required:.3e}.")
        elif gruene_pass:
            msg = (
                f"PASS. Grüne horizon condition estimated with gamma={grune_report.gamma_estimate:.3e} "
                f"and required_horizon={grune_report.required_horizon}.")
        else:
            msg = (
                f"FAIL. lyapunov='{lyap_report.message}', grune='{grune_report.message}'.")

        return StabilityReport(
            method="Empirical Verification",
            is_stable=bool(gruene_pass or lyap_pass),
            details={
                "lyapunov_decrease_report": lyap_report,
                "grune_report": grune_report,
            },
            message=msg,
        )