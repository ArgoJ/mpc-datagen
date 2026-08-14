import math
from rich.console import Console
from rich.table import Table

from ..utils.render import pretty_num, prettify_text
from .reports import AnalyticROAReport, EmpiricalROAReport


class AnalyticROARender(Table):
    """Rich table representation for analytic LQR ROA report."""

    def __init__(self, report: AnalyticROAReport, show_all_constraints: bool = False):
        super().__init__(title="Analytic Region of Attraction (ROA) Report")
        self.add_column("Property / Constraint", style="cyan", no_wrap=True)
        self.add_column("Value", style="magenta")
        self.add_column("Assessment / Details", style="green")

        # Level Set
        self.add_row(
            "Analytic Level Set (c_min)",
            pretty_num(report.c_min),
            "Maximal constraint-admissible ellipsoid level" if report.is_bounded else "Unbounded",
        )

        # Active Constraint
        if report.active_constraint:
            bound_str = f"bound = {pretty_num(report.active_bound_value)}" if not math.isnan(report.active_bound_value) else ""
            self.add_row(
                "Active Limiting Constraint",
                report.active_constraint,
                f"[bold yellow]Tightest bottleneck[/] ({bound_str})",
            )

        # Ellipsoid Volume
        if report.ellipsoid_volume is not None:
            self.add_row(
                "Invariant Ellipsoid Volume",
                pretty_num(report.ellipsoid_volume),
                "Analytical nD volume of {x : 0.5 x^T P x <= c_min}",
            )

        # Eigenvalues
        if report.eigenvalues_P:
            eigs_str = ", ".join([pretty_num(e) for e in report.eigenvalues_P])
            self.add_row(
                "P-Matrix Eigenvalues",
                eigs_str,
                f"min={pretty_num(min(report.eigenvalues_P))}, max={pretty_num(max(report.eigenvalues_P))}",
            )

        # Optional full constraint breakdown
        if show_all_constraints and report.constraint_limits:
            for cl in report.constraint_limits:
                active_flag = " [bold yellow]★ ACTIVE[/]" if cl.is_active else ""
                self.add_row(
                    f"  Constraint '{cl.name}'",
                    f"c_lim={pretty_num(cl.c_limit)}",
                    f"bound={pretty_num(cl.bound_value)}{active_flag}",
                )

        # Status
        status_style = "[bold green]BOUNDED[/]" if report.is_bounded else "[bold yellow]UNBOUNDED[/]"
        self.add_row("Status", prettify_text(report.message), status_style)

    def render(self) -> None:
        """Print the table to the console."""
        console = Console()
        console.print(self)


class EmpiricalROARender(Table):
    """Rich table representation for empirical ROA estimation report."""

    def __init__(self, report: EmpiricalROAReport):
        super().__init__(title="Empirical Region of Attraction (ROA) Report")
        self.add_column("Property / Metric", style="cyan", no_wrap=True)
        self.add_column("Value", style="magenta")
        self.add_column("Assessment", style="green")

        # Rollout Summary
        conv_style = "[green]OK[/]" if report.num_converged > 0 else "[red]NONE[/]"
        self.add_row(
            "Converged Rollouts",
            f"{report.num_converged}/{report.total_trajectories} ({report.convergence_rate:.1f}%)",
            conv_style,
        )
        self.add_row(
            "Feasible Rollouts",
            f"{report.num_feasible}/{report.total_trajectories} ({report.feasibility_rate:.1f}%)",
            "[green]OK[/]" if report.num_feasible > 0 else "[red]NONE[/]",
        )
        self.add_row(
            "Failed / Diverged Rollouts",
            f"{report.num_failed}/{report.total_trajectories}",
            "[green]OK[/]" if report.num_failed == 0 else "[yellow]UNSTABLE/INFEASIBLE[/]",
        )

        # Empirical Level Set
        if report.c_empirical is not None:
            self.add_row(
                "Empirical Sublevel Set (c_empirical)",
                pretty_num(report.c_empirical),
                "Maximal guaranteed convergent cost level set",
            )

        # Convex Hull
        if report.convex_hull_volume is not None:
            self.add_row(
                "Empirical Convex Hull Volume",
                pretty_num(report.convex_hull_volume),
                "Convex hull of all converged initial states x0",
            )

        # Spatial Bounds
        if report.state_bounds_empirical:
            bounds_str = ", ".join(
                [f"x[{k}]: [{pretty_num(lb)}, {pretty_num(ub)}]" for k, (lb, ub) in report.state_bounds_empirical.items()]
            )
            self.add_row("Empirical State Bounds", bounds_str, "Min/Max across converged rollouts")

        # Status
        self.add_row("Status", report.message, "[bold green]PASS[/]" if report.is_valid else "[bold red]FAIL[/]")

    def render(self) -> None:
        """Print the table to the console."""
        console = Console()
        console.print(self)

