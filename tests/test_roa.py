import unittest
import numpy as np

from mpc_datagen.mpc_data import (
    MPCConfig,
    MPCData,
    MPCDataset,
    MPCMeta,
    MPCTrajectory,
    LinearLSCost,
    LinearSystem,
    Constraints,
)
from mpc_datagen.roa import (
    ROAVerifier,
    AnalyticROAVerifier,
    AnalyticROAReport,
    AnalyticROARender,
    EmpiricalROAEstimator,
    EmpiricalROAReport,
    EmpiricalROARender,
    TrajectoryStatus,
    pretty_num,
)


class TestROAModule(unittest.TestCase):
    """Unit test suite for mpc_datagen.roa module (Analytic & Empirical)."""

    def setUp(self) -> None:
        self.nx = 2
        self.nu = 1
        self.dt = 0.1
        self.T_sim = 20

        # Discrete-time double integrator dynamics
        A = np.array([[1.0, 0.1], [0.0, 1.0]], dtype=np.float64)
        B = np.array([[0.005], [0.1]], dtype=np.float64)
        model = LinearSystem(A=A, B=B)

        # Standard quadratic cost
        W = np.diag([1.0, 1.0, 0.1])
        Vx = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        Vu = np.array([[0.0], [0.0], [1.0]])
        yref = np.zeros(3)
        cost = LinearLSCost(W=W, Vx=Vx, Vu=Vu, yref=yref)

        # Constraints
        lbx = np.array([-2.0, -2.0])
        ubx = np.array([2.0, 2.0])
        lbu = np.array([-1.0])
        ubu = np.array([1.0])
        constraints = Constraints(lbx=lbx, ubx=ubx, lbu=lbu, ubu=ubu)

        self.cfg = MPCConfig(
            nx=self.nx,
            nu=self.nu,
            N=10,
            dt=self.dt,
            T_sim=self.T_sim,
            model=model,
            cost=cost,
            constraints=constraints,
        )

    def _create_mock_trajectory(
        self,
        x0: np.ndarray,
        converges: bool = True,
        feasible: bool = True,
        violate_constraints: bool = False,
    ) -> MPCData:
        """Helper to create a single MPCData rollout."""
        states = np.zeros((self.T_sim + 1, self.nx), dtype=np.float64)
        inputs = np.zeros((self.T_sim, self.nu), dtype=np.float64)
        times = np.arange(self.T_sim + 1) * self.dt

        states[0] = x0
        x_curr = x0.copy()

        for t in range(self.T_sim):
            if not converges:
                # Stays at constant offset (does not converge to origin, but within bounds)
                x_next = 0.99 * x_curr
            else:
                # Stable decay to origin
                x_next = 0.7 * x_curr

            if violate_constraints and t == 5:
                x_next = np.array([10.0, 10.0])

            inputs[t] = np.array([0.0])
            states[t + 1] = x_next
            x_curr = x_next

        # Quadratic value function sequence V_N
        P = np.diag([1.0, 2.0])
        V_solver = np.array([0.5 * s.T @ P @ s for s in states[:-1]])
        V_N = V_solver.copy()

        traj = MPCTrajectory(
            states=states,
            inputs=inputs,
            times=times,
            V_solver=V_solver,
            V_N=V_N,
        )

        status_codes = [0] * self.T_sim if feasible else [1] * self.T_sim
        meta = MPCMeta(
            id=0,
            feasible=feasible,
            status_codes=status_codes,
            steps_simulated=self.T_sim,
            solve_time_mean=0.005,
            solve_time_total=0.1,
        )

        return MPCData(trajectory=traj, meta=meta, config=self.cfg)

    def test_analytic_roa_verifier(self) -> None:
        """Test analytic LQR ROA calculation, report generation, and rendering."""
        verifier = ROAVerifier(self.cfg)
        c_min = verifier.compute_min_c()
        self.assertGreater(c_min, 0.0)
        self.assertTrue(np.isfinite(c_min))

        bounds, c_val = verifier.roa_bounds(n_points=50)
        self.assertEqual(bounds.shape, (50, self.nx))
        self.assertEqual(c_val, c_min)

        # Test report generation
        report = verifier.compute_report()
        self.assertIsInstance(report, AnalyticROAReport)
        self.assertTrue(report.is_valid)
        self.assertTrue(report.is_bounded)
        self.assertAlmostEqual(report.c_min, c_min)
        self.assertIsNotNone(report.ellipsoid_volume)
        self.assertGreater(report.ellipsoid_volume, 0.0)
        self.assertTrue(len(report.constraint_limits) > 0)
        self.assertTrue(any(cl.is_active for cl in report.constraint_limits))

        # Test verify() alias
        verify_report = verifier.verify()
        self.assertEqual(verify_report.c_min, report.c_min)

        # Test rendering
        render_table = AnalyticROARender(report, show_all_constraints=True)
        self.assertIsNotNone(render_table)
        self.assertEqual(render_table.title, "Analytic Region of Attraction (ROA) Report")

        # Check alias
        self.assertIs(ROAVerifier, AnalyticROAVerifier)

    def test_classify_trajectory_states(self) -> None:
        """Test trajectory classification under different conditions."""
        entries = [
            self._create_mock_trajectory(np.array([0.1, 0.1]), converges=True, feasible=True),
            self._create_mock_trajectory(np.array([0.5, 0.5]), converges=False, feasible=True),
            self._create_mock_trajectory(np.array([0.2, 0.2]), converges=True, feasible=False),
            self._create_mock_trajectory(np.array([0.3, 0.3]), converges=True, feasible=True, violate_constraints=True),
        ]
        dataset = MPCDataset(data_buffer=entries)

        estimator = EmpiricalROAEstimator(dataset=dataset)

        p0 = estimator.classify_trajectory(dataset[0], index=0)
        self.assertTrue(p0.is_feasible)
        self.assertTrue(p0.is_converged)
        self.assertEqual(p0.status, TrajectoryStatus.FEASIBLE_CONVERGED)

        p1 = estimator.classify_trajectory(dataset[1], index=1)
        self.assertTrue(p1.is_feasible)
        self.assertFalse(p1.is_converged)
        self.assertEqual(p1.status, TrajectoryStatus.FEASIBLE_UNCONVERGED)

        p2 = estimator.classify_trajectory(dataset[2], index=2)
        self.assertFalse(p2.is_feasible)
        self.assertFalse(p2.is_converged)
        self.assertEqual(p2.status, TrajectoryStatus.INFEASIBLE)

        p3 = estimator.classify_trajectory(dataset[3], index=3)
        self.assertFalse(p3.is_converged)
        self.assertEqual(p3.status, TrajectoryStatus.CONSTRAINT_VIOLATED)

    def test_estimate_full_dataset(self) -> None:
        """Test complete dataset estimation and report generation."""
        initial_states = [
            np.array([0.05, 0.05]),
            np.array([-0.05, 0.05]),
            np.array([0.05, -0.05]),
            np.array([-0.05, -0.05]),
            np.array([0.2, 0.2]),
            np.array([-0.2, 0.2]),
        ]

        entries = [self._create_mock_trajectory(x0, converges=True, feasible=True) for x0 in initial_states]
        dataset = MPCDataset(data_buffer=entries)
        estimator = EmpiricalROAEstimator(dataset=dataset)

        report = estimator.estimate(show_progress=False)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_trajectories, len(initial_states))
        self.assertEqual(report.num_converged, len(initial_states))
        self.assertEqual(report.num_feasible, len(initial_states))
        self.assertIsNotNone(report.c_empirical)
        self.assertGreater(report.c_empirical, 0.0)
        self.assertIsNotNone(report.convex_hull_volume)
        self.assertGreater(report.convex_hull_volume, 0.0)

        # Check getter helpers
        succ_states = estimator.get_successful_initial_states()
        self.assertEqual(len(succ_states), len(initial_states))
        fail_states = estimator.get_failed_initial_states()
        self.assertEqual(len(fail_states), 0)

    def test_dataset_type_enforcement(self) -> None:
        """Test that passing a raw list or non-MPCDataset raises a TypeError."""
        entries = [self._create_mock_trajectory(np.array([0.05, 0.05]))]
        with self.assertRaises(TypeError):
            EmpiricalROAEstimator(dataset=entries)  # type: ignore

    def test_render_and_pretty_num(self) -> None:
        """Test formatting and rich table rendering."""
        self.assertEqual(pretty_num(1.0), "1")
        self.assertEqual(pretty_num(0.0001234), "1.234e-4")
        self.assertEqual(pretty_num(float("nan")), "nan")

        entries = [self._create_mock_trajectory(np.array([0.05, 0.05]), converges=True)]
        dataset = MPCDataset(data_buffer=entries)
        estimator = EmpiricalROAEstimator(dataset=dataset)
        report = estimator.estimate(show_progress=False)

        render_table = EmpiricalROARender(report)
        self.assertIsNotNone(render_table)
        self.assertEqual(render_table.title, "Empirical Region of Attraction (ROA) Report")


if __name__ == "__main__":
    unittest.main()
