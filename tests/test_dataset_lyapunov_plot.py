import unittest
import numpy as np

from mpc_datagen import plots
from mpc_datagen.mpc_data import MPCConfig, MPCData, MPCDataset, MPCMeta, MPCTrajectory, LinearLSCost
from mpc_datagen.plots.lyapunov import create_lyapunov_from_dataset
from mpc_datagen.plots.utils import _extract_trajectory_v
try:
    from plot_assertions_mixin import PlotAssertionsMixin
except ImportError:
    from tests.plot_assertions_mixin import PlotAssertionsMixin


class TestDatasetLyapunovPlot(PlotAssertionsMixin):
    """Test suite for dataset-driven Lyapunov value function extraction, interpolation, and plotting."""

    def setUp(self) -> None:
        """Create a synthetic dataset with known value function V_N(x) = x_1^2 + 2 * x_2^2."""
        self.nx = 2
        self.nu = 1
        self.dt = 0.1
        self.T_sim = 20
        self.P = np.diag([1.0, 2.0])

        # Generate a small dataset with multiple rollouts converging to the origin
        self.dataset = self._create_synthetic_dataset()

    def _create_synthetic_dataset(self) -> MPCDataset:
        """Generate synthetic MPC data where V_N is computed along trajectories."""
        initial_states = [
            np.array([2.0, 1.5]),
            np.array([-2.0, 1.0]),
            np.array([1.5, -2.0]),
            np.array([-1.5, -1.5]),
            np.array([0.5, 0.5]),
            np.array([-0.5, -0.5]),
        ]

        entries: list[MPCData] = []
        for idx, x0 in enumerate(initial_states):
            states = np.zeros((self.T_sim + 1, self.nx), dtype=np.float64)
            inputs = np.zeros((self.T_sim, self.nu), dtype=np.float64)
            times = np.arange(self.T_sim + 1) * self.dt

            x_curr = x0.copy()
            states[0] = x_curr

            # Value function array V_N of length T_sim
            V_N = np.zeros(self.T_sim, dtype=np.float64)

            for t in range(self.T_sim):
                # V_N(x_t) = x_t^T P x_t
                V_N[t] = float(x_curr.T @ self.P @ x_curr)

                # Linear state feedback to simulate decay
                u_curr = np.array([-0.5 * x_curr[0] - 0.5 * x_curr[1]])
                inputs[t] = u_curr

                x_next = 0.85 * x_curr + 0.1 * u_curr
                states[t + 1] = x_next
                x_curr = x_next

            cost = LinearLSCost(
                Vx=np.eye(self.nx),
                Vu=np.zeros((self.nx, self.nu)),
                W=self.P,
                yref=np.zeros(self.nx),
            )
            config = MPCConfig(T_sim=self.T_sim, dt=self.dt, nx=self.nx, nu=self.nu, cost=cost)
            meta = MPCMeta(id=idx, steps_simulated=self.T_sim)
            traj = MPCTrajectory(
                states=states,
                inputs=inputs,
                times=times,
                V_solver=V_N.copy(),
                V_N=V_N.copy(),
            )
            entries.append(MPCData(trajectory=traj, meta=meta, config=config))

        return MPCDataset(data_buffer=entries)

    def test_extract_trajectory_v_direct(self) -> None:
        """Test direct extraction of V_N from trajectory."""
        entry = self.dataset[0]
        v_opt = _extract_trajectory_v(entry.trajectory, entry)
        self.assertIsNotNone(v_opt)
        self.assertEqual(len(v_opt), self.T_sim)
        self.assertAlmostEqual(v_opt[0], float(entry.trajectory.states[0].T @ self.P @ entry.trajectory.states[0]))

    def test_extract_trajectory_v_fallback_solver(self) -> None:
        """Test fallback to V_solver when V_N is None."""
        entry = self.dataset[0]
        entry.trajectory.V_N = None
        v_opt = _extract_trajectory_v(entry.trajectory, entry, use_solver_fallback=True)
        self.assertIsNotNone(v_opt)
        self.assertEqual(len(v_opt), self.T_sim)

    def test_create_lyapunov_from_dataset_evaluation(self) -> None:
        """Test creating a callable Lyapunov interpolator from dataset points."""
        lyap_fn = create_lyapunov_from_dataset(self.dataset, method="linear", use_solver_v=True)

        # Evaluate at an exact sampled initial point
        x_sample = self.dataset[0].trajectory.states[0]
        true_val = float(x_sample.T @ self.P @ x_sample)
        interpolated_val = lyap_fn(x_sample)

        # Should match exact value closely at sample points
        self.assertAlmostEqual(interpolated_val, true_val, places=4)

        # Test with use_solver_v=False (using V_N)
        lyap_fn_vn = create_lyapunov_from_dataset(self.dataset, method="linear", use_solver_v=False)
        interpolated_val_vn = lyap_fn_vn(x_sample)
        self.assertAlmostEqual(interpolated_val_vn, true_val, places=4)

        # Test batch evaluation inside convex hull
        x_batch = np.array([
            [2.0, 1.5],
            [-2.0, 1.0],
            [0.0, 0.0],
        ])
        vals = lyap_fn(x_batch)
        self.assertEqual(vals.shape, (3,))
        self.assertGreaterEqual(vals[0], 0.0)

    def test_create_lyapunov_from_dataset_extrapolate(self) -> None:
        """Test extrapolation behavior for create_lyapunov_from_dataset."""
        lyap_no_extrap = create_lyapunov_from_dataset(self.dataset, method="linear", extrapolate=False)
        lyap_with_extrap = create_lyapunov_from_dataset(self.dataset, method="linear", extrapolate=True)

        far_point = np.array([100.0, 100.0])
        val_no_extrap = lyap_no_extrap(far_point)
        val_with_extrap = lyap_with_extrap(far_point)

        self.assertTrue(np.isnan(val_no_extrap))
        self.assertTrue(np.isfinite(val_with_extrap))

    def test_interpolate_dataset_v_grid_preserves_nan(self) -> None:
        """Test that _interpolate_dataset_v_grid preserves NaNs outside convex hull by default."""
        from mpc_datagen.plots.lyapunov import _interpolate_dataset_v_grid

        x_vec = np.linspace(-10.0, 10.0, 21)
        y_vec = np.linspace(-10.0, 10.0, 21)
        X, Y = np.meshgrid(x_vec, y_vec)

        # By default fill_nearest=False -> NaNs exist outside [-2, 2]
        Z_no_fill = _interpolate_dataset_v_grid(self.dataset, 0, 1, X, Y, fill_nearest=False)
        self.assertIsNotNone(Z_no_fill)
        self.assertTrue(np.any(np.isnan(Z_no_fill)))

        # Center should be finite
        center_idx = 10  # corresponds to (0, 0)
        self.assertTrue(np.isfinite(Z_no_fill[center_idx, center_idx]))

        # With fill_nearest=True -> no NaNs
        Z_filled = _interpolate_dataset_v_grid(self.dataset, 0, 1, X, Y, fill_nearest=True)
        self.assertIsNotNone(Z_filled)
        self.assertFalse(np.any(np.isnan(Z_filled)))

    def test_create_lyapunov_from_dataset_empty_raises(self) -> None:
        """Test that attempting to create interpolator from empty dataset raises ValueError."""
        empty_dataset = MPCDataset()
        with self.assertRaises(ValueError):
            create_lyapunov_from_dataset(empty_dataset)

    def test_lyapunov_plot_with_dataset_vn_2d(self) -> None:
        """Test generating 2D Lyapunov landscape directly from dataset V_N (lyapunov_func=None)."""
        results = plots.lyapunov(
            lyapunov_func=None,
            dataset=self.dataset,
            state_labels=["Position $x_1$", "Velocity $x_2$"],
            plot_3d=False,
            html_path=None,
        )
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)

        res = results[0]
        self.assertEqual(res.idx_x, 0)
        self.assertEqual(res.idx_y, 1)
        self.assertIn("Position", res.label_x)
        self.assertIn("Velocity", res.label_y)
        # Should contain contour landscape and trajectory traces
        self.assertGreaterEqual(len(res.figure.data), 1)

    def test_lyapunov_plot_with_dataset_vn_3d(self) -> None:
        """Test generating 3D Lyapunov surface directly from dataset V_N."""
        results = plots.lyapunov(
            lyapunov_func=None,
            dataset=self.dataset,
            state_labels=["x_1", "x_2"],
            plot_3d=True,
            scatter_points=True,
            html_path=None,
        )
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)

        res = results[0]
        self.assertGreaterEqual(len(res.figure.data), 2)  # Surface + Trajectories + Scatter

    def test_lyapunov_plot_file_output(self) -> None:
        """Test saving dataset Lyapunov HTML plot using PlotAssertionsMixin."""
        self._assert_plot_written(
            plot_fn=plots.lyapunov,
            stem="test_dataset_lyapunov_3d",
            plot_kwargs={
                "lyapunov_func": None,
                "dataset": self.dataset,
                "state_labels": ["x_1", "x_2"],
                "plot_3d": True,
            },
        )

    def test_modular_submodule_imports(self) -> None:
        """Test importing submodules directly from mpc_datagen.plots."""
        from mpc_datagen.plots import utils, descent
        from mpc_datagen.plots.lyapunov import lyapunov, create_lyapunov_from_dataset
        from mpc_datagen.plots.trajectories import mpc_trajectories

        self.assertTrue(callable(utils._to_latex))
        self.assertTrue(callable(lyapunov))
        self.assertTrue(callable(create_lyapunov_from_dataset))
        self.assertTrue(callable(mpc_trajectories))
        self.assertTrue(callable(descent.cost_descent))
        self.assertTrue(callable(descent.relaxed_dp_residual))

    def test_cost_descent_and_relaxed_dp(self) -> None:
        """Test cost descent and relaxed DP residual plots."""
        fig_descent = plots.cost_descent(self.dataset, use_optimal_v=True)
        self.assertIsNotNone(fig_descent)

        fig_relaxed = plots.relaxed_dp_residual(self.dataset, alpha=0.9)
        self.assertIsNotNone(fig_relaxed)

    def test_trajectories_plot(self) -> None:
        """Test MPC state and control trajectory plotting."""
        fig_traj = plots.mpc_trajectories(
            dataset=self.dataset,
            state_labels=["x_1", "x_2"],
            control_labels=["u_1"],
        )
        self.assertIsNotNone(fig_traj)

    def test_lyapunov_dataset_solver_v(self) -> None:
        """Test lyapunov function directly using dataset V_solver vs V_N."""
        # 3D with use_solver_v=True
        res_3d = plots.lyapunov(
            dataset=self.dataset,
            plot_3d=True,
            use_solver_v=True,
            state_labels=["x_1", "x_2"],
        )
        self.assertIsNotNone(res_3d)
        self.assertEqual(len(res_3d), 1)
        self.assertGreaterEqual(len(res_3d[0].figure.data), 1)

        # 2D with use_solver_v=False
        res_2d = plots.lyapunov(
            dataset=self.dataset,
            plot_3d=False,
            use_solver_v=False,
            state_labels=["x_1", "x_2"],
        )
        self.assertIsNotNone(res_2d)
        self.assertEqual(len(res_2d), 1)


if __name__ == "__main__":
    unittest.main()
