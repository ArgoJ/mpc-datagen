import unittest

import numpy as np
import torch as th
from scipy.linalg import solve_continuous_are

from mpc_datagen import plots
from mpc_datagen.mpc_data import MPCConfig, MPCData, MPCDataset, MPCMeta, MPCTrajectory
from plot_assertions_mixin import PlotAssertionsMixin
from shared_utils import _LQRRiccatiLyapunov, analytical_quadratic_level_set_measure


class TestLQRLyapunovPlot(PlotAssertionsMixin):
    """Test suite for plotting LQR Riccati Lyapunov landscapes and trajectories."""

    def setUp(self) -> None:
        """Set up continuous-time linear system and compute Riccati matrix P."""
        # 2D Linear system (oscillator/double integrator variant)
        self.A = np.array([[0.0, 1.0], [-2.0, -1.0]], dtype=np.float64)
        self.B = np.array([[0.0], [1.0]], dtype=np.float64)
        self.Q = np.diag([1.0, 1.0])
        self.R = np.array([[1.0]], dtype=np.float64)

        # Solve Continuous Algebraic Riccati Equation: A^T P + P A - P B R^-1 B^T P + Q = 0
        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)

        # Closed-loop gain: K = R^-1 B^T P
        self.K = np.linalg.solve(self.R, self.B.T @ self.P)
        self.A_cl = self.A - self.B @ self.K

        # Wrap in PyTorch Lyapunov module from shared_utils
        self.riccati_module = _LQRRiccatiLyapunov(self.P)

        # Create callable for plots.lyapunov
        def lyapunov_func(x: np.ndarray) -> np.ndarray:
            x_tensor = th.as_tensor(x, dtype=th.float32)
            if x_tensor.ndim == 1:
                x_tensor = x_tensor.unsqueeze(0)
            return self.riccati_module(x_tensor).squeeze(-1).detach().numpy()

        self.lyapunov_func = lyapunov_func

        # Build a small dataset of closed-loop LQR rollouts
        self.dt = 0.05
        self.T_sim = 40
        self.dataset = self._generate_lqr_dataset()

    def _generate_lqr_dataset(self) -> MPCDataset:
        """Simulate closed-loop LQR rollouts using Euler integration."""
        initial_states = [
            np.array([2.0, 1.0]),
            np.array([-1.5, 2.0]),
            np.array([-2.0, -1.0]),
            np.array([1.0, -2.0]),
        ]
        entries: list[MPCData] = []

        for idx, x0 in enumerate(initial_states):
            states = np.zeros((self.T_sim + 1, 2), dtype=np.float64)
            inputs = np.zeros((self.T_sim, 1), dtype=np.float64)
            times = np.arange(self.T_sim + 1) * self.dt

            x_curr = x0.copy()
            states[0] = x_curr

            for t in range(self.T_sim):
                u_curr = -self.K @ x_curr
                inputs[t] = u_curr
                # Closed-loop update: x_next = x_curr + dt * (A_cl * x_curr)
                x_next = x_curr + self.dt * (self.A_cl @ x_curr)
                states[t + 1] = x_next
                x_curr = x_next

            traj = MPCTrajectory(
                states=states,
                inputs=inputs,
                times=times,
                V_solver=np.zeros(self.T_sim),
            )
            config = MPCConfig(T_sim=self.T_sim, dt=self.dt, nx=2, nu=1)
            meta = MPCMeta(id=idx, steps_simulated=self.T_sim)
            entries.append(MPCData(trajectory=traj, meta=meta, config=config))

        return MPCDataset(data_buffer=entries)

    def test_lqr_lyapunov_2d_plot(self) -> None:
        """Test generating and saving 2D Riccati Lyapunov plot with ROA level set."""
        roa_level = 5.0
        # Calculate sublevel set area using shared_utils
        volume = analytical_quadratic_level_set_measure(roa_level, self.P)
        self.assertGreater(volume, 0.0)

        self._assert_plot_written(
            plot_fn=plots.lyapunov,
            stem="test_lqr_riccati_lyapunov_2d",
            plot_kwargs={
                "lyapunov_func": self.lyapunov_func,
                "dataset": self.dataset,
                "roa_level": roa_level,
                "state_labels": ["x_1", "x_2"],
                "limits": [(-3.0, 3.0), (-3.0, 3.0)],
                "plot_3d": False,
            },
        )

    def test_lqr_lyapunov_3d_plot(self) -> None:
        """Test generating and saving 3D Riccati Lyapunov plot with ROA level set."""
        self._assert_plot_written(
            plot_fn=plots.lyapunov,
            stem="test_lqr_riccati_lyapunov_3d",
            plot_kwargs={
                "lyapunov_func": self.lyapunov_func,
                "dataset": self.dataset,
                "roa_level": 5.0,
                "state_labels": ["x_1", "x_2"],
                "limits": [(-3.0, 3.0), (-3.0, 3.0)],
                "plot_3d": True,
            },
        )

    def test_lqr_lyapunov_return_figures(self) -> None:
        """Test direct figure return when html_path is None."""
        results = plots.lyapunov(
            lyapunov_func=self.lyapunov_func,
            dataset=self.dataset,
            roa_level=4.0,
            state_labels=["Position x_1", "Velocity x_2"],
            plot_3d=False,
            html_path=None,
        )
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)

        result = results[0]
        self.assertEqual(result.idx_x, 0)
        self.assertEqual(result.idx_y, 1)
        self.assertIn("x_1", result.label_x)
        self.assertIn("x_2", result.label_y)
        self.assertGreater(len(result.figure.data), 0)


if __name__ == "__main__":
    unittest.main()
