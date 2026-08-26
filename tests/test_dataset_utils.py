import unittest
import numpy as np

from mpc_datagen.mpc_data import (
    MPCConfig,
    MPCData,
    MPCDataset,
    MPCMeta,
    MPCTrajectory,
    LinearLSCost,
)
from mpc_datagen.utils.dataset import (
    MatchedStatePair,
    get_initial_states,
    find_matching_initial_states,
    compute_entry_error,
    create_error_dataset,
)


class TestDatasetUtils(unittest.TestCase):
    """Test suite for dataset comparison, initial state matching, and error dataset creation."""

    def setUp(self) -> None:
        self.nx = 2
        self.nu = 1
        self.T_sim = 10
        self.dt = 0.1

        # Common cost and config
        self.cost = LinearLSCost(
            Vx=np.eye(self.nx),
            Vu=np.zeros((self.nx, self.nu)),
            W=np.eye(self.nx),
            yref=np.zeros(self.nx),
        )
        self.config = MPCConfig(
            T_sim=self.T_sim,
            dt=self.dt,
            nx=self.nx,
            nu=self.nu,
            cost=self.cost,
        )

    def _create_entry(
        self,
        x0: np.ndarray,
        entry_id: int,
        decay_factor: float = 0.9,
        input_gain: float = -0.5,
    ) -> MPCData:
        """Helper to create a synthetic MPCData rollout from initial state x0."""
        states = np.zeros((self.T_sim + 1, self.nx), dtype=np.float64)
        inputs = np.zeros((self.T_sim, self.nu), dtype=np.float64)
        times = np.arange(self.T_sim + 1) * self.dt
        V_solver = np.zeros(self.T_sim, dtype=np.float64)

        x_curr = np.asarray(x0, dtype=np.float64).copy()
        states[0] = x_curr

        for t in range(self.T_sim):
            u_curr = np.array([input_gain * (x_curr[0] + x_curr[1])])
            inputs[t] = u_curr
            V_solver[t] = float(x_curr.T @ np.eye(self.nx) @ x_curr)
            x_next = decay_factor * x_curr + 0.1 * u_curr[0]
            states[t + 1] = x_next
            x_curr = x_next

        meta = MPCMeta(id=entry_id, steps_simulated=self.T_sim, feasible=True)
        traj = MPCTrajectory(
            states=states,
            inputs=inputs,
            times=times,
            V_solver=V_solver,
            V_N=V_solver.copy(),
        )
        return MPCData(trajectory=traj, meta=meta, config=self.config)

    def test_get_initial_states(self) -> None:
        """Test extracting initial states and metadata IDs."""
        x0_list = [np.array([1.0, 2.0]), np.array([-3.0, 4.5]), np.array([0.0, 0.0])]
        entries = [self._create_entry(x0, entry_id=100 + i) for i, x0 in enumerate(x0_list)]
        dataset = MPCDataset(data_buffer=entries)

        extracted = get_initial_states(dataset)
        self.assertEqual(len(extracted), 3)

        for i, (idx, meta_id, x0) in enumerate(extracted):
            self.assertEqual(idx, i)
            self.assertEqual(meta_id, 100 + i)
            np.testing.assert_allclose(x0, x0_list[i])

    def test_find_matching_initial_states(self) -> None:
        """Test finding matching initial states between two datasets with permuted orders."""
        # Dataset A has points: P0, P1, P2
        p0 = np.array([1.0, 2.0])
        p1 = np.array([-1.0, 0.5])
        p2 = np.array([3.0, -2.0])
        ds_a = MPCDataset(data_buffer=[
            self._create_entry(p0, entry_id=0),
            self._create_entry(p1, entry_id=1),
            self._create_entry(p2, entry_id=2),
        ])

        # Dataset B has points: P1, P3 (unmatched), P0
        p3 = np.array([9.9, 9.9])
        ds_b = MPCDataset(data_buffer=[
            self._create_entry(p1, entry_id=10),
            self._create_entry(p3, entry_id=20),
            self._create_entry(p0, entry_id=30),
        ])

        matches = find_matching_initial_states(ds_a, ds_b, atol=1e-6)
        self.assertEqual(len(matches), 2)

        # Match 1: P0 (ds_a idx 0, id 0) <-> (ds_b idx 2, id 30)
        match_p0 = next(m for m in matches if np.allclose(m.x0_a, p0))
        self.assertEqual(match_p0.idx_a, 0)
        self.assertEqual(match_p0.id_a, 0)
        self.assertEqual(match_p0.idx_b, 2)
        self.assertEqual(match_p0.id_b, 30)
        self.assertAlmostEqual(match_p0.diff_norm, 0.0)

        # Match 2: P1 (ds_a idx 1, id 1) <-> (ds_b idx 0, id 10)
        match_p1 = next(m for m in matches if np.allclose(m.x0_a, p1))
        self.assertEqual(match_p1.idx_a, 1)
        self.assertEqual(match_p1.id_a, 1)
        self.assertEqual(match_p1.idx_b, 0)
        self.assertEqual(match_p1.id_b, 10)
        self.assertAlmostEqual(match_p1.diff_norm, 0.0)

    def test_compute_entry_error(self) -> None:
        """Test compute_entry_error differences."""
        x0 = np.array([2.0, -1.0])
        entry_a = self._create_entry(x0, entry_id=1, decay_factor=0.9, input_gain=-0.5)
        entry_b = self._create_entry(x0, entry_id=2, decay_factor=0.8, input_gain=-0.4)

        err_entry = compute_entry_error(entry_a, entry_b, relative=False, error_id=42)

        # At t=0, initial states are equal so error is 0
        np.testing.assert_allclose(err_entry.trajectory.states[0], np.zeros(self.nx), atol=1e-12)

        # At t > 0, difference should equal entry_a - entry_b
        expected_state_diff = entry_a.trajectory.states - entry_b.trajectory.states
        expected_input_diff = entry_a.trajectory.inputs - entry_b.trajectory.inputs
        np.testing.assert_allclose(err_entry.trajectory.states, expected_state_diff)
        np.testing.assert_allclose(err_entry.trajectory.inputs, expected_input_diff)
        self.assertEqual(err_entry.meta.id, 42)
        self.assertTrue(err_entry.meta.feasible)

    def test_create_error_dataset(self) -> None:
        """Test create_error_dataset end-to-end matching and dataset creation."""
        x0_common = [
            np.array([1.0, 1.0]),
            np.array([-2.0, 0.0]),
            np.array([0.5, -1.5]),
        ]

        # Dataset A (e.g. NN controller)
        ds_a = MPCDataset(data_buffer=[
            self._create_entry(x0, entry_id=i, decay_factor=0.85)
            for i, x0 in enumerate(x0_common)
        ])

        # Dataset B (e.g. MPC expert with reversed order)
        ds_b = MPCDataset(data_buffer=[
            self._create_entry(x0, entry_id=100 + i, decay_factor=0.95)
            for i, x0 in enumerate(reversed(x0_common))
        ])

        error_ds = create_error_dataset(ds_a, ds_b, atol=1e-6)
        self.assertEqual(len(error_ds), 3)

        for i in range(len(error_ds)):
            err_entry = error_ds[i]
            # Initial state error must be 0
            np.testing.assert_allclose(err_entry.trajectory.states[0], np.zeros(self.nx), atol=1e-12)
            # Shapes must match expected simulation length
            self.assertEqual(err_entry.trajectory.states.shape, (self.T_sim + 1, self.nx))
            self.assertEqual(err_entry.trajectory.inputs.shape, (self.T_sim, self.nu))

    def test_create_error_dataset_with_plot_integration(self) -> None:
        """Test that the created error dataset works with trajectory_error_bands plot."""
        from mpc_datagen.plots.trajectories import trajectory_error_bands

        x0_common = [np.array([1.0, 1.0]), np.array([-2.0, 0.0])]
        ds_a = MPCDataset(data_buffer=[
            self._create_entry(x0, entry_id=i, decay_factor=0.85)
            for i, x0 in enumerate(x0_common)
        ])
        ds_b = MPCDataset(data_buffer=[
            self._create_entry(x0, entry_id=10 + i, decay_factor=0.95)
            for i, x0 in enumerate(x0_common)
        ])

        error_ds = create_error_dataset(ds_a, ds_b)
        fig = trajectory_error_bands(
            errors_dataset=error_ds,
            state_labels=["e_{x1}", "e_{x2}"],
            control_labels=["e_u"],
            plot_controls=True,
        )
        self.assertIsNotNone(fig)

    def test_no_matching_initial_states(self) -> None:
        """Test create_error_dataset when no states match returns empty dataset."""
        ds_a = MPCDataset(data_buffer=[self._create_entry(np.array([1.0, 2.0]), entry_id=0)])
        ds_b = MPCDataset(data_buffer=[self._create_entry(np.array([5.0, 6.0]), entry_id=1)])

        error_ds = create_error_dataset(ds_a, ds_b, atol=1e-6)
        self.assertEqual(len(error_ds), 0)


if __name__ == "__main__":
    unittest.main()

