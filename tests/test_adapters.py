import sys
import unittest
from pathlib import Path

import numpy as np


# Ensure we can import from ./src without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from mpc_datagen.adapters.acados import (
    extract_cfg, 
    extract_discretized_dynamics
)

from acados_solver_example import get_basic_double_integrator_ocp_solver


class TestAdapters(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.solver, self.system_info = get_basic_double_integrator_ocp_solver()
        
    def test_passes_linear_system_extraction(self) -> None:
        ocp = self.solver.acados_ocp
        nx = ocp.dims.nx
        nu = ocp.dims.nu
        dt = float(ocp.solver_options.tf) / int(ocp.solver_options.N_horizon)
        extracted = extract_discretized_dynamics(ocp, np.zeros(nx), np.zeros(nu), dt)

        np.testing.assert_allclose(extracted.A, self.system_info["A_d"])
        np.testing.assert_allclose(extracted.B, self.system_info["B_d"])
        
    def test_passes_cfg_extraction(self) -> None:
        extracted = extract_cfg(self.solver)
        
        N = self.solver.acados_ocp.solver_options.N_horizon
        dt = float(self.solver.acados_ocp.solver_options.tf) / int(N)

        self.assertEqual(extracted.N, N)
        self.assertEqual(extracted.dt, dt)
        self.assertEqual(extracted.T_sim, 0)
        self.assertEqual(extracted.nx, self.solver.acados_ocp.dims.nx)
        self.assertEqual(extracted.nu, self.solver.acados_ocp.dims.nu)
        
        # Check x0 only if it's not None
        if self.solver.acados_ocp.constraints.x0 is not None:
            np.testing.assert_allclose(extracted.constraints.x0, self.solver.acados_ocp.constraints.x0)
        else:
            self.assertEqual(extracted.constraints.x0.size, 0)

    def test_euler_discretized_dynamics_extraction(self) -> None:
        """Test extraction of discrete-time dynamics using 1-step explicit Euler."""
        from mpc_datagen.utils.linalg import lin_c2d_euler

        ocp = self.solver.acados_ocp
        ocp.solver_options.sim_method_num_stages = 1
        ocp.solver_options.sim_method_num_steps = 1

        nx = ocp.dims.nx
        nu = ocp.dims.nu
        dt = float(ocp.solver_options.tf) / int(ocp.solver_options.N_horizon)
        extracted = extract_discretized_dynamics(ocp, np.zeros(nx), np.zeros(nu), dt)

        A_c = np.array([[0.0, 1.0], [0.0, 0.0]])
        B_c = np.array([[0.0], [1.0]])
        expected_A, expected_B = lin_c2d_euler(A_c, B_c, dt, num_steps=1)

        # For 1-step Euler on double integrator: A_d = I + dt*A_c, B_d = dt*B_c
        np.testing.assert_allclose(extracted.A, expected_A)
        np.testing.assert_allclose(extracted.B, expected_B)
        np.testing.assert_allclose(extracted.A, np.eye(2) + dt * A_c)
        np.testing.assert_allclose(extracted.B, dt * B_c)

        # Also check extract_cfg works seamlessly with 1-stage Euler
        cfg = extract_cfg(self.solver)
        np.testing.assert_allclose(cfg.model.A, expected_A)
        np.testing.assert_allclose(cfg.model.B, expected_B)

        # Reset solver options for other tests
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1

    def test_euler_multi_step_discretization(self) -> None:
        """Test multi-step Euler discretization."""
        from mpc_datagen.utils.linalg import lin_c2d_euler

        ocp = self.solver.acados_ocp
        ocp.solver_options.sim_method_num_stages = 1
        ocp.solver_options.sim_method_num_steps = 3

        nx = ocp.dims.nx
        nu = ocp.dims.nu
        dt = float(ocp.solver_options.tf) / int(ocp.solver_options.N_horizon)
        extracted = extract_discretized_dynamics(ocp, np.zeros(nx), np.zeros(nu), dt)

        A_c = np.array([[0.0, 1.0], [0.0, 0.0]])
        B_c = np.array([[0.0], [1.0]])
        expected_A, expected_B = lin_c2d_euler(A_c, B_c, dt, num_steps=3)

        np.testing.assert_allclose(extracted.A, expected_A)
        np.testing.assert_allclose(extracted.B, expected_B)

        # Reset solver options
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1

    def test_unsupported_stages_raises(self) -> None:
        """Test that unsupported number of stages raises NotImplementedError."""
        ocp = self.solver.acados_ocp
        ocp.solver_options.sim_method_num_stages = 3

        nx = ocp.dims.nx
        nu = ocp.dims.nu
        dt = float(ocp.solver_options.tf) / int(ocp.solver_options.N_horizon)

        with self.assertRaises(NotImplementedError):
            extract_discretized_dynamics(ocp, np.zeros(nx), np.zeros(nu), dt)

        # Reset solver options
        ocp.solver_options.sim_method_num_stages = 4


if __name__ == "__main__":
    unittest.main()
