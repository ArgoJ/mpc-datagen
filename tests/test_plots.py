import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from mpc_datagen.plots import _extract_roa_boundary


class TestExtractRoaBoundary(unittest.TestCase):
    def test_clipped_roa_follows_box_edge(self) -> None:
        x_vec = np.linspace(-1.0, 1.0, 81)
        y_vec = np.linspace(-1.0, 1.0, 81)
        X, Y = np.meshgrid(x_vec, y_vec)
        Z = X**2 + Y**2

        x_points, y_points = _extract_roa_boundary(x_vec, y_vec, Z, c_level=1.2)

        self.assertGreater(x_points.size, 0)
        self.assertTrue(np.isclose(np.min(x_points), x_vec[0]))
        self.assertTrue(np.isclose(np.max(x_points), x_vec[-1]))
        self.assertTrue(np.isclose(np.min(y_points), y_vec[0]))
        self.assertTrue(np.isclose(np.max(y_points), y_vec[-1]))

    def test_full_box_is_returned_when_roa_covers_domain(self) -> None:
        x_vec = np.linspace(-1.0, 1.0, 41)
        y_vec = np.linspace(-1.0, 1.0, 41)
        X, Y = np.meshgrid(x_vec, y_vec)
        Z = X**2 + Y**2

        x_points, y_points = _extract_roa_boundary(x_vec, y_vec, Z, c_level=3.0)

        self.assertGreater(x_points.size, 0)
        self.assertTrue(np.isclose(np.min(x_points), x_vec[0]))
        self.assertTrue(np.isclose(np.max(x_points), x_vec[-1]))
        self.assertTrue(np.isclose(np.min(y_points), y_vec[0]))
        self.assertTrue(np.isclose(np.max(y_points), y_vec[-1]))


if __name__ == "__main__":
    unittest.main()