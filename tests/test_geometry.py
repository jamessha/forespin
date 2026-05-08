from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.domain import Point2D
from forespin.geometry import point_in_court, point_in_singles_court


class GeometryTests(unittest.TestCase):
    def test_singles_in_bounds_excludes_doubles_alleys(self) -> None:
        self.assertTrue(point_in_court(Point2D(0.05, 0.5)))
        self.assertFalse(point_in_singles_court(Point2D(0.05, 0.5)))
        self.assertTrue(point_in_singles_court(Point2D(0.5, 0.5)))
        self.assertFalse(point_in_singles_court(Point2D(0.95, 0.5)))


if __name__ == "__main__":
    unittest.main()
