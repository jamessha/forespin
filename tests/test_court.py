from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.court import CourtCalibrator
from forespin.config import Thresholds
from forespin.domain import Point2D


class CourtCalibrationTests(unittest.TestCase):
    def test_clipped_near_corners_are_still_plausible(self) -> None:
        corners = [
            Point2D(320.0, 120.0),
            Point2D(980.0, 120.0),
            Point2D(1480.0, 840.0),
            Point2D(-260.0, 840.0),
        ]

        self.assertTrue(
            CourtCalibrator(Thresholds())._corners_plausible_for_baseline_view(
                corners,
                width=1280,
                height=720,
            )
        )

    def test_implausible_corners_are_rejected(self) -> None:
        corners = [
            Point2D(-3200.0, 100.0),
            Point2D(400.0, 100.0),
            Point2D(500.0, 140.0),
            Point2D(-3300.0, 140.0),
        ]

        self.assertFalse(
            CourtCalibrator(Thresholds())._corners_plausible_for_baseline_view(
                corners,
                width=1280,
                height=720,
            )
        )


if __name__ == "__main__":
    unittest.main()
