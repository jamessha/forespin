from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.domain import Point2D
from forespin.overlay import _clip_point_to_frame


class OverlayTests(unittest.TestCase):
    def test_clip_point_to_frame_allows_offscreen_court_corners(self) -> None:
        clipped = _clip_point_to_frame(Point2D(-20.0, 900.0), width=1280, height=720)

        self.assertIsNotNone(clipped)
        assert clipped is not None
        self.assertEqual(clipped.x, 0.0)
        self.assertEqual(clipped.y, 719.0)


if __name__ == "__main__":
    unittest.main()
