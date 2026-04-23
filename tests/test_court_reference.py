from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.court_reference import CourtReference


class CourtReferenceTests(unittest.TestCase):
    def test_normalized_border_points_cover_unit_square(self) -> None:
        reference = CourtReference()
        border = reference.normalized_border_points()

        self.assertEqual((border[0].x, border[0].y), (0.0, 0.0))
        self.assertEqual((border[1].x, border[1].y), (1.0, 0.0))
        self.assertEqual((border[2].x, border[2].y), (1.0, 1.0))
        self.assertEqual((border[3].x, border[3].y), (0.0, 1.0))

    def test_configuration_indices_resolve_to_keypoint_indices(self) -> None:
        reference = CourtReference()
        configurations = reference.configuration_indices()

        self.assertEqual(len(configurations), 12)
        for configuration in configurations:
            self.assertEqual(len(configuration), 4)
            for index in configuration:
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, len(reference.key_points))


if __name__ == "__main__":
    unittest.main()
