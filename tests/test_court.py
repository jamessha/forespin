from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.court import CourtCalibrator, _sampled_frame_indices
from forespin.config import Thresholds
from forespin.domain import CourtCalibration, Point2D


class CourtCalibrationTests(unittest.TestCase):
    def test_sampled_frame_indices_cover_opening_window(self) -> None:
        self.assertEqual(_sampled_frame_indices(max_scan_frames=30, sample_count=5), [0, 7, 14, 22, 29])

    def test_calibrate_video_selects_highest_confidence_sample(self) -> None:
        class FakeCapture:
            def __init__(self) -> None:
                self.index = 0

            def isOpened(self) -> bool:
                return True

            def read(self):
                if self.index >= 10:
                    return False, None
                frame = {"index": self.index}
                self.index += 1
                return True, frame

            def release(self) -> None:
                pass

        class FakeBackend:
            def __init__(self) -> None:
                self.evaluated_frames: list[int] = []

            def evaluate_frame(self, frame):
                frame_index = frame["index"]
                self.evaluated_frames.append(frame_index)
                confidence = {0: 0.55, 4: 0.81, 9: 0.68}.get(frame_index, 0.0)
                calibration = CourtCalibration(
                    corners_px=[Point2D(0.0, 0.0), Point2D(1.0, 0.0), Point2D(1.0, 1.0), Point2D(0.0, 1.0)],
                    homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    confidence=confidence,
                    source=f"frame_{frame_index}",
                )
                return {
                    "calibration": calibration,
                    "summary": {"confidence": confidence, "reason": "accepted"},
                }

            def write_debug_artifacts(self, debug_dir, frame_index, frame, evaluation) -> None:
                pass

        cv2 = MagicMock()
        cv2.VideoCapture.return_value = FakeCapture()
        calibrator = CourtCalibrator(Thresholds(court_calibration_sample_frames=3))
        fake_backend = FakeBackend()
        calibrator.backend = fake_backend

        with patch("forespin.court.require_vision_stack", return_value=(cv2, object())):
            calibration = calibrator.calibrate_video("match.mp4", max_scan_frames=10)

        self.assertEqual(fake_backend.evaluated_frames, [0, 4, 9])
        self.assertEqual(calibration.source, "frame_4")

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
            Point2D(-6200.0, 100.0),
            Point2D(400.0, 100.0),
            Point2D(500.0, 140.0),
            Point2D(-6300.0, 140.0),
        ]

        self.assertFalse(
            CourtCalibrator(Thresholds())._corners_plausible_for_baseline_view(
                corners,
                width=1280,
                height=720,
            )
        )

    def test_relative_width_and_vertical_span_do_not_reject_unknown_tilt(self) -> None:
        corners = [
            Point2D(200.0, -1200.0),
            Point2D(220.0, -1190.0),
            Point2D(820.0, 2000.0),
            Point2D(-700.0, 2010.0),
        ]

        self.assertTrue(
            CourtCalibrator(Thresholds())._corners_plausible_for_baseline_view(
                corners,
                width=1280,
                height=720,
            )
        )

    def test_horizontal_flip_does_not_reject_reference_corner_order(self) -> None:
        corners = [
            Point2D(980.0, 120.0),
            Point2D(320.0, 120.0),
            Point2D(-260.0, 840.0),
            Point2D(1480.0, 840.0),
        ]

        self.assertTrue(
            CourtCalibrator(Thresholds())._corners_plausible_for_baseline_view(
                corners,
                width=1280,
                height=720,
            )
        )


if __name__ == "__main__":
    unittest.main()
