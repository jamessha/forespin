from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.config import Thresholds
from forespin.domain import FrameObservation, Point2D
from forespin.tracking.observation_builder import (
    _clean_ball_track,
    _compute_ball_velocity,
    _interpolate_ball_track,
    _smooth_ball_track,
)


def _observation(
    frame_index: int,
    x: float | None,
    y: float | None,
    *,
    confidence: float = 0.9,
    court_x: float | None = None,
    court_y: float | None = None,
) -> FrameObservation:
    ball_px = Point2D(x, y) if x is not None and y is not None else None
    ball_court = Point2D(court_x, court_y) if court_x is not None and court_y is not None else (Point2D(0.5, 0.5) if ball_px else None)
    return FrameObservation(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        ball_px=ball_px,
        ball_court=ball_court,
        ball_confidence=confidence if ball_px is not None else 0.0,
    )


class BallCleanupTests(unittest.TestCase):
    def test_single_frame_jump_is_rejected(self) -> None:
        observations = [
            _observation(0, 0.0, 0.0),
            _observation(1, 500.0, 500.0),
            _observation(2, 10.0, 0.0),
        ]

        segment_breaks = _clean_ball_track(observations, fps=30.0, thresholds=Thresholds())

        self.assertEqual(segment_breaks, set())
        self.assertIsNone(observations[1].ball_px)
        self.assertIsNone(observations[1].ball_court)
        self.assertEqual(observations[1].ball_confidence, 0.0)

    def test_short_false_burst_between_plausible_points_is_rejected(self) -> None:
        observations = [
            _observation(0, 0.0, 0.0),
            _observation(1, 500.0, 500.0),
            _observation(2, 510.0, 510.0),
            _observation(3, 30.0, 0.0),
        ]

        _clean_ball_track(observations, fps=30.0, thresholds=Thresholds(ball_outlier_run_max_frames=2))

        self.assertIsNone(observations[1].ball_px)
        self.assertIsNone(observations[2].ball_px)
        self.assertIsNotNone(observations[3].ball_px)

    def test_static_distractor_segment_is_rejected(self) -> None:
        observations = [
            _observation(index, 100.0 + (index % 2), 200.0 + (index % 2), confidence=0.95)
            for index in range(6)
        ]

        _clean_ball_track(observations, fps=30.0, thresholds=Thresholds())

        self.assertTrue(all(observation.ball_px is None for observation in observations))

    def test_long_no_ball_gap_allows_reappearance_anywhere(self) -> None:
        observations = [_observation(0, 0.0, 0.0)]
        observations.extend(_observation(index, None, None) for index in range(1, 32))
        observations.append(_observation(32, 5000.0, 5000.0))

        segment_breaks = _clean_ball_track(observations, fps=30.0, thresholds=Thresholds())
        _compute_ball_velocity(observations, fps=30.0, segment_breaks=segment_breaks)

        self.assertIn(32, segment_breaks)
        self.assertIsNotNone(observations[32].ball_px)
        self.assertIsNone(observations[32].ball_velocity_px_s)

    def test_short_gap_is_interpolated_after_cleanup(self) -> None:
        observations = [
            _observation(0, 0.0, 0.0),
            _observation(1, None, None),
            _observation(2, 10.0, 0.0),
        ]

        segment_breaks = _clean_ball_track(observations, fps=30.0, thresholds=Thresholds())
        _interpolate_ball_track(observations, max_gap=5, segment_breaks=segment_breaks)

        self.assertIsNotNone(observations[1].ball_px)
        self.assertAlmostEqual(observations[1].ball_px.x, 5.0)
        self.assertAlmostEqual(observations[1].ball_px.y, 0.0)
        self.assertAlmostEqual(observations[1].ball_confidence, 0.675)

    def test_medium_gap_is_not_filled_by_smoothing(self) -> None:
        observations = [_observation(0, 0.0, 0.0)]
        observations.extend(_observation(index, None, None) for index in range(1, 7))
        observations.append(_observation(7, 70.0, 0.0))

        segment_breaks = _clean_ball_track(observations, fps=30.0, thresholds=Thresholds())
        _interpolate_ball_track(observations, max_gap=5, segment_breaks=segment_breaks)
        _smooth_ball_track(observations, window_size=3, segment_breaks=segment_breaks)

        self.assertIsNone(observations[1].ball_px)
        self.assertIsNone(observations[6].ball_px)

    def test_long_gap_created_by_rejected_reappearance_starts_new_segment(self) -> None:
        observations = [_observation(0, 0.0, 0.0)]
        observations.extend(_observation(index, None, None) for index in range(1, 7))
        observations.append(_observation(7, 5000.0, 5000.0))
        observations.extend(_observation(index, None, None) for index in range(8, 32))
        observations.append(_observation(32, 9000.0, 9000.0))

        segment_breaks = _clean_ball_track(observations, fps=30.0, thresholds=Thresholds())

        self.assertIsNone(observations[7].ball_px)
        self.assertIn(32, segment_breaks)
        self.assertIsNotNone(observations[32].ball_px)

    def test_out_of_play_run_starts_new_segment(self) -> None:
        observations = [
            _observation(0, 0.0, 0.0, court_x=2.0, court_y=0.5),
            _observation(1, 1.0, 0.0, court_x=2.0, court_y=0.5),
            _observation(2, 2.0, 0.0, court_x=2.0, court_y=0.5),
            _observation(3, 3.0, 0.0, court_x=2.0, court_y=0.5),
            _observation(4, 500.0, 500.0, court_x=0.5, court_y=0.5),
        ]

        segment_breaks = _clean_ball_track(observations, fps=30.0, thresholds=Thresholds())
        _smooth_ball_track(observations, window_size=3, segment_breaks=segment_breaks)
        _compute_ball_velocity(observations, fps=30.0, segment_breaks=segment_breaks)

        self.assertIn(4, segment_breaks)
        self.assertIsNotNone(observations[4].ball_px)
        self.assertAlmostEqual(observations[4].ball_px.x, 500.0)
        self.assertAlmostEqual(observations[4].ball_px.y, 500.0)
        self.assertIsNone(observations[4].ball_velocity_px_s)


if __name__ == "__main__":
    unittest.main()
