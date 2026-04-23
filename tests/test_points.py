from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.config import Thresholds
from forespin.domain import BounceEvent, FrameObservation, HitEvent, InputConfig, PlayerActor, Point2D, PointTermination, ShotOutcome, ShotType, TrackedPlayerSide, Handedness
from forespin.points import assemble_points


class PointTests(unittest.TestCase):
    def test_assemble_points_marks_lost_point_and_final_location(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
        )
        thresholds = Thresholds(dead_ball_gap_frames=20)
        observations = [
            FrameObservation(frame_index=index, timestamp_s=index / 30.0, tracked_player_feet_court=Point2D(0.35 + index * 0.001, 0.88))
            for index in range(80)
        ]
        hits = [
            HitEvent(10, 0.33, PlayerActor.TRACKED, ShotType.SERVE, None, Point2D(0.5, 0.92), Point2D(0.5, 0.94), 0.9, is_serve=True, result=ShotOutcome.IN),
            HitEvent(22, 0.73, PlayerActor.OPPONENT, ShotType.UNKNOWN, None, Point2D(0.45, 0.12), None, 0.7),
            HitEvent(35, 1.16, PlayerActor.TRACKED, ShotType.FOREHAND, None, Point2D(0.58, 0.78), Point2D(0.52, 0.89), 0.8, result=ShotOutcome.OUT),
            HitEvent(66, 2.20, PlayerActor.TRACKED, ShotType.SERVE, None, Point2D(0.5, 0.92), Point2D(0.5, 0.94), 0.9, is_serve=True, result=ShotOutcome.IN),
        ]
        bounces = [
            BounceEvent(frame_index=28, timestamp_s=0.93, ball_court=Point2D(0.54, 0.80), in_bounds=True, confidence=0.8),
        ]

        points = assemble_points(observations, hits, bounces, input_config, thresholds)

        self.assertEqual(len(points), 2)
        self.assertTrue(points[0].tracked_player_lost)
        self.assertEqual(points[0].terminal_reason, PointTermination.TRACKED_ERROR)
        self.assertIsNotNone(points[0].tracked_player_final_location)


if __name__ == "__main__":
    unittest.main()
