from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.config import Thresholds
from forespin.domain import HitEvent, InputConfig, MetricBucket, PlayerActor, Point2D, PointEvent, PointTermination, ShotMetric, ShotOutcome, ShotType, TrackedPlayerSide, Handedness
from forespin.metrics import compute_metrics


class MetricTests(unittest.TestCase):
    def test_compute_metrics_aggregates_shots_and_loss_zones(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.LEFT,
        )
        hits = [
            HitEvent(10, 0.33, PlayerActor.TRACKED, ShotType.SERVE, None, Point2D(0.5, 0.9), Point2D(0.5, 0.93), 0.9, result=ShotOutcome.IN),
            HitEvent(35, 1.16, PlayerActor.TRACKED, ShotType.FOREHAND, None, Point2D(0.42, 0.8), Point2D(0.5, 0.9), 0.9, result=ShotOutcome.OUT),
            HitEvent(48, 1.60, PlayerActor.TRACKED, ShotType.VOLLEY, None, Point2D(0.55, 0.6), Point2D(0.5, 0.62), 0.9, result=ShotOutcome.IN),
            HitEvent(60, 2.00, PlayerActor.TRACKED, ShotType.BACKHAND, None, Point2D(0.64, 0.82), Point2D(0.5, 0.89), 0.9, result=ShotOutcome.UNKNOWN),
        ]
        points = [
            PointEvent(
                point_index=1,
                start_frame=10,
                end_frame=35,
                hit_events=[],
                bounce_events=[],
                tracked_player_lost=True,
                winner=PlayerActor.OPPONENT,
                terminal_reason=PointTermination.TRACKED_ERROR,
                tracked_player_final_location=Point2D(0.28, 0.91),
                confidence=0.8,
            ),
            PointEvent(
                point_index=2,
                start_frame=48,
                end_frame=70,
                hit_events=[],
                bounce_events=[],
                tracked_player_lost=False,
                winner=PlayerActor.TRACKED,
                terminal_reason=PointTermination.TRACKED_WINNER,
                tracked_player_final_location=Point2D(0.62, 0.62),
                confidence=0.8,
            ),
        ]

        metrics = compute_metrics(hits, points, input_config, Thresholds())

        self.assertEqual(metrics.shot_stats["serve"].attempts, 1)
        self.assertEqual(metrics.shot_stats["serve"].in_rate, 1.0)
        self.assertEqual(metrics.shot_stats["forehand"].attempts, 1)
        self.assertEqual(metrics.shot_stats["forehand"].in_rate, 0.0)
        self.assertEqual(metrics.lost_point_depth["baseline"].count, 1)
        self.assertEqual(metrics.lost_point_lateral["left"].count, 1)
        self.assertEqual(metrics.ignored_tracked_shots, 1)


if __name__ == "__main__":
    unittest.main()
