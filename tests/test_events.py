from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.config import Thresholds
from forespin.domain import BounceEvent, FrameObservation, Handedness, HitEvent, InputConfig, PlayerActor, Point2D, ShotOutcome, ShotType, TrackedPlayerSide
from forespin.events import annotate_hit_outcomes, annotate_shot_types, detect_bounces


class EventTests(unittest.TestCase):
    def test_detect_bounces_accepts_floor_contact_local_vertical_maximum(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
        )
        observations = _ball_observations([160, 168, 176, 182, 186, 190, 184, 180, 176, 172])

        bounces = detect_bounces(observations, [], input_config, Thresholds())

        self.assertEqual([bounce.frame_index for bounce in bounces], [5])
        self.assertTrue(bounces[0].in_bounds)

    def test_detect_bounces_rejects_monotonic_ball_motion(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
        )
        observations = _ball_observations([100, 122, 146, 170, 196, 224, 250, 278, 304, 330])

        bounces = detect_bounces(observations, [], input_config, Thresholds())

        self.assertEqual(bounces, [])

    def test_detect_bounces_suppresses_hit_contact_reversals(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
        )
        observations = _ball_observations([160, 168, 176, 182, 186, 190, 184, 180, 176, 172])
        hits = [
            HitEvent(
                frame_index=4,
                timestamp_s=4 / 30.0,
                actor=PlayerActor.TRACKED,
                shot_type=ShotType.UNKNOWN,
                ball_px=Point2D(100, 230),
                ball_court=Point2D(0.5, 0.7),
                player_court=Point2D(0.5, 0.9),
                confidence=0.9,
            )
        ]

        bounces = detect_bounces(observations, hits, input_config, Thresholds())

        self.assertEqual(bounces, [])

    def test_detect_bounces_allows_strong_floor_contact_near_bad_hit_label(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
        )
        observations = _ball_observations([100, 125, 150, 190, 230, 252, 244, 220, 196, 176])
        hits = [
            HitEvent(
                frame_index=4,
                timestamp_s=4 / 30.0,
                actor=PlayerActor.TRACKED,
                shot_type=ShotType.UNKNOWN,
                ball_px=Point2D(100, 230),
                ball_court=Point2D(0.5, 0.7),
                player_court=Point2D(0.5, 0.9),
                confidence=0.9,
            )
        ]

        bounces = detect_bounces(observations, hits, input_config, Thresholds())

        self.assertEqual([bounce.frame_index for bounce in bounces], [5])

    def test_detect_bounces_accepts_court_projection_y_turn_when_pixel_y_is_monotonic(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
        )
        observations = _ball_observations_with_court_y(
            px_y_values=[100, 104, 108, 112, 116, 120, 124, 128, 132, 136],
            court_y_values=[0.62, 0.68, 0.73, 0.77, 0.80, 0.815, 0.805, 0.78, 0.74, 0.70],
        )

        bounces = detect_bounces(observations, [], input_config, Thresholds())

        self.assertEqual([bounce.frame_index for bounce in bounces], [5])
        self.assertTrue(bounces[0].in_bounds)

    def test_detect_bounces_rejects_monotonic_court_projection_motion(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
        )
        observations = _ball_observations_with_court_y(
            px_y_values=[100, 104, 108, 112, 116, 120, 124, 128, 132, 136],
            court_y_values=[0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80],
        )

        bounces = detect_bounces(observations, [], input_config, Thresholds())

        self.assertEqual(bounces, [])

    def test_annotate_shot_types_marks_serve_forehand_and_volley(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
        )
        thresholds = Thresholds()
        hits = [
            HitEvent(
                frame_index=10,
                timestamp_s=0.33,
                actor=PlayerActor.TRACKED,
                shot_type=ShotType.UNKNOWN,
                ball_px=None,
                ball_court=Point2D(0.55, 0.92),
                player_court=Point2D(0.50, 0.95),
                confidence=0.9,
            ),
            HitEvent(
                frame_index=24,
                timestamp_s=0.80,
                actor=PlayerActor.OPPONENT,
                shot_type=ShotType.UNKNOWN,
                ball_px=None,
                ball_court=Point2D(0.44, 0.10),
                player_court=None,
                confidence=0.7,
            ),
            HitEvent(
                frame_index=44,
                timestamp_s=1.46,
                actor=PlayerActor.TRACKED,
                shot_type=ShotType.UNKNOWN,
                ball_px=None,
                ball_court=Point2D(0.58, 0.68),
                player_court=Point2D(0.52, 0.70),
                confidence=0.8,
            ),
            HitEvent(
                frame_index=62,
                timestamp_s=2.06,
                actor=PlayerActor.OPPONENT,
                shot_type=ShotType.UNKNOWN,
                ball_px=None,
                ball_court=Point2D(0.40, 0.12),
                player_court=None,
                confidence=0.7,
            ),
            HitEvent(
                frame_index=73,
                timestamp_s=2.43,
                actor=PlayerActor.TRACKED,
                shot_type=ShotType.UNKNOWN,
                ball_px=None,
                ball_court=Point2D(0.62, 0.63),
                player_court=Point2D(0.51, 0.61),
                confidence=0.9,
            ),
        ]
        bounces = [
            BounceEvent(frame_index=35, timestamp_s=1.16, ball_court=Point2D(0.46, 0.78), in_bounds=True, confidence=0.8),
        ]

        annotated = annotate_shot_types(hits, [], bounces, input_config, thresholds)

        self.assertEqual(annotated[0].shot_type, ShotType.SERVE)
        self.assertEqual(annotated[2].shot_type, ShotType.FOREHAND)
        self.assertEqual(annotated[4].shot_type, ShotType.VOLLEY)

    def test_annotate_hit_outcomes_uses_first_continuation(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
        )
        hits = [
            HitEvent(10, 0.33, PlayerActor.TRACKED, ShotType.FOREHAND, None, Point2D(0.55, 0.80), Point2D(0.50, 0.90), 0.9),
            HitEvent(20, 0.66, PlayerActor.OPPONENT, ShotType.UNKNOWN, None, Point2D(0.45, 0.15), None, 0.8),
            HitEvent(40, 1.33, PlayerActor.TRACKED, ShotType.BACKHAND, None, Point2D(0.42, 0.78), Point2D(0.50, 0.89), 0.9),
        ]
        bounces = [
            BounceEvent(frame_index=52, timestamp_s=1.73, ball_court=Point2D(1.05, 0.22), in_bounds=False, confidence=0.7),
        ]

        annotated = annotate_hit_outcomes(hits, bounces, input_config)

        self.assertEqual(annotated[0].result, ShotOutcome.IN)
        self.assertEqual(annotated[1].result, ShotOutcome.UNKNOWN)
        self.assertEqual(annotated[2].result, ShotOutcome.OUT)


def _ball_observations(y_values: list[float]) -> list[FrameObservation]:
    return [
        FrameObservation(
            frame_index=index,
            timestamp_s=index / 30.0,
            ball_px=Point2D(100 + index * 8, y),
            ball_court=Point2D(0.5, 0.7),
            ball_confidence=0.9,
        )
        for index, y in enumerate(y_values)
    ]


def _ball_observations_with_court_y(px_y_values: list[float], court_y_values: list[float]) -> list[FrameObservation]:
    return [
        FrameObservation(
            frame_index=index,
            timestamp_s=index / 30.0,
            ball_px=Point2D(100 + index * 8, px_y),
            ball_court=Point2D(0.5 + index * 0.01, court_y),
            ball_confidence=0.9,
        )
        for index, (px_y, court_y) in enumerate(zip(px_y_values, court_y_values, strict=True))
    ]


if __name__ == "__main__":
    unittest.main()
