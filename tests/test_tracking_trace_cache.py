from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.config import AnalysisOptions
from forespin.domain import CourtCalibration, FrameObservation, Handedness, InputConfig, Point2D, TrackedPlayerSide, VideoMetadata
from forespin.model_weights import ResolvedModelWeights
from forespin.tracking_trace_cache import read_cached_tracking_trace, tracking_trace_cache_path, write_cached_tracking_trace


class TrackingTraceCacheTests(unittest.TestCase):
    def test_tracking_trace_cache_round_trips_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "match.mp4"
            tracknet_path = root / "tracknet.pt"
            player_path = root / "player.pt"
            court_path = root / "court.pt"
            for path in (video_path, tracknet_path, player_path, court_path):
                path.write_bytes(b"stub")

            input_config = InputConfig(
                video_path=str(video_path),
                tracked_player_side=TrackedPlayerSide.NEAR,
                handedness=Handedness.RIGHT,
                output_dir=str(root / "outputs"),
            )
            metadata = VideoMetadata(width=1920, height=1080, fps=30.0, frame_count=1, duration_s=1 / 30)
            weights = ResolvedModelWeights(str(tracknet_path), str(player_path), str(court_path))
            options = AnalysisOptions()
            court = CourtCalibration(
                corners_px=[Point2D(0, 0), Point2D(1, 0), Point2D(1, 1), Point2D(0, 1)],
                homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                confidence=0.9,
                source="learned_court",
            )
            observations = [
                FrameObservation(
                    frame_index=0,
                    timestamp_s=0.0,
                    ball_px=Point2D(100, 200),
                    ball_court=Point2D(0.4, 0.7),
                    ball_confidence=0.8,
                    court_confidence=0.9,
                )
            ]
            cache_path = tracking_trace_cache_path(output_dir=Path(input_config.output_dir), video_path=video_path)

            write_cached_tracking_trace(
                cache_path,
                video_path=video_path,
                input_config=input_config,
                metadata=metadata,
                resolved_weights=weights,
                options=options,
                observations=observations,
                court=court,
            )
            cached = read_cached_tracking_trace(
                cache_path,
                video_path=video_path,
                input_config=input_config,
                metadata=metadata,
                resolved_weights=weights,
                options=options,
            )

        self.assertIsNotNone(cached)
        cached_metadata, cached_observations, cached_court = cached
        self.assertEqual(cached_metadata.frame_count, 1)
        self.assertEqual(cached_observations[0].ball_px, Point2D(100, 200))
        self.assertEqual(cached_court.source, "learned_court:trace_cache")

    def test_tracking_trace_cache_invalidates_when_video_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "match.mp4"
            tracknet_path = root / "tracknet.pt"
            player_path = root / "player.pt"
            court_path = root / "court.pt"
            for path in (video_path, tracknet_path, player_path, court_path):
                path.write_bytes(b"stub")

            input_config = InputConfig(
                video_path=str(video_path),
                tracked_player_side=TrackedPlayerSide.NEAR,
                handedness=Handedness.RIGHT,
                output_dir=str(root / "outputs"),
            )
            metadata = VideoMetadata(width=1920, height=1080, fps=30.0, frame_count=1, duration_s=1 / 30)
            weights = ResolvedModelWeights(str(tracknet_path), str(player_path), str(court_path))
            options = AnalysisOptions()
            court = CourtCalibration(
                corners_px=[Point2D(0, 0), Point2D(1, 0), Point2D(1, 1), Point2D(0, 1)],
                homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                confidence=0.9,
            )
            cache_path = tracking_trace_cache_path(output_dir=Path(input_config.output_dir), video_path=video_path)
            write_cached_tracking_trace(
                cache_path,
                video_path=video_path,
                input_config=input_config,
                metadata=metadata,
                resolved_weights=weights,
                options=options,
                observations=[],
                court=court,
            )
            video_path.write_bytes(b"changed")

            cached = read_cached_tracking_trace(
                cache_path,
                video_path=video_path,
                input_config=input_config,
                metadata=metadata,
                resolved_weights=weights,
                options=options,
            )

        self.assertIsNone(cached)


if __name__ == "__main__":
    unittest.main()
