from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.analysis import TennisAnalyzer
from forespin.config import AnalysisOptions
from forespin.domain import CourtCalibration, Handedness, InputConfig, Point2D, TrackedPlayerSide
from forespin.tracking.observation_builder import build_observations
from forespin.model_weights import MissingModelWeightsError


class AnalysisValidationTests(unittest.TestCase):
    def test_analyze_requires_tracknet_source(self) -> None:
        analyzer = TennisAnalyzer()
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("forespin.model_weights.DEFAULT_TRACKNET_WEIGHTS_PATH", Path(temp_dir) / "missing-tracknet.pt"):
                with self.assertRaisesRegex(MissingModelWeightsError, "TrackNetV2 weights are required"):
                    analyzer.analyze(input_config)

    def test_analyze_requires_yolo_source_when_tracknet_exists(self) -> None:
        analyzer = TennisAnalyzer()
        with tempfile.TemporaryDirectory() as temp_dir:
            tracknet_path = Path(temp_dir) / "tracknetv2.torchscript.pt"
            court_path = Path(temp_dir) / "tennis_court_detector.pt"
            tracknet_path.write_bytes(b"stub")
            court_path.write_bytes(b"stub")
            input_config = InputConfig(
                video_path="match.mp4",
                tracked_player_side=TrackedPlayerSide.NEAR,
                handedness=Handedness.RIGHT,
            )
            with patch("forespin.model_weights.DEFAULT_TRACKNET_WEIGHTS_PATH", tracknet_path):
                with patch("forespin.model_weights.DEFAULT_COURT_WEIGHTS_PATH", court_path):
                    with patch("forespin.model_weights.DEFAULT_PLAYER_POSE_WEIGHTS_PATH", Path(temp_dir) / "missing-yolo.pt"):
                        with self.assertRaisesRegex(MissingModelWeightsError, "YOLO26 pose weights are required"):
                            analyzer.analyze(input_config)

    def test_analyze_requires_court_source_when_other_weights_exist(self) -> None:
        analyzer = TennisAnalyzer()
        with tempfile.TemporaryDirectory() as temp_dir:
            tracknet_path = Path(temp_dir) / "tracknetv2.torchscript.pt"
            player_path = Path(temp_dir) / "yolo26n-pose.pt"
            tracknet_path.write_bytes(b"stub")
            player_path.write_bytes(b"stub")
            input_config = InputConfig(
                video_path="match.mp4",
                tracked_player_side=TrackedPlayerSide.NEAR,
                handedness=Handedness.RIGHT,
            )
            with patch("forespin.model_weights.DEFAULT_TRACKNET_WEIGHTS_PATH", tracknet_path):
                with patch("forespin.model_weights.DEFAULT_PLAYER_POSE_WEIGHTS_PATH", player_path):
                    with patch("forespin.model_weights.DEFAULT_COURT_WEIGHTS_PATH", Path(temp_dir) / "missing-court.pt"):
                        with patch("forespin.model_weights.DEFAULT_COURT_WEIGHTS_DIR", Path(temp_dir) / "missing-court-dir"):
                            with self.assertRaisesRegex(MissingModelWeightsError, "Learned court detector weights are required"):
                                analyzer.analyze(input_config)

    def test_openai_net_removal_uses_first_frame_for_court_calibration(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
            tracknet_weights="tracknet.pt",
            player_pose_weights="player.pt",
            court_weights="court.pt",
        )
        options = AnalysisOptions(
            remove_net_for_court_calibration=True,
            use_court_calibration_cache=False,
            use_tracking_trace_cache=False,
        )
        fake_capture = MagicMock()
        fake_capture.isOpened.return_value = True
        fake_capture.get.return_value = 0
        fake_capture.read.return_value = (False, None)
        calibrated_court = CourtCalibration(
            corners_px=[Point2D(0.0, 0.0), Point2D(1.0, 0.0), Point2D(1.0, 1.0), Point2D(0.0, 1.0)],
            homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            confidence=0.9,
            source="learned_court",
        )

        with patch("forespin.tracking.observation_builder.resolve_model_weights") as resolve_weights:
            resolve_weights.return_value = MagicMock(
                tracknet_weights="tracknet.pt",
                player_pose_weights="player.pt",
                court_weights="court.pt",
            )
            with patch("forespin.tracking.observation_builder.require_vision_stack") as vision_stack:
                cv2 = MagicMock()
                cv2.VideoCapture.return_value = fake_capture
                cv2.CAP_PROP_FRAME_COUNT = 0
                cv2.CAP_PROP_FPS = 1
                cv2.CAP_PROP_FRAME_WIDTH = 2
                cv2.CAP_PROP_FRAME_HEIGHT = 3
                vision_stack.return_value = (cv2, object())
                with patch("forespin.tracking.observation_builder.OpenAINetRemovalPreprocessor") as preprocessor:
                    preprocessor_instance = preprocessor.return_value
                    with patch("forespin.tracking.observation_builder.CourtCalibrator") as calibrator:
                        calibrator_instance = calibrator.return_value
                        calibrator_instance.calibrate_video.return_value = calibrated_court
                        with patch("forespin.tracking.observation_builder.create_ball_tracker"):
                            with patch("forespin.tracking.observation_builder.create_player_tracker"):
                                build_observations(input_config, options)

        preprocessor.assert_called_once_with(model="gpt-image-2")
        calibrator.assert_called_once()
        self.assertIs(calibrator.call_args.kwargs["frame_preprocessor"], preprocessor_instance)
        calibrator_instance.calibrate_video.assert_called_once()
        self.assertEqual(calibrator_instance.calibrate_video.call_args.kwargs["max_scan_frames"], 30)

    def test_cached_court_calibration_skips_calibrator(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
            output_dir="outputs",
            tracknet_weights="tracknet.pt",
            player_pose_weights="player.pt",
            court_weights="court.pt",
        )
        options = AnalysisOptions(use_tracking_trace_cache=False)
        fake_capture = MagicMock()
        fake_capture.isOpened.return_value = True
        fake_capture.get.return_value = 0
        fake_capture.read.return_value = (False, None)
        cached_court = CourtCalibration(
            corners_px=[Point2D(0.0, 0.0), Point2D(1.0, 0.0), Point2D(1.0, 1.0), Point2D(0.0, 1.0)],
            homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            confidence=0.9,
            source="learned_court",
        )

        with patch("forespin.tracking.observation_builder.resolve_model_weights") as resolve_weights:
            resolve_weights.return_value = MagicMock(
                tracknet_weights="tracknet.pt",
                player_pose_weights="player.pt",
                court_weights="court.pt",
            )
            with patch("forespin.tracking.observation_builder.require_vision_stack") as vision_stack:
                cv2 = MagicMock()
                cv2.VideoCapture.return_value = fake_capture
                cv2.CAP_PROP_FRAME_COUNT = 0
                cv2.CAP_PROP_FPS = 1
                cv2.CAP_PROP_FRAME_WIDTH = 2
                cv2.CAP_PROP_FRAME_HEIGHT = 3
                vision_stack.return_value = (cv2, object())
                with patch("forespin.tracking.observation_builder.read_cached_court_calibration", return_value=cached_court):
                    with patch("forespin.tracking.observation_builder.CourtCalibrator") as calibrator:
                        with patch("forespin.tracking.observation_builder.create_ball_tracker"):
                            with patch("forespin.tracking.observation_builder.create_player_tracker"):
                                _, _, court = build_observations(input_config, options)

        calibrator.return_value.calibrate_video.assert_not_called()
        self.assertIs(court, cached_court)
        self.assertEqual(court.source, "learned_court:cache")

    def test_uncached_court_calibration_writes_cache(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
            output_dir="outputs",
            tracknet_weights="tracknet.pt",
            player_pose_weights="player.pt",
            court_weights="court.pt",
        )
        options = AnalysisOptions(use_tracking_trace_cache=False)
        fake_capture = MagicMock()
        fake_capture.isOpened.return_value = True
        fake_capture.get.return_value = 0
        fake_capture.read.return_value = (False, None)
        calibrated_court = CourtCalibration(
            corners_px=[Point2D(0.0, 0.0), Point2D(1.0, 0.0), Point2D(1.0, 1.0), Point2D(0.0, 1.0)],
            homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            confidence=0.9,
            source="learned_court",
        )

        with patch("forespin.tracking.observation_builder.resolve_model_weights") as resolve_weights:
            resolve_weights.return_value = MagicMock(
                tracknet_weights="tracknet.pt",
                player_pose_weights="player.pt",
                court_weights="court.pt",
            )
            with patch("forespin.tracking.observation_builder.require_vision_stack") as vision_stack:
                cv2 = MagicMock()
                cv2.VideoCapture.return_value = fake_capture
                cv2.CAP_PROP_FRAME_COUNT = 0
                cv2.CAP_PROP_FPS = 1
                cv2.CAP_PROP_FRAME_WIDTH = 2
                cv2.CAP_PROP_FRAME_HEIGHT = 3
                vision_stack.return_value = (cv2, object())
                with patch("forespin.tracking.observation_builder.read_cached_court_calibration", return_value=None):
                    with patch("forespin.tracking.observation_builder.write_cached_court_calibration") as write_cache:
                        with patch("forespin.tracking.observation_builder.CourtCalibrator") as calibrator:
                            calibrator.return_value.calibrate_video.return_value = calibrated_court
                            with patch("forespin.tracking.observation_builder.create_ball_tracker"):
                                with patch("forespin.tracking.observation_builder.create_player_tracker"):
                                    build_observations(input_config, options)

        calibrator.return_value.calibrate_video.assert_called_once()
        write_cache.assert_called_once()

    def test_build_observations_keeps_initial_court_fixed(self) -> None:
        input_config = InputConfig(
            video_path="match.mp4",
            tracked_player_side=TrackedPlayerSide.NEAR,
            handedness=Handedness.RIGHT,
            output_dir="outputs",
            tracknet_weights="tracknet.pt",
            player_pose_weights="player.pt",
            court_weights="court.pt",
        )
        options = AnalysisOptions(use_court_calibration_cache=False, use_tracking_trace_cache=False)
        fake_capture = MagicMock()
        fake_capture.isOpened.return_value = True
        fake_capture.get.side_effect = [2, 30, 1920, 1080]
        fake_capture.read.side_effect = [(True, object()), (True, object()), (False, None)]
        calibrated_court = CourtCalibration(
            corners_px=[Point2D(0.0, 0.0), Point2D(1.0, 0.0), Point2D(1.0, 1.0), Point2D(0.0, 1.0)],
            homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            confidence=0.9,
            source="learned_court",
        )

        with patch("forespin.tracking.observation_builder.resolve_model_weights") as resolve_weights:
            resolve_weights.return_value = MagicMock(
                tracknet_weights="tracknet.pt",
                player_pose_weights="player.pt",
                court_weights="court.pt",
            )
            with patch("forespin.tracking.observation_builder.require_vision_stack") as vision_stack:
                cv2 = MagicMock()
                cv2.VideoCapture.return_value = fake_capture
                cv2.CAP_PROP_FRAME_COUNT = 0
                cv2.CAP_PROP_FPS = 1
                cv2.CAP_PROP_FRAME_WIDTH = 2
                cv2.CAP_PROP_FRAME_HEIGHT = 3
                vision_stack.return_value = (cv2, object())
                with patch("forespin.tracking.observation_builder.CourtCalibrator") as calibrator:
                    calibrator.return_value.calibrate_video.return_value = calibrated_court
                    with patch("forespin.tracking.observation_builder.create_ball_tracker") as ball_tracker:
                        ball_tracker.return_value.track.return_value = MagicMock(position_px=None, confidence=0.0)
                        with patch("forespin.tracking.observation_builder.create_player_tracker") as player_tracker:
                            player_tracker.return_value.track.return_value = MagicMock(
                                bbox_px=None,
                                feet_px=None,
                                torso_px=None,
                                confidence=0.0,
                            )
                            build_observations(input_config, options)

        calibrator.return_value.calibrate_video.assert_called_once()
        calibrator.return_value.maybe_refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
