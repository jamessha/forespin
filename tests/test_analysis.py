from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.analysis import TennisAnalyzer
from forespin.domain import Handedness, InputConfig, TrackedPlayerSide
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


if __name__ == "__main__":
    unittest.main()
