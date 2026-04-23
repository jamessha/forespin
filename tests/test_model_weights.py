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

from forespin.domain import Handedness, InputConfig, TrackedPlayerSide
from forespin.model_weights import (
    DEFAULT_COURT_WEIGHTS_PATH,
    DEFAULT_PLAYER_POSE_WEIGHTS_PATH,
    DEFAULT_TRACKNET_WEIGHTS_PATH,
    MissingModelWeightsError,
    discover_default_local_artifact,
    resolve_local_artifact_path,
    resolve_model_weights,
)


class ModelWeightsTests(unittest.TestCase):
    def test_resolve_local_artifact_path_prefers_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"stub")
            resolved = resolve_local_artifact_path(
                local_path=str(model_path),
                model_label="TestModel",
                missing_hint="unused",
            )
        self.assertTrue(resolved.endswith("model.pt"))

    def test_resolve_model_weights_uses_explicit_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tracknet_path = Path(temp_dir) / "tracknet.torchscript.pt"
            player_path = Path(temp_dir) / "yolo26n-pose.pt"
            court_path = Path(temp_dir) / "tennis_court_detector.pt"
            tracknet_path.write_bytes(b"stub")
            player_path.write_bytes(b"stub")
            court_path.write_bytes(b"stub")
            resolved = resolve_model_weights(
                InputConfig(
                    video_path="match.mp4",
                    tracked_player_side=TrackedPlayerSide.NEAR,
                    handedness=Handedness.RIGHT,
                    tracknet_weights=str(tracknet_path),
                    player_pose_weights=str(player_path),
                    court_weights=str(court_path),
                )
            )
        self.assertTrue(resolved.tracknet_weights.endswith("tracknet.torchscript.pt"))
        self.assertTrue(resolved.player_pose_weights.endswith("yolo26n-pose.pt"))
        self.assertTrue(resolved.court_weights.endswith("tennis_court_detector.pt"))

    def test_resolve_model_weights_uses_default_directories_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            default_tracknet = Path(temp_dir) / "tracknetv2.torchscript.pt"
            default_player = Path(temp_dir) / "yolo26n-pose.pt"
            default_court = Path(temp_dir) / "tennis_court_detector.pt"
            default_tracknet.write_bytes(b"stub")
            default_player.write_bytes(b"stub")
            default_court.write_bytes(b"stub")
            with patch("forespin.model_weights.DEFAULT_TRACKNET_WEIGHTS_PATH", default_tracknet):
                with patch("forespin.model_weights.DEFAULT_PLAYER_POSE_WEIGHTS_PATH", default_player):
                    with patch("forespin.model_weights.DEFAULT_COURT_WEIGHTS_PATH", default_court):
                        resolved = resolve_model_weights(
                            InputConfig(
                                video_path="match.mp4",
                                tracked_player_side=TrackedPlayerSide.NEAR,
                                handedness=Handedness.RIGHT,
                            )
                        )
        self.assertTrue(resolved.tracknet_weights.endswith("tracknetv2.torchscript.pt"))
        self.assertTrue(resolved.player_pose_weights.endswith("yolo26n-pose.pt"))
        self.assertTrue(resolved.court_weights.endswith("tennis_court_detector.pt"))

    def test_resolve_model_weights_requires_yolo_path_when_default_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tracknet_path = Path(temp_dir) / "tracknet.torchscript.pt"
            court_path = Path(temp_dir) / "tennis_court_detector.pt"
            tracknet_path.write_bytes(b"stub")
            court_path.write_bytes(b"stub")
            with patch("forespin.model_weights.DEFAULT_PLAYER_POSE_WEIGHTS_PATH", Path(temp_dir) / "missing.pt"):
                with self.assertRaisesRegex(MissingModelWeightsError, "YOLO26 pose weights are required"):
                    resolve_model_weights(
                        InputConfig(
                            video_path="match.mp4",
                            tracked_player_side=TrackedPlayerSide.NEAR,
                            handedness=Handedness.RIGHT,
                            tracknet_weights=str(tracknet_path),
                            court_weights=str(court_path),
                        )
                    )

    def test_resolve_model_weights_requires_court_path_when_default_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tracknet_path = Path(temp_dir) / "tracknet.torchscript.pt"
            player_path = Path(temp_dir) / "yolo26n-pose.pt"
            tracknet_path.write_bytes(b"stub")
            player_path.write_bytes(b"stub")
            with patch("forespin.model_weights.DEFAULT_COURT_WEIGHTS_PATH", Path(temp_dir) / "missing.pt"):
                with patch("forespin.model_weights.DEFAULT_COURT_WEIGHTS_DIR", Path(temp_dir) / "missing-court-dir"):
                    with self.assertRaisesRegex(MissingModelWeightsError, "Learned court detector weights are required"):
                        resolve_model_weights(
                            InputConfig(
                                video_path="match.mp4",
                                tracked_player_side=TrackedPlayerSide.NEAR,
                                handedness=Handedness.RIGHT,
                                tracknet_weights=str(tracknet_path),
                                player_pose_weights=str(player_path),
                            )
                        )

    def test_discover_default_local_artifact_accepts_single_checkpoint_in_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint = temp_path / "downloaded_model.pth"
            checkpoint.write_bytes(b"stub")
            discovered = discover_default_local_artifact(
                default_path=DEFAULT_COURT_WEIGHTS_PATH,
                search_dir=temp_path,
            )
        self.assertEqual(discovered, checkpoint)

    def test_resolve_local_artifact_path_requires_local_file(self) -> None:
        with self.assertRaisesRegex(MissingModelWeightsError, "weights are required"):
            resolve_local_artifact_path(
                local_path=None,
                model_label="TestModel",
                missing_hint="unused",
            )


if __name__ == "__main__":
    unittest.main()
