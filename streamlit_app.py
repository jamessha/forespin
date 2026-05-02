from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.analysis import TennisAnalyzer
from forespin.config import AnalysisOptions
from forespin.deps import MissingDependencyError, require_streamlit
from forespin.domain import Handedness, InputConfig, TrackedPlayerSide
from forespin.model_weights import (
    DEFAULT_COURT_WEIGHTS_DIR,
    DEFAULT_COURT_WEIGHTS_PATH,
    DEFAULT_PLAYER_POSE_WEIGHTS_DIR,
    DEFAULT_PLAYER_POSE_WEIGHTS_PATH,
    DEFAULT_TRACKNET_WEIGHTS_DIR,
    DEFAULT_TRACKNET_WEIGHTS_PATH,
    has_default_court_weights,
    has_default_player_pose_weights,
    has_default_tracknet_weights,
)
from forespin.serialization import to_jsonable


def main() -> None:
    st = require_streamlit()
    st.set_page_config(page_title="Forespin", layout="wide")
    st.title("Forespin")
    st.caption("Batch analysis for fixed elevated-baseline tennis recordings.")

    uploaded = st.file_uploader("Upload a tennis video", type=["mp4", "mov", "m4v", "avi"])
    player_side = st.selectbox("Tracked player side", [item.value for item in TrackedPlayerSide], index=0)
    handedness = st.selectbox("Handedness", [item.value for item in Handedness], index=0)
    st.subheader("TrackNetV2 Ball Model")
    tracknet_weights = st.text_input("Local TrackNetV2 TorchScript path (optional)")
    st.caption(
        f"If left blank, the app will look for {DEFAULT_TRACKNET_WEIGHTS_PATH}. "
        f"Current status: {'found' if has_default_tracknet_weights() else 'not found'}."
    )
    st.subheader("YOLO26 Pose Model")
    player_pose_weights = st.text_input("Local YOLO26 pose weights path (optional)")
    st.caption(
        f"If left blank, the app will look for {DEFAULT_PLAYER_POSE_WEIGHTS_PATH}. "
        f"Current status: {'found' if has_default_player_pose_weights() else 'not found'}. "
        "Use `forespin download yolo` to fetch the default checkpoint."
    )
    st.subheader("Learned Court Detector")
    court_weights = st.text_input("Local tennis-court detector weights path (optional)")
    st.caption(
        f"If left blank, the app will look in {DEFAULT_COURT_WEIGHTS_DIR} and prefer "
        f"{DEFAULT_COURT_WEIGHTS_PATH}. Current status: "
        f"{'found' if has_default_court_weights() else 'not found'}."
    )
    allow_low_quality = st.checkbox("Persist low-confidence results", value=False)
    remove_net_for_court_calibration = st.checkbox(
        "Remove net from first frame before court calibration with OpenAI",
        value=False,
    )
    if remove_net_for_court_calibration:
        st.caption("Requires `OPENAI_API_KEY` in the environment or a local `.env` file.")

    if uploaded is None:
        st.info("Upload a fixed elevated-baseline clip to begin.")
        return

    if st.button("Analyze video", type="primary"):
        if not tracknet_weights.strip() and not has_default_tracknet_weights():
            st.error(
                "Provide TrackNetV2 weights as a local path or place tracknetv2.torchscript.pt "
                f"in {DEFAULT_TRACKNET_WEIGHTS_DIR}."
            )
            return
        if not player_pose_weights.strip() and not has_default_player_pose_weights():
            st.error(
                "Provide YOLO26 pose weights as a local path, place yolo26n-pose.pt in "
                f"{DEFAULT_PLAYER_POSE_WEIGHTS_DIR}, or run `forespin download yolo`."
            )
            return
        if not court_weights.strip() and not has_default_court_weights():
            st.error(
                "Provide learned court detector weights as a local path or place a checkpoint in "
                f"{DEFAULT_COURT_WEIGHTS_DIR}."
            )
            return
        try:
            with tempfile.TemporaryDirectory(prefix="forespin-") as temp_dir:
                temp_root = Path(temp_dir)
                video_path = temp_root / uploaded.name
                video_path.write_bytes(uploaded.getbuffer())
                output_dir = ROOT / "outputs" / video_path.stem
                input_config = InputConfig(
                    video_path=str(video_path),
                    tracked_player_side=TrackedPlayerSide(player_side),
                    handedness=Handedness(handedness),
                    output_dir=str(output_dir),
                    tracknet_weights=tracknet_weights.strip() or None,
                    player_pose_weights=player_pose_weights.strip() or None,
                    court_weights=court_weights.strip() or None,
                )
                options = AnalysisOptions(
                    reject_low_quality=not allow_low_quality,
                    remove_net_for_court_calibration=remove_net_for_court_calibration,
                )
                result = TennisAnalyzer(options=options).analyze_and_persist(input_config)
        except MissingDependencyError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - UI surface
            st.exception(exc)
            return

        if result.status.value == "rejected":
            st.warning("Clip rejected by the quality gate.")
        else:
            st.success("Analysis complete.")

        metrics_col, location_col = st.columns(2)
        with metrics_col:
            st.subheader("Shot-In Rate")
            for shot_name, metric in result.metrics.shot_stats.items():
                st.metric(shot_name.title(), f"{metric.in_rate:.0%}", f"{metric.in_count}/{metric.attempts}")
        with location_col:
            st.subheader("Lost Points by Location")
            for zone_name, bucket in result.metrics.lost_point_depth.items():
                st.metric(zone_name.title(), f"{bucket.percentage:.0%}", bucket.count)
            for zone_name, bucket in result.metrics.lost_point_lateral.items():
                st.metric(zone_name.title(), f"{bucket.percentage:.0%}", bucket.count)

        if result.warnings:
            st.subheader("Warnings")
            for warning in result.warnings:
                st.write(f"- {warning}")

        if result.artifacts.overlay_path and Path(result.artifacts.overlay_path).exists():
            st.subheader("Overlay Playback")
            st.video(result.artifacts.overlay_path)

        st.subheader("Timeline JSON")
        st.code(json.dumps(to_jsonable(result), indent=2, sort_keys=True), language="json")


if __name__ == "__main__":
    main()
