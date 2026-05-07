from __future__ import annotations

import argparse
import json
from pathlib import Path

from forespin.analysis import TennisAnalyzer
from forespin.config import AnalysisOptions
from forespin.downloads import DEFAULT_YOLO_HF_FILE, DEFAULT_YOLO_HF_REPO, download_yolo_weights
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forespin", description="Analyze a fixed baseline tennis video.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run the full video analysis pipeline.")
    analyze.add_argument("video_path", help="Path to the input video.")
    analyze.add_argument("--player-side", choices=[item.value for item in TrackedPlayerSide], required=True)
    analyze.add_argument("--handedness", choices=[item.value for item in Handedness], required=True)
    analyze.add_argument("--output-dir", default="outputs", help="Directory for timeline and overlay artifacts.")
    analyze.add_argument(
        "--tracknet-weights",
        help=(
            "Path to a local TrackNetV2 state-dict or TorchScript weights file. "
            f"If omitted, the app will look for {DEFAULT_TRACKNET_WEIGHTS_PATH}."
        ),
    )
    analyze.add_argument(
        "--player-pose-weights",
        help=(
            "Path to a local YOLO26 pose weights file. "
            f"If omitted, the app will look for {DEFAULT_PLAYER_POSE_WEIGHTS_PATH}."
        ),
    )
    analyze.add_argument(
        "--court-weights",
        help=(
            "Path to a local learned tennis-court detector checkpoint. "
            f"If omitted, the app will look in {DEFAULT_COURT_WEIGHTS_DIR} "
            f"and prefer {DEFAULT_COURT_WEIGHTS_PATH}."
        ),
    )
    analyze.add_argument("--skip-overlay", action="store_true", help="Do not render the overlay MP4.")
    analyze.add_argument("--allow-low-quality", action="store_true", help="Persist metrics even when the quality gate rejects the clip.")
    analyze.add_argument(
        "--remove-net-for-court-calibration",
        action="store_true",
        help=(
            "Use OpenAI image editing to remove the net from the first frame before court calibration. "
            "Requires OPENAI_API_KEY in the environment or a .env file."
        ),
    )
    analyze.add_argument(
        "--net-removal-model",
        default="gpt-image-2",
        help="OpenAI image model for --remove-net-for-court-calibration. Defaults to gpt-image-2.",
    )
    analyze.add_argument(
        "--no-court-cache",
        action="store_true",
        help="Ignore and overwrite the per-video cached court calibration.",
    )

    download = subparsers.add_parser("download", help="Download supported model weights into the repo.")
    download.add_argument("artifact", choices=["yolo"], help="The model artifact to download.")
    download.add_argument(
        "--repo",
        default=DEFAULT_YOLO_HF_REPO,
        help=f"Hugging Face repo id to download from. Defaults to {DEFAULT_YOLO_HF_REPO}.",
    )
    download.add_argument(
        "--file",
        dest="filename",
        default=DEFAULT_YOLO_HF_FILE,
        help=f"Filename to download. Defaults to {DEFAULT_YOLO_HF_FILE}.",
    )
    download.add_argument(
        "--output-dir",
        default=str(DEFAULT_PLAYER_POSE_WEIGHTS_DIR),
        help=f"Directory to save the downloaded weights. Defaults to {DEFAULT_PLAYER_POSE_WEIGHTS_DIR}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "download":
        return _handle_download(args)
    if args.command != "analyze":
        parser.error(f"Unsupported command: {args.command}")
    _validate_weight_args(parser, args)

    options = AnalysisOptions(
        render_overlay=not args.skip_overlay,
        persist_overlay=not args.skip_overlay,
        reject_low_quality=not args.allow_low_quality,
        remove_net_for_court_calibration=args.remove_net_for_court_calibration,
        net_removal_model=args.net_removal_model,
        use_court_calibration_cache=not args.no_court_cache,
    )
    input_config = InputConfig(
        video_path=str(Path(args.video_path).expanduser().resolve()),
        tracked_player_side=TrackedPlayerSide(args.player_side),
        handedness=Handedness(args.handedness),
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        tracknet_weights=str(Path(args.tracknet_weights).expanduser().resolve()) if args.tracknet_weights else None,
        player_pose_weights=str(Path(args.player_pose_weights).expanduser().resolve()) if args.player_pose_weights else None,
        court_weights=str(Path(args.court_weights).expanduser().resolve()) if args.court_weights else None,
    )
    result = TennisAnalyzer(options=options).analyze_and_persist(input_config)
    print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
    return 0


def _validate_weight_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.tracknet_weights and not has_default_tracknet_weights():
        parser.error(
            "TrackNetV2 weights are required. Place tracknetv2.torchscript.pt in "
            f"{DEFAULT_TRACKNET_WEIGHTS_DIR} or provide --tracknet-weights."
        )
    if not args.player_pose_weights and not has_default_player_pose_weights():
        parser.error(
            "YOLO26 pose weights are required. Run `forespin download yolo`, place yolo26n-pose.pt in "
            f"{DEFAULT_PLAYER_POSE_WEIGHTS_DIR}, or provide --player-pose-weights."
        )
    if not args.court_weights and not has_default_court_weights():
        parser.error(
            "Learned court detector weights are required. Place a checkpoint in "
            f"{DEFAULT_COURT_WEIGHTS_DIR} or provide --court-weights."
        )


def _handle_download(args: argparse.Namespace) -> int:
    if args.artifact != "yolo":
        raise ValueError(f"Unsupported artifact: {args.artifact}")
    downloaded = download_yolo_weights(
        repo_id=args.repo,
        filename=args.filename,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "artifact": "yolo",
                "repo": args.repo,
                "file": args.filename,
                "saved_to": str(downloaded),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
