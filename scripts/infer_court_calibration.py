from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.config import Thresholds
from forespin.court import CourtCalibrator, LearnedCourtCalibratorBackend
from forespin.deps import require_vision_stack
from forespin.domain import CourtCalibration, Point2D
from forespin.model_weights import DEFAULT_COURT_WEIGHTS_DIR, DEFAULT_COURT_WEIGHTS_PATH, discover_default_local_artifact
from forespin.net_removal import DEFAULT_NET_REMOVAL_MODEL, OpenAINetRemovalPreprocessor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.is_dir() or input_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Expected an image file with one of {sorted(IMAGE_SUFFIXES)}: {input_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = resolve_court_weights(args.weights)
    thresholds = Thresholds(min_court_confidence=args.min_court_confidence)
    backend = LearnedCourtCalibratorBackend(
        thresholds=thresholds,
        weights_path=str(weights_path),
        plausibility_check=CourtCalibrator._corners_plausible_for_baseline_view,
    )
    frame_preprocessor = (
        OpenAINetRemovalPreprocessor(model=args.net_removal_model)
        if args.remove_net_with_openai
        else None
    )
    cv2, _ = require_vision_stack()

    result = infer_image(
        input_path=input_path,
        output_dir=output_dir,
        backend=backend,
        cv2=cv2,
        frame_preprocessor=frame_preprocessor,
    )
    payload = {
        "input_path": str(input_path),
        "weights_path": str(weights_path),
        "output_dir": str(output_dir),
        "accepted": result["accepted"],
        "result": result,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    calibration_path = output_dir / "calibration.json"
    if result["calibration"] is not None:
        calibration_path.write_text(json.dumps(result["calibration"], indent=2, sort_keys=True))
    elif calibration_path.exists():
        calibration_path.unlink()

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_reject and not result["accepted"]:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run learned tennis court calibration inference on a single image."
    )
    parser.add_argument("input_path", help="Image file to evaluate.")
    parser.add_argument(
        "--weights",
        help=(
            "Path to a court detector checkpoint. If omitted, uses the default model discovery "
            f"under {DEFAULT_COURT_WEIGHTS_DIR}."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/court_calibration", help="Directory for overlays and JSON outputs.")
    parser.add_argument(
        "--remove-net-with-openai",
        action="store_true",
        help="Use OpenAI image editing to remove the net before court calibration. Requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--net-removal-model",
        default=DEFAULT_NET_REMOVAL_MODEL,
        help=f"OpenAI image model for --remove-net-with-openai. Defaults to {DEFAULT_NET_REMOVAL_MODEL}.",
    )
    parser.add_argument(
        "--min-court-confidence",
        type=float,
        default=Thresholds().min_court_confidence,
        help="Minimum accepted calibration confidence.",
    )
    parser.add_argument("--fail-on-reject", action="store_true", help="Exit with status 1 if no frame is accepted.")
    return parser


def resolve_court_weights(weights_arg: str | None) -> Path:
    if weights_arg:
        weights_path = Path(weights_arg).expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(weights_path)
        return weights_path
    discovered = discover_default_local_artifact(default_path=DEFAULT_COURT_WEIGHTS_PATH, search_dir=DEFAULT_COURT_WEIGHTS_DIR)
    if discovered is None:
        raise FileNotFoundError(
            "No court detector checkpoint found. Place one in "
            f"{DEFAULT_COURT_WEIGHTS_DIR} or pass --weights /path/to/checkpoint.pt."
        )
    return discovered


def infer_image(
    *,
    input_path: Path,
    output_dir: Path,
    backend: LearnedCourtCalibratorBackend,
    cv2: Any,
    frame_preprocessor: OpenAINetRemovalPreprocessor | None = None,
) -> dict[str, Any]:
    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Unable to read image: {input_path}")
    if frame_preprocessor is not None:
        frame = frame_preprocessor.preprocess_frame(frame, debug_dir=output_dir, frame_index=0)
    return evaluate_and_write(
        frame=frame,
        frame_index=0,
        source=str(input_path),
        output_dir=output_dir,
        backend=backend,
    )


def evaluate_and_write(
    *,
    frame: Any,
    frame_index: int,
    source: str,
    output_dir: Path,
    backend: LearnedCourtCalibratorBackend,
) -> dict[str, Any]:
    evaluation = backend.evaluate_frame(frame)
    backend.write_debug_artifacts(output_dir, frame_index, frame, evaluation)
    diagnostics = describe_evaluation(evaluation, frame_width=frame.shape[1], frame_height=frame.shape[0])
    return {
        "frame_index": frame_index,
        "source": source,
        "accepted": evaluation["calibration"] is not None,
        "raw_image": str(output_dir / f"frame_{frame_index:03d}_raw.jpg"),
        "overlay_image": str(output_dir / f"frame_{frame_index:03d}_overlay.jpg"),
        "summary": evaluation["summary"],
        "diagnostics": diagnostics,
        "calibration": calibration_payload(evaluation["calibration"]),
        "reference_to_image": evaluation["reference_to_image"],
    }


def describe_evaluation(evaluation: dict[str, Any], *, frame_width: int, frame_height: int) -> dict[str, Any]:
    summary = evaluation["summary"]
    visible_keypoints = int(summary.get("visible_keypoints") or 0)
    confidences = [float(keypoint["confidence"]) for keypoint in summary.get("keypoints", [])]
    low_confidence_keypoints = [
        {
            "index": keypoint["index"],
            "confidence": keypoint["confidence"],
        }
        for keypoint in summary.get("keypoints", [])
        if not keypoint.get("visible")
    ]
    diagnostics: dict[str, Any] = {
        "reason": summary.get("reason"),
        "description": build_reason_description(summary, frame_width=frame_width, frame_height=frame_height),
        "visible_keypoints": visible_keypoints,
        "required_visible_keypoints": LearnedCourtCalibratorBackend.MIN_VISIBLE_KEYPOINTS,
        "mean_visible_keypoint_confidence": summary.get("mean_visible_keypoint_confidence"),
        "minimum_peak_confidence": LearnedCourtCalibratorBackend.MIN_PEAK_CONFIDENCE,
        "low_confidence_keypoints": low_confidence_keypoints,
        "reprojection_error_px": summary.get("reprojection_error_px"),
        "confidence": summary.get("confidence"),
    }
    if confidences:
        diagnostics["keypoint_confidence_range"] = {
            "min": min(confidences),
            "max": max(confidences),
        }

    corners = summary.get("corners") or []
    if len(corners) == 4 and all(corner is not None for corner in corners):
        diagnostics["corner_plausibility"] = describe_corner_plausibility(corners, frame_width=frame_width, frame_height=frame_height)
    return diagnostics


def build_reason_description(summary: dict[str, Any], *, frame_width: int, frame_height: int) -> str:
    reason = str(summary.get("reason") or "unknown")
    visible_keypoints = int(summary.get("visible_keypoints") or 0)
    if reason == "accepted":
        return "Calibration accepted."
    if reason == "The learned calibrator did not find enough court keypoints.":
        return (
            f"Only {visible_keypoints} court keypoints exceeded the peak-confidence threshold "
            f"of {LearnedCourtCalibratorBackend.MIN_PEAK_CONFIDENCE:.2f}; "
            f"at least {LearnedCourtCalibratorBackend.MIN_VISIBLE_KEYPOINTS} are required to fit a homography."
        )
    if reason == "Unable to fit a court homography from the detected keypoints.":
        return (
            "Enough keypoints were detected, but they did not match any valid court keypoint configuration "
            "needed to estimate the reference-to-image homography."
        )
    if reason == "The learned homography could not reconstruct the outer court corners.":
        return "A homography was found, but projecting the canonical outer court corners failed."
    if reason == "The reconstructed court corners were implausible for a baseline-view court.":
        return (
            "The model produced a homography, but the projected outer court failed baseline-view geometry checks. "
            f"For this {frame_width}x{frame_height} image, see corner_plausibility for the exact failed bounds."
        )
    if reason == "The learned court calibration confidence fell below the acceptance threshold.":
        confidence = float(summary.get("confidence") or 0.0)
        return (
            f"The geometry was plausible, but combined confidence was {confidence:.3f}, below the configured "
            "minimum. Confidence combines visible keypoint ratio, mean keypoint confidence, and homography consistency."
        )
    if reason == "The learned court homography was numerically unstable.":
        return "The homography matrix could not be inverted reliably, so image-to-court coordinates would be unstable."
    return reason


def describe_corner_plausibility(corners: list[dict[str, float]], *, frame_width: int, frame_height: int) -> dict[str, Any]:
    top_a, top_b, bottom_b, bottom_a = corners
    x_min = -frame_width * 4.0
    x_max = frame_width * 5.0
    y_min = -frame_height * 4.0
    y_max = frame_height * 5.0

    checks = {
        "finite_coordinates": all(math.isfinite(corner["x"]) and math.isfinite(corner["y"]) for corner in corners),
        "corners_within_loose_x_bounds": all(x_min <= corner["x"] <= x_max for corner in corners),
        "corners_within_loose_y_bounds": all(y_min <= corner["y"] <= y_max for corner in corners),
    }
    return {
        "passed": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "measurements": {
            "top_edge_delta_x_px": top_b["x"] - top_a["x"],
            "bottom_edge_delta_x_px": bottom_b["x"] - bottom_a["x"],
            "reference_side_a_delta_y_px": bottom_a["y"] - top_a["y"],
            "reference_side_b_delta_y_px": bottom_b["y"] - top_b["y"],
        },
        "bounds": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
    }


def calibration_payload(calibration: CourtCalibration | None) -> dict[str, Any] | None:
    if calibration is None:
        return None
    return {
        "corners_px": [point_payload(point) for point in calibration.corners_px],
        "homography": calibration.homography,
        "confidence": calibration.confidence,
        "source": calibration.source,
    }


def point_payload(point: Point2D) -> dict[str, float]:
    return {"x": point.x, "y": point.y}


if __name__ == "__main__":
    raise SystemExit(main())
