from __future__ import annotations

import argparse
import json
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
    cv2, _ = require_vision_stack()

    result = infer_image(input_path=input_path, output_dir=output_dir, backend=backend, cv2=cv2)
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
) -> dict[str, Any]:
    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Unable to read image: {input_path}")
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
    return {
        "frame_index": frame_index,
        "source": source,
        "accepted": evaluation["calibration"] is not None,
        "raw_image": str(output_dir / f"frame_{frame_index:03d}_raw.jpg"),
        "overlay_image": str(output_dir / f"frame_{frame_index:03d}_overlay.jpg"),
        "summary": evaluation["summary"],
        "calibration": calibration_payload(evaluation["calibration"]),
        "reference_to_image": evaluation["reference_to_image"],
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
