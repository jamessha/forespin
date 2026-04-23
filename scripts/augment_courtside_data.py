from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.court_reference import CourtReference

SINGLES_WIDTH_FT = 27.0
COURT_LENGTH_FT = 78.0
DEFAULT_SOURCE_ROOT = ROOT / "calib_model_data" / "tennis_court_detector"
DEFAULT_OUTPUT_ROOT = ROOT / "calib_model_data" / "courtside_data"
DEFAULT_SOURCE_HEIGHT_FT = 30.0
DEFAULT_TARGET_HEIGHT_RANGE_FT = (4.0, 6.0)
DEFAULT_SOURCE_BASELINE_DISTANCE_FT = 21.0
DEFAULT_TARGET_BASELINE_DISTANCE_FT = DEFAULT_SOURCE_BASELINE_DISTANCE_FT - 6.0
DEFAULT_SOURCE_DOWNWARD_TILT = 0.20
DEFAULT_TARGET_DOWNWARD_ANGLE_RANGE_DEG = (8.0, 10.0)
DEFAULT_TARGET_Y_JITTER_FT = 1.0
DEFAULT_VALIDATION_FRACTION = 0.25


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv2, np = require_cv2_numpy()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = resolve_output_root(args)
    image_output_dir = output_root / "images"
    rng = random.Random(args.seed)

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output_root} already exists and is not empty. Re-run with --overwrite to replace generated files.")
    image_output_dir.mkdir(parents=True, exist_ok=True)

    reference = CourtReference()
    world_points = reference_world_points(reference, np)
    records = load_source_records(
        source_root=source_root,
        split_files=[args.test_split] if args.test else args.splits,
        test=args.test,
        test_id=args.test_id,
        max_items=args.max_items,
    )
    report: dict[str, Any] = {
        "mode": "test" if args.test else "full",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_splits": [args.test_split] if args.test else args.splits,
        "validation_fraction": args.validation_fraction,
        "source_height_ft": args.source_height_ft,
        "target_height_ft": args.target_height_ft,
        "target_height_range_ft": args.target_height_range_ft,
        "source_baseline_distance_ft": args.source_baseline_distance_ft,
        "target_baseline_distance_ft": args.target_baseline_distance_ft,
        "target_y_jitter_ft": args.target_y_jitter_ft,
        "source_downward_tilt": args.source_downward_tilt,
        "target_downward_tilt": args.target_downward_tilt,
        "target_downward_angle_deg": args.target_downward_angle_deg,
        "target_downward_angle_range_deg": args.target_downward_angle_range_deg,
        "focal_scale": args.focal_scale,
        "orientation": args.orientation,
        "seed": args.seed,
        "source_records": len(records),
        "written_records": 0,
        "skipped_records": 0,
        "visible_keypoint_counts": {},
        "skips": [],
        "camera_params": {},
        "splits": {},
    }

    transformed_records: list[dict[str, Any]] = []
    for record in records:
        source_image_path = source_root / "images" / f"{record['id']}.png"
        image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
        if image is None:
            report["skipped_records"] += 1
            report["skips"].append({"id": record["id"], "reason": "image_not_readable"})
            continue
        camera_params = sample_target_camera_params(args, rng)

        try:
            transformed = transform_record(
                record=record,
                image=image,
                world_points=world_points,
                cv2=cv2,
                np=np,
                focal_scale=args.focal_scale,
                source_height_ft=args.source_height_ft,
                target_height_ft=camera_params["target_height_ft"],
                source_baseline_distance_ft=args.source_baseline_distance_ft,
                target_baseline_distance_ft=camera_params["target_baseline_distance_ft"],
                source_downward_tilt=args.source_downward_tilt,
                target_downward_tilt=camera_params["target_downward_tilt"],
                orientation=args.orientation,
                suffix=args.output_suffix,
                round_labels=not args.keep_float_labels,
                border_mode=args.border_mode,
            )
        except ValueError as exc:
            report["skipped_records"] += 1
            report["skips"].append({"id": record["id"], "reason": str(exc)})
            continue

        output_image_path = image_output_dir / f"{transformed['record']['id']}.png"
        if output_image_path.exists() and not args.overwrite:
            raise SystemExit(f"{output_image_path} already exists. Re-run with --overwrite to replace generated files.")
        cv2.imwrite(str(output_image_path), transformed["image"])
        transformed_records.append(transformed["record"])
        report["written_records"] += 1
        report["camera_params"][transformed["record"]["id"]] = camera_params
        visible_count = str(transformed["visible_keypoints"])
        report["visible_keypoint_counts"][visible_count] = report["visible_keypoint_counts"].get(visible_count, 0) + 1

    train_records, validation_records = split_train_validation(
        records=transformed_records,
        validation_fraction=args.validation_fraction,
        rng=rng,
        test=args.test,
    )
    write_split(output_root / "data_train.json", train_records, overwrite=args.overwrite)
    write_split(output_root / "data_val.json", validation_records, overwrite=args.overwrite)
    report["splits"] = {
        "data_train.json": {"written_records": len(train_records)},
        "data_val.json": {"written_records": len(validation_records)},
    }

    (output_root / "augmentation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warp TennisCourtDetector calibration data from a high professional baseline view toward a lower courtside camera height."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="Source TennisCourtDetector dataset root.")
    parser.add_argument(
        "--output-root",
        help=f"Destination dataset root. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument("--splits", nargs="+", default=["data_train.json", "data_val.json"], help="JSON split files to transform.")
    parser.add_argument("--test", action="store_true", help="Transform one image into the output dataset for quick validation.")
    parser.add_argument("--test-split", default="data_train.json", help="Split file to sample from in --test mode.")
    parser.add_argument("--test-id", help="Specific record id to transform in --test mode. Defaults to the first record in --test-split.")
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION, help="Fraction of generated records to write to data_val.json.")
    parser.add_argument("--source-height-ft", type=float, default=DEFAULT_SOURCE_HEIGHT_FT, help="Approximate source camera height.")
    parser.add_argument("--target-height-ft", type=float, help="Fixed target camera height. If omitted, each image samples from --target-height-range-ft.")
    parser.add_argument(
        "--target-height-range-ft",
        nargs=2,
        type=float,
        default=list(DEFAULT_TARGET_HEIGHT_RANGE_FT),
        metavar=("MIN", "MAX"),
        help="Per-image target camera height range in feet.",
    )
    parser.add_argument(
        "--source-baseline-distance-ft",
        type=float,
        default=DEFAULT_SOURCE_BASELINE_DISTANCE_FT,
        help="Approximate source camera distance behind the near baseline.",
    )
    parser.add_argument(
        "--target-baseline-distance-ft",
        type=float,
        default=DEFAULT_TARGET_BASELINE_DISTANCE_FT,
        help="Base target camera distance behind the near baseline before per-image Y jitter.",
    )
    parser.add_argument(
        "--source-downward-tilt",
        type=float,
        default=DEFAULT_SOURCE_DOWNWARD_TILT,
        help="Approximate source downward camera tilt as a grade. 0.20 means 20 percent.",
    )
    parser.add_argument(
        "--target-downward-tilt",
        type=float,
        help="Fixed target downward camera tilt as a grade. If omitted, each image samples from --target-downward-angle-range-deg.",
    )
    parser.add_argument(
        "--target-downward-angle-deg",
        type=float,
        help="Fixed target downward camera angle in degrees. Takes precedence over --target-downward-tilt.",
    )
    parser.add_argument(
        "--target-downward-angle-range-deg",
        nargs=2,
        type=float,
        default=list(DEFAULT_TARGET_DOWNWARD_ANGLE_RANGE_DEG),
        metavar=("MIN", "MAX"),
        help="Per-image target downward camera angle range in degrees.",
    )
    parser.add_argument(
        "--focal-scale",
        type=float,
        default=1.35,
        help="Assumed focal length as a multiple of max(image_width, image_height) for homography decomposition.",
    )
    parser.add_argument(
        "--orientation",
        choices=["target_tilt", "look_at_original", "look_at_court_center", "same_rotation"],
        default="target_tilt",
        help="How to orient the lower virtual camera after preserving the estimated X/Y camera position.",
    )
    parser.add_argument("--output-suffix", default="_court6ft", help="Suffix appended to generated image ids.")
    parser.add_argument("--border-mode", choices=["constant", "replicate", "reflect"], default="constant", help="Border mode for newly exposed pixels.")
    parser.add_argument("--keep-float-labels", action="store_true", help="Keep warped labels as floats instead of rounded integers.")
    parser.add_argument("--target-y-jitter-ft", type=float, default=DEFAULT_TARGET_Y_JITTER_FT, help="Per-image random Y-axis jitter in feet, sampled from +/- this value.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible camera sampling.")
    parser.add_argument("--max-items", type=int, help="Limit transformed records per split for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files in the output directory.")
    return parser


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root).expanduser().resolve()
    return DEFAULT_OUTPUT_ROOT.resolve()


def load_source_records(
    *,
    source_root: Path,
    split_files: list[str],
    test: bool,
    test_id: str | None,
    max_items: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for split_file in split_files:
        split_records = json.loads((source_root / split_file).read_text())
        for record in split_records:
            if record["id"] in seen_ids:
                continue
            seen_ids.add(record["id"])
            records.append(record)

    if test:
        if not records:
            raise SystemExit("No records found in the selected test split.")
        if test_id:
            for record in records:
                if record["id"] == test_id:
                    return [record]
            raise SystemExit(f"Unable to find --test-id {test_id!r} in the selected split.")
        return [records[0]]
    if max_items is None:
        return records
    return records[:max_items]


def split_train_validation(
    *,
    records: list[dict[str, Any]],
    validation_fraction: float,
    rng: random.Random,
    test: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be >= 0 and < 1.")
    shuffled = list(records)
    rng.shuffle(shuffled)
    if test or len(shuffled) <= 1:
        return shuffled, []
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def write_split(path: Path, records: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} already exists. Re-run with --overwrite to replace generated files.")
    path.write_text(json.dumps(records, separators=(",", ":")))


def sample_target_camera_params(args: argparse.Namespace, rng: random.Random) -> dict[str, float]:
    target_height_ft = args.target_height_ft
    if target_height_ft is None:
        target_height_ft = sample_range(args.target_height_range_ft, rng, "target_height_range_ft")

    target_angle_deg = args.target_downward_angle_deg
    target_downward_tilt = args.target_downward_tilt
    if target_angle_deg is None and target_downward_tilt is None:
        target_angle_deg = sample_range(args.target_downward_angle_range_deg, rng, "target_downward_angle_range_deg")
        target_downward_tilt = math.tan(math.radians(target_angle_deg))
    elif target_angle_deg is not None:
        target_downward_tilt = math.tan(math.radians(target_angle_deg))
    else:
        target_angle_deg = math.degrees(math.atan(target_downward_tilt))

    if args.target_y_jitter_ft < 0.0:
        raise SystemExit("--target-y-jitter-ft must be non-negative.")
    target_baseline_distance_ft = args.target_baseline_distance_ft + rng.uniform(-args.target_y_jitter_ft, args.target_y_jitter_ft)

    if target_height_ft <= 0.0:
        raise SystemExit("Sampled target height must be positive.")
    if target_downward_tilt <= 0.0:
        raise SystemExit("Sampled target downward tilt must be positive.")
    if target_baseline_distance_ft <= 0.0:
        raise SystemExit("Sampled target baseline distance must be positive.")

    return {
        "target_height_ft": target_height_ft,
        "target_baseline_distance_ft": target_baseline_distance_ft,
        "target_y_shift_ft": target_baseline_distance_ft - args.target_baseline_distance_ft,
        "target_downward_angle_deg": target_angle_deg,
        "target_downward_tilt": target_downward_tilt,
    }


def sample_range(values: list[float], rng: random.Random, label: str) -> float:
    if len(values) != 2:
        raise SystemExit(f"--{label.replace('_', '-')} must contain exactly two values.")
    low, high = values
    if low <= 0.0 or high <= 0.0:
        raise SystemExit(f"--{label.replace('_', '-')} values must be positive.")
    if low > high:
        raise SystemExit(f"--{label.replace('_', '-')} minimum must be <= maximum.")
    return rng.uniform(low, high)


def require_cv2_numpy() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script requires OpenCV and NumPy. Install the vision extra with: "
            "python3 -m pip install -e '.[vision]'"
        ) from exc
    return cv2, np


def reference_world_points(reference: CourtReference, np: Any) -> Any:
    points = []
    for reference_point in reference.key_points:
        normalized = reference.normalized_point(reference_point)
        points.append(
            [
                (normalized.x - 0.5) * SINGLES_WIDTH_FT,
                (normalized.y - 0.5) * COURT_LENGTH_FT,
            ]
        )
    return np.array(points, dtype=np.float64)


def move_camera_toward_near_baseline(
    *,
    y: float,
    source_baseline_distance_ft: float,
    target_baseline_distance_ft: float,
) -> float:
    near_baseline_y = COURT_LENGTH_FT / 2.0
    far_baseline_y = -COURT_LENGTH_FT / 2.0
    near_distance = abs(y - near_baseline_y)
    far_distance = abs(y - far_baseline_y)
    camera_behind_near_baseline = near_distance <= far_distance
    if camera_behind_near_baseline:
        direction = 1.0
        baseline_y = near_baseline_y
    else:
        direction = -1.0
        baseline_y = far_baseline_y
    scaled_distance = abs(y - baseline_y) * (target_baseline_distance_ft / source_baseline_distance_ft)
    return baseline_y + (direction * scaled_distance)


def transform_record(
    *,
    record: dict[str, Any],
    image: Any,
    world_points: Any,
    cv2: Any,
    np: Any,
    focal_scale: float,
    source_height_ft: float,
    target_height_ft: float,
    source_baseline_distance_ft: float,
    target_baseline_distance_ft: float,
    source_downward_tilt: float,
    target_downward_tilt: float,
    orientation: str,
    suffix: str,
    round_labels: bool,
    border_mode: str,
) -> dict[str, Any]:
    if source_height_ft <= 0.0 or target_height_ft <= 0.0:
        raise ValueError("camera_heights_must_be_positive")
    if source_baseline_distance_ft <= 0.0 or target_baseline_distance_ft <= 0.0:
        raise ValueError("baseline_distances_must_be_positive")
    if source_downward_tilt <= 0.0 or target_downward_tilt <= 0.0:
        raise ValueError("downward_tilts_must_be_positive")
    image_height, image_width = image.shape[:2]
    labels = np.array(record["kps"], dtype=np.float64)
    if labels.shape != (14, 2):
        raise ValueError("expected_14_keypoints")

    source_homography, _ = cv2.findHomography(world_points.astype(np.float32), labels.astype(np.float32), method=0)
    if source_homography is None:
        raise ValueError("source_homography_failed")

    intrinsics = camera_intrinsics(image_width=image_width, image_height=image_height, focal_scale=focal_scale, np=np)
    rotation, camera_center = decompose_planar_homography(source_homography, intrinsics, np)
    if camera_center[2] <= 0.0 or not np.isfinite(camera_center).all():
        raise ValueError("camera_decomposition_failed")

    lower_camera_center = camera_center.copy()
    lower_camera_center[2] = camera_center[2] * (target_height_ft / source_height_ft)
    lower_camera_center[1] = move_camera_toward_near_baseline(
        y=camera_center[1],
        source_baseline_distance_ft=source_baseline_distance_ft,
        target_baseline_distance_ft=target_baseline_distance_ft,
    )

    source_target = estimate_look_at_point(rotation, camera_center, np)
    if orientation == "same_rotation":
        target_rotation = rotation
    elif orientation == "target_tilt":
        target = target_from_downward_tilt(
            source_camera_center=camera_center,
            target_camera_center=lower_camera_center,
            source_look_at=source_target,
            source_downward_tilt=source_downward_tilt,
            target_downward_tilt=target_downward_tilt,
            np=np,
        )
        target_rotation = look_at_rotation(lower_camera_center, target, np)
    else:
        target = source_target
        if orientation == "look_at_court_center" or target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        target_rotation = look_at_rotation(lower_camera_center, target, np)

    target_translation = -target_rotation @ lower_camera_center
    target_homography = intrinsics @ np.column_stack(
        [target_rotation[:, 0], target_rotation[:, 1], target_translation]
    )
    warp = target_homography @ np.linalg.inv(source_homography)
    warp = normalize_homography(warp)
    warped_image = cv2.warpPerspective(
        image,
        warp,
        (image_width, image_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv_border_mode(border_mode, cv2),
        borderValue=(0, 0, 0),
    )
    warped_labels = project_points(labels, warp, np)
    output_labels = labels_to_json(warped_labels, round_labels=round_labels)
    visible_keypoints = count_visible_keypoints(warped_labels, image_width=image_width, image_height=image_height, np=np)

    output_record = dict(record)
    output_record["id"] = f"{record['id']}{suffix}"
    output_record["kps"] = output_labels
    return {
        "record": output_record,
        "image": warped_image,
        "visible_keypoints": visible_keypoints,
    }


def camera_intrinsics(*, image_width: int, image_height: int, focal_scale: float, np: Any) -> Any:
    focal_length = focal_scale * max(image_width, image_height)
    return np.array(
        [
            [focal_length, 0.0, image_width / 2.0],
            [0.0, focal_length, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def decompose_planar_homography(homography: Any, intrinsics: Any, np: Any) -> tuple[Any, Any]:
    inv_intrinsics = np.linalg.inv(intrinsics)
    best: tuple[Any, Any] | None = None
    for sign in (1.0, -1.0):
        normalized = inv_intrinsics @ (homography * sign)
        scale = 2.0 / (np.linalg.norm(normalized[:, 0]) + np.linalg.norm(normalized[:, 1]))
        r1 = normalized[:, 0] * scale
        r2 = normalized[:, 1] * scale
        translation = normalized[:, 2] * scale
        r3 = np.cross(r1, r2)
        rotation_estimate = np.column_stack([r1, r2, r3])
        u, _, vt = np.linalg.svd(rotation_estimate)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            rotation *= -1.0
            translation *= -1.0
        camera_center = -rotation.T @ translation
        if np.isfinite(camera_center).all() and camera_center[2] > 0.0:
            best = (rotation, camera_center)
            break
    if best is None:
        raise ValueError("camera_decomposition_failed")
    return best


def estimate_look_at_point(rotation: Any, camera_center: Any, np: Any) -> Any | None:
    forward_world = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(forward_world[2]) < 1e-8:
        return None
    distance_to_ground = -camera_center[2] / forward_world[2]
    if distance_to_ground <= 0.0 or not math.isfinite(float(distance_to_ground)):
        return None
    return camera_center + (forward_world * distance_to_ground)


def target_from_downward_tilt(
    *,
    source_camera_center: Any,
    target_camera_center: Any,
    source_look_at: Any | None,
    source_downward_tilt: float,
    target_downward_tilt: float,
    np: Any,
) -> Any:
    if source_look_at is None:
        source_look_at = fallback_source_look_at(source_camera_center, source_downward_tilt, np)
    direction_xy = source_look_at[:2] - source_camera_center[:2]
    magnitude = np.linalg.norm(direction_xy)
    if magnitude < 1e-8:
        direction_xy = np.array([0.0, -1.0], dtype=np.float64)
    else:
        direction_xy = direction_xy / magnitude
    horizontal_distance = target_camera_center[2] / target_downward_tilt
    target_xy = target_camera_center[:2] + (direction_xy * horizontal_distance)
    return np.array([target_xy[0], target_xy[1], 0.0], dtype=np.float64)


def fallback_source_look_at(camera_center: Any, source_downward_tilt: float, np: Any) -> Any:
    direction_xy = np.array([0.0, 0.0], dtype=np.float64) - camera_center[:2]
    magnitude = np.linalg.norm(direction_xy)
    if magnitude < 1e-8:
        direction_xy = np.array([0.0, -1.0], dtype=np.float64)
    else:
        direction_xy = direction_xy / magnitude
    horizontal_distance = camera_center[2] / source_downward_tilt
    target_xy = camera_center[:2] + (direction_xy * horizontal_distance)
    return np.array([target_xy[0], target_xy[1], 0.0], dtype=np.float64)


def look_at_rotation(camera_center: Any, target: Any, np: Any) -> Any:
    forward = normalize_vector(target - camera_center, np)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = normalize_vector(right, np)
    down = normalize_vector(np.cross(forward, right), np)
    return np.vstack([right, down, forward])


def normalize_vector(vector: Any, np: Any) -> Any:
    magnitude = np.linalg.norm(vector)
    if magnitude < 1e-8:
        raise ValueError("zero_length_vector")
    return vector / magnitude


def normalize_homography(homography: Any) -> Any:
    denominator = homography[2, 2]
    if abs(float(denominator)) < 1e-12:
        return homography
    return homography / denominator


def project_points(points: Any, homography: Any, np: Any) -> Any:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = (homography @ homogeneous.T).T
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    return projected[:, :2]


def labels_to_json(points: Any, *, round_labels: bool) -> list[list[float | int]]:
    labels: list[list[float | int]] = []
    for x, y in points.tolist():
        if round_labels:
            labels.append([int(round(x)), int(round(y))])
        else:
            labels.append([float(x), float(y)])
    return labels


def count_visible_keypoints(points: Any, *, image_width: int, image_height: int, np: Any) -> int:
    finite = np.isfinite(points).all(axis=1)
    in_frame = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < image_width)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < image_height)
    )
    return int((finite & in_frame).sum())


def cv_border_mode(name: str, cv2: Any) -> int:
    if name == "constant":
        return cv2.BORDER_CONSTANT
    if name == "reflect":
        return cv2.BORDER_REFLECT_101
    return cv2.BORDER_REPLICATE


if __name__ == "__main__":
    raise SystemExit(main())
