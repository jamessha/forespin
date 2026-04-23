from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "calib_model_data" / "courtside_data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "label_overlays"

COURT_SEGMENTS = [
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv2, np = require_cv2_numpy()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = set(args.ids or [])
    summary: dict[str, Any] = {"data_root": str(data_root), "output_dir": str(output_dir), "splits": {}}
    for split_file in args.splits:
        records = json.loads((data_root / split_file).read_text())
        split_output_dir = output_dir / Path(split_file).stem
        split_output_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        missing = 0

        for record in records:
            if selected_ids and record["id"] not in selected_ids:
                continue
            if args.max_images is not None and written >= args.max_images:
                break
            image_path = data_root / "images" / f"{record['id']}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                missing += 1
                continue
            overlay = draw_overlay(image=image, keypoints=np.array(record["kps"], dtype=np.float64), cv2=cv2, np=np)
            output_path = split_output_dir / f"{record['id']}_labels.png"
            cv2.imwrite(str(output_path), overlay)
            written += 1

        summary["splits"][split_file] = {"written": written, "missing_images": missing}

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render TennisCourtDetector labels over dataset images.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Dataset root containing images/ and split JSON files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for rendered overlays.")
    parser.add_argument("--splits", nargs="+", default=["data_train.json", "data_val.json"], help="Split JSON files to visualize.")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum overlays to render per split.")
    parser.add_argument("--ids", nargs="*", help="Optional explicit record ids to render.")
    return parser


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


def draw_overlay(*, image: Any, keypoints: Any, cv2: Any, np: Any) -> Any:
    overlay = image.copy()
    height, width = overlay.shape[:2]
    visible = [point_in_frame(point, width=width, height=height, np=np) for point in keypoints]

    for start, end in COURT_SEGMENTS:
        start_point = keypoints[start]
        end_point = keypoints[end]
        if visible[start] and visible[end]:
            cv2.line(
                overlay,
                (int(round(start_point[0])), int(round(start_point[1]))),
                (int(round(end_point[0])), int(round(end_point[1]))),
                (0, 220, 255),
                2,
                lineType=cv2.LINE_AA,
            )

    for index, point in enumerate(keypoints):
        color = (0, 255, 0) if visible[index] else (0, 0, 255)
        x, y = clipped_point(point, width=width, height=height)
        cv2.circle(overlay, (x, y), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 9, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(index),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    visible_count = sum(visible)
    cv2.putText(
        overlay,
        f"visible keypoints: {visible_count}/14",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return overlay


def point_in_frame(point: Any, *, width: int, height: int, np: Any) -> bool:
    return bool(
        np.isfinite(point).all()
        and 0.0 <= point[0] < width
        and 0.0 <= point[1] < height
    )


def clipped_point(point: Any, *, width: int, height: int) -> tuple[int, int]:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


if __name__ == "__main__":
    raise SystemExit(main())
