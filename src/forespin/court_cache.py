from __future__ import annotations

import json
from pathlib import Path

from forespin.domain import CourtCalibration, Point2D, VideoMetadata

CACHE_VERSION = 1


def court_calibration_cache_path(
    *,
    output_dir: Path,
    video_path: Path,
    remove_net: bool,
) -> Path:
    mode = "net_removed" if remove_net else "raw"
    return output_dir / video_path.stem / "cache" / "court_calibration" / f"{video_path.name}.{mode}.json"


def read_cached_court_calibration(
    path: Path,
    *,
    video_path: Path,
    metadata: VideoMetadata,
    remove_net: bool,
) -> CourtCalibration | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("video_filename") != video_path.name:
        return None
    if payload.get("remove_net_for_court_calibration") != remove_net:
        return None
    video_meta = payload.get("video_metadata") or {}
    if int(video_meta.get("width") or 0) != metadata.width:
        return None
    if int(video_meta.get("height") or 0) != metadata.height:
        return None
    if int(video_meta.get("frame_count") or 0) != metadata.frame_count:
        return None

    calibration = payload.get("calibration") or {}
    corners = calibration.get("corners_px") or []
    homography = calibration.get("homography")
    if len(corners) != 4 or not homography:
        return None
    return CourtCalibration(
        corners_px=[Point2D(float(point["x"]), float(point["y"])) for point in corners],
        homography=homography,
        confidence=float(calibration.get("confidence") or 0.0),
        source=str(calibration.get("source") or "cached_court"),
    )


def write_cached_court_calibration(
    path: Path,
    *,
    video_path: Path,
    metadata: VideoMetadata,
    remove_net: bool,
    calibration: CourtCalibration,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "video_filename": video_path.name,
        "remove_net_for_court_calibration": remove_net,
        "video_metadata": {
            "width": metadata.width,
            "height": metadata.height,
            "frame_count": metadata.frame_count,
            "fps": metadata.fps,
        },
        "calibration": {
            "corners_px": [{"x": point.x, "y": point.y} for point in calibration.corners_px],
            "homography": calibration.homography,
            "confidence": calibration.confidence,
            "source": calibration.source,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
