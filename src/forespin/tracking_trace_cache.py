from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import Any

from forespin.config import AnalysisOptions
from forespin.domain import BBox, CourtCalibration, FrameObservation, InputConfig, Point2D, VideoMetadata
from forespin.model_weights import ResolvedModelWeights
from forespin.serialization import to_jsonable

TRACE_CACHE_SCHEMA_VERSION = 1


def tracking_trace_cache_path(*, output_dir: Path, video_path: Path) -> Path:
    return output_dir / video_path.stem / "cache" / "tracking_trace" / f"{video_path.name}.json"


def read_cached_tracking_trace(
    path: Path,
    *,
    video_path: Path,
    input_config: InputConfig,
    metadata: VideoMetadata,
    resolved_weights: ResolvedModelWeights,
    options: AnalysisOptions,
) -> tuple[VideoMetadata, list[FrameObservation], CourtCalibration] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    expected_key = _tracking_trace_cache_key(
        video_path=video_path,
        input_config=input_config,
        metadata=metadata,
        resolved_weights=resolved_weights,
        options=options,
    )
    if payload.get("schema_version") != TRACE_CACHE_SCHEMA_VERSION or payload.get("cache_key") != expected_key:
        return None

    try:
        cached_metadata = _metadata_from_json(payload["metadata"])
        observations = [_observation_from_json(item) for item in payload["observations"]]
        court = _court_from_json(payload["court"])
    except (KeyError, TypeError, ValueError):
        return None
    court.source = f"{court.source}:trace_cache"
    return cached_metadata, observations, court


def write_cached_tracking_trace(
    path: Path,
    *,
    video_path: Path,
    input_config: InputConfig,
    metadata: VideoMetadata,
    resolved_weights: ResolvedModelWeights,
    options: AnalysisOptions,
    observations: list[FrameObservation],
    court: CourtCalibration,
) -> Path:
    payload = {
        "schema_version": TRACE_CACHE_SCHEMA_VERSION,
        "cache_key": _tracking_trace_cache_key(
            video_path=video_path,
            input_config=input_config,
            metadata=metadata,
            resolved_weights=resolved_weights,
            options=options,
        ),
        "metadata": to_jsonable(metadata),
        "court": to_jsonable(court),
        "observations": to_jsonable(observations),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _tracking_trace_cache_key(
    *,
    video_path: Path,
    input_config: InputConfig,
    metadata: VideoMetadata,
    resolved_weights: ResolvedModelWeights,
    options: AnalysisOptions,
) -> dict[str, Any]:
    thresholds = options.thresholds
    return {
        "video": _file_signature(video_path),
        "metadata": to_jsonable(metadata),
        "tracked_player_side": input_config.tracked_player_side.value,
        "handedness": input_config.handedness.value,
        "tracknet_weights": _file_signature(Path(resolved_weights.tracknet_weights)),
        "player_pose_weights": _file_signature(Path(resolved_weights.player_pose_weights)),
        "court_weights": _file_signature(Path(resolved_weights.court_weights)),
        "remove_net_for_court_calibration": options.remove_net_for_court_calibration,
        "net_removal_model": options.net_removal_model,
        "tracking_thresholds": {
            name: getattr(thresholds, name)
            for name in (
                "min_ball_confidence",
                "min_player_confidence",
                "smoothing_window",
                "interpolate_ball_gaps_up_to_frames",
                "ball_track_segment_gap_frames",
                "ball_outlier_max_speed_px_s",
                "ball_outlier_max_acceleration_px_s2",
                "ball_outlier_run_max_frames",
                "ball_static_segment_min_frames",
                "ball_static_segment_max_displacement_px",
                "ball_out_of_play_court_margin",
                "ball_out_of_play_confirm_frames",
            )
        },
    }


def _file_signature(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    try:
        resolved = expanded.resolve()
    except OSError:
        resolved = expanded
    try:
        stat = resolved.stat()
    except OSError:
        return {"path": str(resolved), "exists": False}
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _metadata_from_json(payload: dict[str, Any]) -> VideoMetadata:
    return VideoMetadata(**{field.name: payload[field.name] for field in fields(VideoMetadata)})


def _court_from_json(payload: dict[str, Any]) -> CourtCalibration:
    return CourtCalibration(
        corners_px=[_point_from_json(point) for point in payload["corners_px"]],
        homography=payload["homography"],
        confidence=payload["confidence"],
        source=payload.get("source", "cached_trace"),
    )


def _observation_from_json(payload: dict[str, Any]) -> FrameObservation:
    return FrameObservation(
        frame_index=payload["frame_index"],
        timestamp_s=payload["timestamp_s"],
        ball_px=_optional_point_from_json(payload.get("ball_px")),
        ball_court=_optional_point_from_json(payload.get("ball_court")),
        ball_velocity_px_s=_optional_point_from_json(payload.get("ball_velocity_px_s")),
        ball_confidence=payload.get("ball_confidence", 0.0),
        tracked_player_bbox_px=_optional_bbox_from_json(payload.get("tracked_player_bbox_px")),
        tracked_player_feet_px=_optional_point_from_json(payload.get("tracked_player_feet_px")),
        tracked_player_feet_court=_optional_point_from_json(payload.get("tracked_player_feet_court")),
        tracked_player_torso_px=_optional_point_from_json(payload.get("tracked_player_torso_px")),
        tracked_player_torso_court=_optional_point_from_json(payload.get("tracked_player_torso_court")),
        tracked_player_confidence=payload.get("tracked_player_confidence", 0.0),
        court_confidence=payload.get("court_confidence", 0.0),
    )


def _optional_point_from_json(payload: dict[str, Any] | None) -> Point2D | None:
    if payload is None:
        return None
    return _point_from_json(payload)


def _point_from_json(payload: dict[str, Any]) -> Point2D:
    return Point2D(x=payload["x"], y=payload["y"])


def _optional_bbox_from_json(payload: dict[str, Any] | None) -> BBox | None:
    if payload is None:
        return None
    return BBox(x1=payload["x1"], y1=payload["y1"], x2=payload["x2"], y2=payload["y2"])
