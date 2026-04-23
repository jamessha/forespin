from __future__ import annotations

from pathlib import Path

from forespin.config import AnalysisOptions
from forespin.court import CourtCalibrator
from forespin.deps import require_vision_stack
from forespin.domain import FrameObservation, InputConfig, Point2D, VideoMetadata
from forespin.geometry import average, scale, transform_point
from forespin.model_weights import resolve_model_weights
from forespin.tracking.ball import create_ball_tracker
from forespin.tracking.player import create_player_tracker


def build_observations(
    input_config: InputConfig,
    options: AnalysisOptions,
) -> tuple[VideoMetadata, list[FrameObservation], object]:
    resolved_weights = resolve_model_weights(input_config)
    cv2, _ = require_vision_stack()
    video_path = Path(input_config.video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    metadata = VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=(frame_count / fps) if fps else 0.0,
    )

    calibrator = CourtCalibrator(options.thresholds, weights_path=resolved_weights.court_weights)
    analysis_output_dir = Path(input_config.output_dir or "outputs") / video_path.stem
    court_debug_dir = analysis_output_dir / "debug" / "court_calibration"
    court = calibrator.calibrate_video(video_path, debug_dir=court_debug_dir)
    ball_tracker = create_ball_tracker(resolved_weights.tracknet_weights)
    player_tracker = create_player_tracker(input_config.tracked_player_side, resolved_weights.player_pose_weights)

    observations: list[FrameObservation] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            court = calibrator.maybe_refresh(frame_index, frame, court)
            ball_track = ball_tracker.track(frame)
            player_track = player_tracker.track(frame)
            observations.append(
                FrameObservation(
                    frame_index=frame_index,
                    timestamp_s=frame_index / fps if fps else 0.0,
                    ball_px=ball_track.position_px,
                    ball_court=transform_point(ball_track.position_px, court.homography) if ball_track.position_px else None,
                    ball_confidence=ball_track.confidence,
                    tracked_player_bbox_px=player_track.bbox_px,
                    tracked_player_feet_px=player_track.feet_px,
                    tracked_player_feet_court=transform_point(player_track.feet_px, court.homography) if player_track.feet_px else None,
                    tracked_player_torso_px=player_track.torso_px,
                    tracked_player_torso_court=transform_point(player_track.torso_px, court.homography) if player_track.torso_px else None,
                    tracked_player_confidence=player_track.confidence,
                    court_confidence=court.confidence,
                )
            )
            frame_index += 1
    finally:
        capture.release()

    _interpolate_ball_track(observations, options.thresholds.interpolate_ball_gaps_up_to_frames)
    _smooth_ball_track(observations, options.thresholds.smoothing_window)
    _compute_ball_velocity(observations, fps)
    return metadata, observations, court


def _interpolate_ball_track(observations: list[FrameObservation], max_gap: int) -> None:
    index = 0
    while index < len(observations):
        if observations[index].ball_px is not None:
            index += 1
            continue
        gap_start = index - 1
        gap_end = index
        while gap_end < len(observations) and observations[gap_end].ball_px is None:
            gap_end += 1
        gap_length = gap_end - index
        if gap_start >= 0 and gap_end < len(observations) and 0 < gap_length <= max_gap:
            left_obs = observations[gap_start]
            right_obs = observations[gap_end]
            if left_obs.ball_px is not None and right_obs.ball_px is not None:
                for fill_index in range(index, gap_end):
                    ratio = (fill_index - gap_start) / (gap_end - gap_start)
                    observations[fill_index].ball_px = _lerp_point(left_obs.ball_px, right_obs.ball_px, ratio)
                    observations[fill_index].ball_court = _lerp_optional_point(left_obs.ball_court, right_obs.ball_court, ratio)
                    observations[fill_index].ball_confidence = min(left_obs.ball_confidence, right_obs.ball_confidence) * 0.75
        index = gap_end


def _smooth_ball_track(observations: list[FrameObservation], window_size: int) -> None:
    if window_size <= 1:
        return
    radius = window_size // 2
    original_ball = [observation.ball_px for observation in observations]
    original_ball_court = [observation.ball_court for observation in observations]
    for index, observation in enumerate(observations):
        px_window = [
            point
            for point in original_ball[max(0, index - radius) : min(len(original_ball), index + radius + 1)]
            if point is not None
        ]
        court_window = [
            point
            for point in original_ball_court[max(0, index - radius) : min(len(original_ball_court), index + radius + 1)]
            if point is not None
        ]
        averaged_px = average(px_window)
        averaged_court = average(court_window)
        if averaged_px is not None:
            observation.ball_px = averaged_px
        if averaged_court is not None:
            observation.ball_court = averaged_court


def _compute_ball_velocity(observations: list[FrameObservation], fps: float) -> None:
    if fps <= 0:
        return
    for index in range(1, len(observations)):
        current = observations[index]
        previous = observations[index - 1]
        if current.ball_px is None or previous.ball_px is None:
            continue
        delta = Point2D(current.ball_px.x - previous.ball_px.x, current.ball_px.y - previous.ball_px.y)
        current.ball_velocity_px_s = scale(delta, fps)


def _lerp_optional_point(left: Point2D | None, right: Point2D | None, ratio: float) -> Point2D | None:
    if left is None or right is None:
        return None
    return _lerp_point(left, right, ratio)


def _lerp_point(left: Point2D, right: Point2D, ratio: float) -> Point2D:
    return Point2D(
        left.x + ((right.x - left.x) * ratio),
        left.y + ((right.y - left.y) * ratio),
    )
