from __future__ import annotations

import math
from pathlib import Path

from forespin.config import AnalysisOptions, Thresholds
from forespin.court import CourtCalibrator
from forespin.court_cache import court_calibration_cache_path, read_cached_court_calibration, write_cached_court_calibration
from forespin.deps import require_vision_stack
from forespin.domain import FrameObservation, InputConfig, Point2D, VideoMetadata
from forespin.geometry import average, scale, transform_point
from forespin.model_weights import resolve_model_weights
from forespin.net_removal import OpenAINetRemovalPreprocessor
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

    frame_preprocessor = (
        OpenAINetRemovalPreprocessor(model=options.net_removal_model)
        if options.remove_net_for_court_calibration
        else None
    )
    calibrator = CourtCalibrator(
        options.thresholds,
        weights_path=resolved_weights.court_weights,
        frame_preprocessor=frame_preprocessor,
    )
    analysis_output_dir = Path(input_config.output_dir or "outputs") / video_path.stem
    court_debug_dir = analysis_output_dir / "debug" / "court_calibration"
    cache_path = court_calibration_cache_path(
        output_dir=Path(input_config.output_dir or "outputs"),
        video_path=video_path,
        remove_net=options.remove_net_for_court_calibration,
    )
    court = (
        read_cached_court_calibration(
            cache_path,
            video_path=video_path,
            metadata=metadata,
            remove_net=options.remove_net_for_court_calibration,
        )
        if options.use_court_calibration_cache
        else None
    )
    if court is None:
        court = calibrator.calibrate_video(
            video_path,
            max_scan_frames=1 if frame_preprocessor is not None else 30,
            debug_dir=court_debug_dir,
        )
        if options.use_court_calibration_cache:
            write_cached_court_calibration(
                cache_path,
                video_path=video_path,
                metadata=metadata,
                remove_net=options.remove_net_for_court_calibration,
                calibration=court,
            )
    else:
        court.source = f"{court.source}:cache"
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

    segment_breaks = _clean_ball_track(observations, fps, options.thresholds)
    _interpolate_ball_track(observations, options.thresholds.interpolate_ball_gaps_up_to_frames, segment_breaks)
    _smooth_ball_track(observations, options.thresholds.smoothing_window, segment_breaks)
    _compute_ball_velocity(observations, fps, segment_breaks)
    return metadata, observations, court


def _clean_ball_track(observations: list[FrameObservation], fps: float, thresholds: Thresholds) -> set[int]:
    if fps <= 0.0 or not observations:
        return set()

    for observation in observations:
        if observation.ball_px is None or observation.ball_confidence < thresholds.min_ball_confidence:
            _clear_ball_observation(observation)

    segment_breaks = _initial_ball_segment_breaks(observations, thresholds)
    _reject_implausible_medium_gap_reappearances(observations, fps, thresholds, segment_breaks)
    segment_breaks = _initial_ball_segment_breaks(observations, thresholds)
    _reject_isolated_ball_outliers(observations, fps, thresholds, segment_breaks)
    return _initial_ball_segment_breaks(observations, thresholds)


def _initial_ball_segment_breaks(observations: list[FrameObservation], thresholds: Thresholds) -> set[int]:
    segment_breaks: set[int] = set()
    missing_run = 0
    seen_valid_detection = False
    out_of_play_run = 0
    segment_ended_by_out_of_play = False
    max_missing_gap = max(0, thresholds.ball_track_segment_gap_frames)

    for index, observation in enumerate(observations):
        if observation.ball_px is None:
            missing_run += 1
            out_of_play_run = 0
            continue

        if seen_valid_detection and missing_run > max_missing_gap:
            segment_breaks.add(index)
            segment_ended_by_out_of_play = False
            out_of_play_run = 0
        elif segment_ended_by_out_of_play:
            segment_breaks.add(index)
            segment_ended_by_out_of_play = False
            out_of_play_run = 0

        missing_run = 0
        seen_valid_detection = True

        if _ball_out_of_play(observation, thresholds):
            out_of_play_run += 1
            if out_of_play_run >= max(1, thresholds.ball_out_of_play_confirm_frames):
                segment_ended_by_out_of_play = True
        else:
            out_of_play_run = 0

    return segment_breaks


def _reject_implausible_medium_gap_reappearances(
    observations: list[FrameObservation],
    fps: float,
    thresholds: Thresholds,
    segment_breaks: set[int],
) -> None:
    valid_indices = _valid_ball_indices(observations)
    previous_index = None
    for index in valid_indices:
        if previous_index is None or _has_segment_break_between(previous_index, index, segment_breaks):
            previous_index = index
            continue

        gap_length = index - previous_index - 1
        if thresholds.interpolate_ball_gaps_up_to_frames < gap_length <= thresholds.ball_track_segment_gap_frames:
            speed = _speed_px_s(observations[previous_index], observations[index])
            if speed > thresholds.ball_outlier_max_speed_px_s:
                _clear_ball_observation(observations[index])

        if observations[index].ball_px is not None:
            previous_index = index


def _reject_isolated_ball_outliers(
    observations: list[FrameObservation],
    fps: float,
    thresholds: Thresholds,
    segment_breaks: set[int],
) -> None:
    rejected_indices: set[int] = set()
    for segment in _ball_track_segments(observations, segment_breaks):
        if len(segment) < 3:
            continue
        for segment_position in range(1, len(segment) - 1):
            previous_index = segment[segment_position - 1]
            current_index = segment[segment_position]
            following_index = segment[segment_position + 1]
            if _is_isolated_motion_outlier(
                observations[previous_index],
                observations[current_index],
                observations[following_index],
                fps,
                thresholds,
            ):
                rejected_indices.add(current_index)

    for index in rejected_indices:
        _clear_ball_observation(observations[index])


def _is_isolated_motion_outlier(
    previous: FrameObservation,
    current: FrameObservation,
    following: FrameObservation,
    fps: float,
    thresholds: Thresholds,
) -> bool:
    if previous.ball_px is None or current.ball_px is None or following.ball_px is None:
        return False

    speed_in = _speed_px_s(previous, current)
    speed_out = _speed_px_s(current, following)
    bridge_speed = _speed_px_s(previous, following)
    acceleration = _acceleration_px_s2(previous, current, following)
    expected_current = _interpolate_by_timestamp(previous, following, current.timestamp_s)
    residual_px = _point_distance(current.ball_px, expected_current)
    residual_floor_px = max(20.0, thresholds.ball_outlier_max_speed_px_s / max(fps, 1.0) * 0.25)

    implausible_local_motion = (
        max(speed_in, speed_out) > thresholds.ball_outlier_max_speed_px_s
        or acceleration > thresholds.ball_outlier_max_acceleration_px_s2
    )
    return implausible_local_motion and bridge_speed <= thresholds.ball_outlier_max_speed_px_s and residual_px >= residual_floor_px


def _interpolate_ball_track(
    observations: list[FrameObservation],
    max_gap: int,
    segment_breaks: set[int] | None = None,
) -> None:
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
            if (
                left_obs.ball_px is not None
                and right_obs.ball_px is not None
                and not _has_segment_break_between(gap_start, gap_end, segment_breaks or set())
            ):
                for fill_index in range(index, gap_end):
                    ratio = (fill_index - gap_start) / (gap_end - gap_start)
                    observations[fill_index].ball_px = _lerp_point(left_obs.ball_px, right_obs.ball_px, ratio)
                    observations[fill_index].ball_court = _lerp_optional_point(left_obs.ball_court, right_obs.ball_court, ratio)
                    observations[fill_index].ball_confidence = min(left_obs.ball_confidence, right_obs.ball_confidence) * 0.75
        index = gap_end


def _smooth_ball_track(
    observations: list[FrameObservation],
    window_size: int,
    segment_breaks: set[int] | None = None,
) -> None:
    if window_size <= 1:
        return
    radius = window_size // 2
    segment_breaks = segment_breaks or set()
    original_ball = [observation.ball_px for observation in observations]
    original_ball_court = [observation.ball_court for observation in observations]
    for index, observation in enumerate(observations):
        if original_ball[index] is None:
            continue
        px_window = [
            point
            for sample_index, point in enumerate(
                original_ball[max(0, index - radius) : min(len(original_ball), index + radius + 1)],
                start=max(0, index - radius),
            )
            if point is not None
            and not _has_segment_break_between(sample_index, index, segment_breaks)
        ]
        court_window = [
            point
            for sample_index, point in enumerate(
                original_ball_court[max(0, index - radius) : min(len(original_ball_court), index + radius + 1)],
                start=max(0, index - radius),
            )
            if point is not None
            and not _has_segment_break_between(sample_index, index, segment_breaks)
        ]
        averaged_px = average(px_window)
        averaged_court = average(court_window)
        if averaged_px is not None:
            observation.ball_px = averaged_px
        if averaged_court is not None:
            observation.ball_court = averaged_court


def _compute_ball_velocity(
    observations: list[FrameObservation],
    fps: float,
    segment_breaks: set[int] | None = None,
) -> None:
    if fps <= 0:
        return
    segment_breaks = segment_breaks or set()
    for index in range(1, len(observations)):
        current = observations[index]
        previous = observations[index - 1]
        if index in segment_breaks:
            continue
        if current.ball_px is None or previous.ball_px is None:
            continue
        delta = Point2D(current.ball_px.x - previous.ball_px.x, current.ball_px.y - previous.ball_px.y)
        current.ball_velocity_px_s = scale(delta, fps)


def _ball_track_segments(observations: list[FrameObservation], segment_breaks: set[int]) -> list[list[int]]:
    segments: list[list[int]] = []
    current_segment: list[int] = []
    for index, observation in enumerate(observations):
        if observation.ball_px is None:
            continue
        if current_segment and _has_segment_break_between(current_segment[-1], index, segment_breaks):
            segments.append(current_segment)
            current_segment = []
        current_segment.append(index)
    if current_segment:
        segments.append(current_segment)
    return segments


def _valid_ball_indices(observations: list[FrameObservation]) -> list[int]:
    return [index for index, observation in enumerate(observations) if observation.ball_px is not None]


def _ball_out_of_play(observation: FrameObservation, thresholds: Thresholds) -> bool:
    if observation.ball_court is None:
        return False
    margin = thresholds.ball_out_of_play_court_margin
    return not (-margin <= observation.ball_court.x <= 1.0 + margin and -margin <= observation.ball_court.y <= 1.0 + margin)


def _clear_ball_observation(observation: FrameObservation) -> None:
    observation.ball_px = None
    observation.ball_court = None
    observation.ball_velocity_px_s = None
    observation.ball_confidence = 0.0


def _has_segment_break_between(left_index: int, right_index: int, segment_breaks: set[int]) -> bool:
    lower = min(left_index, right_index)
    upper = max(left_index, right_index)
    return any(break_index in segment_breaks for break_index in range(lower + 1, upper + 1))


def _speed_px_s(left: FrameObservation, right: FrameObservation) -> float:
    if left.ball_px is None or right.ball_px is None:
        return math.inf
    duration_s = max(1e-6, right.timestamp_s - left.timestamp_s)
    return _point_distance(left.ball_px, right.ball_px) / duration_s


def _acceleration_px_s2(previous: FrameObservation, current: FrameObservation, following: FrameObservation) -> float:
    if previous.ball_px is None or current.ball_px is None or following.ball_px is None:
        return math.inf
    velocity_in = _velocity_between(previous, current)
    velocity_out = _velocity_between(current, following)
    duration_s = max(1e-6, following.timestamp_s - previous.timestamp_s)
    return _point_distance(velocity_in, velocity_out) / duration_s


def _velocity_between(left: FrameObservation, right: FrameObservation) -> Point2D:
    if left.ball_px is None or right.ball_px is None:
        return Point2D(0.0, 0.0)
    duration_s = max(1e-6, right.timestamp_s - left.timestamp_s)
    return Point2D(
        (right.ball_px.x - left.ball_px.x) / duration_s,
        (right.ball_px.y - left.ball_px.y) / duration_s,
    )


def _interpolate_by_timestamp(left: FrameObservation, right: FrameObservation, timestamp_s: float) -> Point2D:
    if left.ball_px is None or right.ball_px is None:
        return Point2D(0.0, 0.0)
    duration_s = max(1e-6, right.timestamp_s - left.timestamp_s)
    ratio = (timestamp_s - left.timestamp_s) / duration_s
    return _lerp_point(left.ball_px, right.ball_px, ratio)


def _point_distance(left: Point2D, right: Point2D) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


def _lerp_optional_point(left: Point2D | None, right: Point2D | None, ratio: float) -> Point2D | None:
    if left is None or right is None:
        return None
    return _lerp_point(left, right, ratio)


def _lerp_point(left: Point2D, right: Point2D, ratio: float) -> Point2D:
    return Point2D(
        left.x + ((right.x - left.x) * ratio),
        left.y + ((right.y - left.y) * ratio),
    )
