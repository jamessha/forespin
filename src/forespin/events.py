from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from forespin.config import Thresholds
from forespin.domain import (
    BounceEvent,
    FrameObservation,
    Handedness,
    HitEvent,
    InputConfig,
    PlayerActor,
    Point2D,
    ShotOutcome,
    ShotType,
    TrackedPlayerSide,
)
from forespin.geometry import (
    angle_change_degrees,
    distance,
    dot,
    expected_forehand_is_ball_right_of_player,
    inside_service_line,
    magnitude,
    point_in_court,
    point_in_singles_court,
    point_on_opponent_side,
    scale,
    subtract,
)


@dataclass(slots=True)
class _BounceCandidate:
    frame_index: int
    timestamp_s: float
    ball_court: Point2D | None
    in_bounds: bool
    confidence: float
    score: float


def detect_hits(
    observations: list[FrameObservation],
    input_config: InputConfig,
    thresholds: Thresholds,
) -> list[HitEvent]:
    hits: list[HitEvent] = []
    for index in range(1, len(observations) - 1):
        previous = observations[index - 1]
        current = observations[index]
        following = observations[index + 1]
        if not _has_ball_triplet(previous, current, following):
            continue

        v1 = subtract(current.ball_px, previous.ball_px)
        v2 = subtract(following.ball_px, current.ball_px)
        speed1 = magnitude(v1) / max(1e-6, current.timestamp_s - previous.timestamp_s)
        speed2 = magnitude(v2) / max(1e-6, following.timestamp_s - current.timestamp_s)
        if min(speed1, speed2) < thresholds.hit_min_speed_px_s:
            continue

        normalized_dot = dot(v1, v2) / max(1e-6, magnitude(v1) * magnitude(v2))
        acceleration = magnitude(subtract(v2, v1)) / max(1e-6, following.timestamp_s - previous.timestamp_s)
        if normalized_dot > thresholds.hit_reversal_dot_threshold and acceleration < thresholds.hit_min_acceleration_px_s2:
            continue

        if hits and current.frame_index - hits[-1].frame_index <= thresholds.hit_suppress_window_frames:
            continue

        actor = _infer_actor(current, input_config.tracked_player_side, thresholds)
        confidence = min(1.0, current.ball_confidence + current.tracked_player_confidence * 0.25 + max(0.0, -normalized_dot) * 0.5)
        hits.append(
            HitEvent(
                frame_index=current.frame_index,
                timestamp_s=current.timestamp_s,
                actor=actor,
                shot_type=ShotType.UNKNOWN,
                ball_px=current.ball_px,
                ball_court=current.ball_court,
                player_court=current.tracked_player_feet_court if actor == PlayerActor.TRACKED else None,
                confidence=confidence,
            )
        )
    return annotate_shot_types(hits, observations, [], input_config, thresholds)


def detect_bounces(
    observations: list[FrameObservation],
    hits: list[HitEvent],
    input_config: InputConfig,
    thresholds: Thresholds,
) -> list[BounceEvent]:
    hit_frames = {hit.frame_index for hit in hits}
    candidates: list[_BounceCandidate] = []
    for index in range(len(observations)):
        floor_candidate = _score_floor_bounce_candidate(index, observations, hit_frames, thresholds)
        if floor_candidate is not None:
            candidates.append(floor_candidate)

        court_projection_candidate = _score_court_projection_bounce_candidate(index, observations, hit_frames, thresholds)
        if court_projection_candidate is not None:
            candidates.append(court_projection_candidate)

    bounces = [
        BounceEvent(
            frame_index=candidate.frame_index,
            timestamp_s=candidate.timestamp_s,
            ball_court=candidate.ball_court,
            in_bounds=candidate.in_bounds,
            confidence=candidate.confidence,
        )
        for candidate in _select_bounce_candidates(candidates, thresholds)
    ]

    return _inject_fallback_bounces(observations, hits, bounces, input_config, thresholds)


def annotate_shot_types(
    hits: list[HitEvent],
    observations: list[FrameObservation],
    bounces: list[BounceEvent],
    input_config: InputConfig,
    thresholds: Thresholds,
) -> list[HitEvent]:
    bounce_frames = [bounce.frame_index for bounce in bounces]
    last_hit_frame = None
    last_opponent_hit_frame = None
    for hit in hits:
        is_new_point = last_hit_frame is None or hit.frame_index - last_hit_frame > thresholds.dead_ball_gap_frames
        if hit.actor == PlayerActor.TRACKED:
            near_baseline = _player_near_baseline(hit.player_court, input_config.tracked_player_side, thresholds)
            if is_new_point and near_baseline:
                hit.shot_type = ShotType.SERVE
                hit.is_serve = True
            else:
                has_bounce_since_opponent = False
                if last_opponent_hit_frame is not None and bounce_frames:
                    left = bisect_left(bounce_frames, last_opponent_hit_frame + 1)
                    right = bisect_left(bounce_frames, hit.frame_index)
                    has_bounce_since_opponent = left < right
                if last_opponent_hit_frame is not None and not has_bounce_since_opponent and inside_service_line(hit.player_court, input_config.tracked_player_side, thresholds):
                    hit.shot_type = ShotType.VOLLEY
                    hit.is_volley = True
                else:
                    hit.shot_type = _classify_groundstroke(hit, input_config.handedness, input_config.tracked_player_side, observations)
        elif hit.actor == PlayerActor.OPPONENT:
            last_opponent_hit_frame = hit.frame_index

        last_hit_frame = hit.frame_index
    return hits


def annotate_hit_outcomes(
    hits: list[HitEvent],
    bounces: list[BounceEvent],
    input_config: InputConfig,
) -> list[HitEvent]:
    bounce_frames = [bounce.frame_index for bounce in bounces]
    for index, hit in enumerate(hits):
        if hit.actor != PlayerActor.TRACKED:
            continue

        next_hit = hits[index + 1] if index + 1 < len(hits) else None
        next_bounce = _first_bounce_after(hit.frame_index, bounces, bounce_frames)
        if next_hit is not None and next_bounce is not None:
            if next_hit.frame_index < next_bounce.frame_index:
                next_event = next_hit
            else:
                next_event = next_bounce
        else:
            next_event = next_hit or next_bounce

        if next_event is None:
            hit.result = ShotOutcome.UNKNOWN
            hit.result_reason = "no_continuation"
            continue
        if isinstance(next_event, HitEvent):
            if next_event.actor == PlayerActor.OPPONENT:
                hit.result = ShotOutcome.IN
                hit.result_reason = "opponent_contact"
            else:
                hit.result = ShotOutcome.UNKNOWN
                hit.result_reason = "missing_intermediate_event"
            continue
        if next_event.in_bounds and point_on_opponent_side(next_event.ball_court, input_config.tracked_player_side):
            hit.result = ShotOutcome.IN
            hit.result_reason = "in_bounds_bounce"
        else:
            hit.result = ShotOutcome.OUT
            hit.result_reason = "out_or_wrong_side_bounce"
    return hits


def _inject_fallback_bounces(
    observations: list[FrameObservation],
    hits: list[HitEvent],
    bounces: list[BounceEvent],
    input_config: InputConfig,
    thresholds: Thresholds,
) -> list[BounceEvent]:
    existing_frames = {bounce.frame_index for bounce in bounces}
    for hit_index, hit in enumerate(hits):
        if hit.actor != PlayerActor.TRACKED:
            continue
        next_hit_frame = hits[hit_index + 1].frame_index if hit_index + 1 < len(hits) else observations[-1].frame_index
        first_bounce = next(
            (bounce for bounce in bounces if hit.frame_index < bounce.frame_index < next_hit_frame),
            None,
        )
        if first_bounce is not None:
            continue
        search_start = min(len(observations) - 1, hit.frame_index + 1)
        search_end = min(len(observations) - 1, next_hit_frame)
        exit_index = next(
            (
                observation.frame_index
                for observation in observations[search_start:search_end]
                if observation.ball_court is not None and not point_in_court(observation.ball_court, tolerance=0.08)
            ),
            None,
        )
        if exit_index is None:
            continue
        window_start = max(hit.frame_index + 1, exit_index - thresholds.fallback_bounce_window_frames)
        best_candidate = None
        for frame_index in range(window_start + 1, exit_index):
            candidate = _score_floor_bounce_candidate(frame_index, observations, {hit.frame_index}, thresholds)
            if candidate is not None and (best_candidate is None or candidate.score > best_candidate.score):
                best_candidate = candidate
        if best_candidate is None or best_candidate.frame_index in existing_frames:
            continue
        bounces.append(
            BounceEvent(
                frame_index=best_candidate.frame_index,
                timestamp_s=best_candidate.timestamp_s,
                ball_court=best_candidate.ball_court,
                in_bounds=best_candidate.in_bounds and point_on_opponent_side(best_candidate.ball_court, input_config.tracked_player_side),
                confidence=min(0.7, best_candidate.confidence),
                inferred_from_fallback=True,
            )
        )
        existing_frames.add(best_candidate.frame_index)

    bounces.sort(key=lambda bounce: bounce.frame_index)
    return bounces


def _score_floor_bounce_candidate(
    index: int,
    observations: list[FrameObservation],
    hit_frames: set[int],
    thresholds: Thresholds,
) -> _BounceCandidate | None:
    current = observations[index]
    if current.ball_px is None or current.ball_confidence < thresholds.min_ball_confidence:
        return None
    if current.ball_court is not None and not point_in_court(current.ball_court, tolerance=thresholds.bounce_court_margin):
        return None

    previous = _nearest_ball_observation(observations, index - 1, -1, -1, thresholds.bounce_floor_window_frames)
    following = _nearest_ball_observation(observations, index + 1, len(observations), 1, thresholds.bounce_floor_window_frames)
    if previous is None or following is None or previous.ball_px is None or following.ball_px is None:
        return None

    left_neighbors = _ball_neighbors(observations, index, -1, thresholds.bounce_floor_window_frames)
    right_neighbors = _ball_neighbors(observations, index, 1, thresholds.bounce_floor_window_frames)
    if not left_neighbors or not right_neighbors:
        return None

    current_y = current.ball_px.y
    if current_y < max(observation.ball_px.y for observation in left_neighbors if observation.ball_px is not None):
        return None
    if current_y < max(observation.ball_px.y for observation in right_neighbors if observation.ball_px is not None):
        return None

    left_lowest_approach_y = min(observation.ball_px.y for observation in left_neighbors if observation.ball_px is not None)
    right_lowest_exit_y = min(observation.ball_px.y for observation in right_neighbors if observation.ball_px is not None)
    vertical_prominence = current_y - max(left_lowest_approach_y, right_lowest_exit_y)
    if vertical_prominence < thresholds.bounce_min_vertical_prominence_px:
        return None

    v1 = subtract(current.ball_px, previous.ball_px)
    v2 = subtract(following.ball_px, current.ball_px)
    if v1.y < 0.0 or v2.y > 0.0:
        return None

    angle_change = angle_change_degrees(v1, v2)
    speed1 = magnitude(v1)
    speed2 = magnitude(v2)
    if speed1 <= 1e-6:
        return None
    speed_ratio = speed2 / speed1
    if angle_change < thresholds.bounce_angle_change_deg and speed_ratio > thresholds.bounce_speed_drop_ratio:
        return None

    in_bounds = point_in_singles_court(current.ball_court, tolerance=0.03)
    near_hit = any(abs(current.frame_index - hit_frame) <= thresholds.bounce_suppress_frames_after_hit for hit_frame in hit_frames)
    strong_floor_contact = (
        in_bounds
        and vertical_prominence >= thresholds.bounce_hit_overlap_min_vertical_prominence_px
        and speed_ratio <= thresholds.bounce_hit_overlap_max_speed_ratio
    )
    if near_hit and not strong_floor_contact:
        return None

    score = vertical_prominence + (angle_change * 0.35) + (max(0.0, 1.0 - speed_ratio) * 20.0) + (current.ball_confidence * 10.0)
    confidence = min(1.0, current.ball_confidence + (vertical_prominence / 40.0) * 0.35 + (angle_change / 90.0) * 0.25)
    return _BounceCandidate(
        frame_index=current.frame_index,
        timestamp_s=current.timestamp_s,
        ball_court=current.ball_court,
        in_bounds=in_bounds,
        confidence=confidence,
        score=score,
    )


def _score_court_projection_bounce_candidate(
    index: int,
    observations: list[FrameObservation],
    hit_frames: set[int],
    thresholds: Thresholds,
) -> _BounceCandidate | None:
    current = observations[index]
    if current.ball_court is None or current.ball_confidence < thresholds.min_ball_confidence:
        return None
    if not point_in_court(current.ball_court, tolerance=thresholds.bounce_court_margin):
        return None

    previous = _nearest_ball_observation(observations, index - 1, -1, -1, thresholds.bounce_floor_window_frames)
    following = _nearest_ball_observation(observations, index + 1, len(observations), 1, thresholds.bounce_floor_window_frames)
    if previous is None or following is None or previous.ball_court is None or following.ball_court is None:
        return None

    left_neighbors = _court_neighbors(observations, index, -1, thresholds.bounce_floor_window_frames)
    right_neighbors = _court_neighbors(observations, index, 1, thresholds.bounce_floor_window_frames)
    if not left_neighbors or not right_neighbors:
        return None

    current_y = current.ball_court.y
    left_y_values = [observation.ball_court.y for observation in left_neighbors if observation.ball_court is not None]
    right_y_values = [observation.ball_court.y for observation in right_neighbors if observation.ball_court is not None]
    if not left_y_values or not right_y_values:
        return None

    local_max_y = current_y >= max(left_y_values) and current_y >= max(right_y_values)
    local_min_y = current_y <= min(left_y_values) and current_y <= min(right_y_values)
    if local_max_y:
        y_prominence = current_y - max(min(left_y_values), min(right_y_values))
    elif local_min_y:
        y_prominence = min(max(left_y_values), max(right_y_values)) - current_y
    else:
        y_prominence = 0.0

    v1 = subtract(current.ball_court, previous.ball_court)
    v2 = subtract(following.ball_court, current.ball_court)
    previous_dt = max(1e-6, current.timestamp_s - previous.timestamp_s)
    following_dt = max(1e-6, following.timestamp_s - current.timestamp_s)
    velocity_in = scale(v1, 1.0 / previous_dt)
    velocity_out = scale(v2, 1.0 / following_dt)
    speed1 = magnitude(velocity_in)
    speed2 = magnitude(velocity_out)
    if min(speed1, speed2) < thresholds.bounce_court_projection_min_speed_per_s:
        return None

    angle_change = angle_change_degrees(v1, v2)
    y_velocity_change = abs(velocity_out.y - velocity_in.y)
    y_sign_turn = (velocity_in.y >= 0.0 >= velocity_out.y) or (velocity_in.y <= 0.0 <= velocity_out.y)
    has_projected_floor_turn = (
        y_sign_turn
        and y_prominence >= thresholds.bounce_court_projection_min_y_prominence
        and angle_change >= thresholds.bounce_court_projection_min_angle_change_deg
    )
    has_projected_kink = (
        y_prominence >= thresholds.bounce_court_projection_min_y_prominence * 1.6
        and angle_change >= thresholds.bounce_court_projection_min_angle_change_deg * 1.4
        and y_velocity_change >= thresholds.bounce_court_projection_min_speed_per_s
    )
    if not has_projected_floor_turn and not has_projected_kink:
        return None

    in_bounds = point_in_singles_court(current.ball_court, tolerance=0.03)
    near_hit = any(abs(current.frame_index - hit_frame) <= thresholds.bounce_suppress_frames_after_hit for hit_frame in hit_frames)
    strong_court_contact = (
        in_bounds
        and y_prominence >= thresholds.bounce_court_projection_min_y_prominence * 2.0
        and angle_change >= thresholds.bounce_court_projection_min_angle_change_deg * 1.25
    )
    if near_hit and not strong_court_contact:
        return None

    score = (
        y_prominence * 1200.0
        + angle_change * 0.45
        + y_velocity_change * 4.0
        + current.ball_confidence * 8.0
    )
    confidence = min(
        1.0,
        current.ball_confidence
        + min(0.25, y_prominence / 0.05 * 0.25)
        + min(0.2, angle_change / 120.0 * 0.2),
    )
    return _BounceCandidate(
        frame_index=current.frame_index,
        timestamp_s=current.timestamp_s,
        ball_court=current.ball_court,
        in_bounds=in_bounds,
        confidence=confidence,
        score=score,
    )


def _select_bounce_candidates(candidates: list[_BounceCandidate], thresholds: Thresholds) -> list[_BounceCandidate]:
    selected: list[_BounceCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.frame_index):
        if selected and candidate.frame_index - selected[-1].frame_index <= thresholds.bounce_min_separation_frames:
            if candidate.score > selected[-1].score:
                selected[-1] = candidate
            continue
        selected.append(candidate)
    return selected


def _nearest_ball_observation(
    observations: list[FrameObservation],
    start: int,
    stop: int,
    step: int,
    window_frames: int,
) -> FrameObservation | None:
    for index in range(start, stop, step):
        if abs(observations[index].frame_index - observations[start].frame_index) > window_frames:
            return None
        if observations[index].ball_px is not None:
            return observations[index]
    return None


def _ball_neighbors(
    observations: list[FrameObservation],
    center_index: int,
    step: int,
    window_frames: int,
) -> list[FrameObservation]:
    neighbors: list[FrameObservation] = []
    center_frame = observations[center_index].frame_index
    index = center_index + step
    while 0 <= index < len(observations) and abs(observations[index].frame_index - center_frame) <= window_frames:
        if observations[index].ball_px is not None:
            neighbors.append(observations[index])
        index += step
    return neighbors


def _court_neighbors(
    observations: list[FrameObservation],
    center_index: int,
    step: int,
    window_frames: int,
) -> list[FrameObservation]:
    neighbors: list[FrameObservation] = []
    center_frame = observations[center_index].frame_index
    index = center_index + step
    while 0 <= index < len(observations) and abs(observations[index].frame_index - center_frame) <= window_frames:
        if observations[index].ball_court is not None:
            neighbors.append(observations[index])
        index += step
    return neighbors


def _first_bounce_after(frame_index: int, bounces: list[BounceEvent], bounce_frames: list[int]) -> BounceEvent | None:
    if not bounce_frames:
        return None
    offset = bisect_right(bounce_frames, frame_index)
    if offset >= len(bounces):
        return None
    return bounces[offset]


def _player_near_baseline(point: Point2D | None, tracked_side: TrackedPlayerSide, thresholds: Thresholds) -> bool:
    if point is None:
        return False
    baseline_y = 1.0 if tracked_side == TrackedPlayerSide.NEAR else 0.0
    return abs(point.y - baseline_y) <= thresholds.baseline_zone_margin + 0.02


def _classify_groundstroke(
    hit: HitEvent,
    handedness: Handedness,
    tracked_side: TrackedPlayerSide,
    observations: list[FrameObservation],
) -> ShotType:
    if hit.ball_court is None or hit.player_court is None:
        return ShotType.UNKNOWN
    ball_on_right = hit.ball_court.x >= hit.player_court.x
    forehand_right_side = expected_forehand_is_ball_right_of_player(tracked_side, handedness)
    is_forehand = ball_on_right == forehand_right_side
    return ShotType.FOREHAND if is_forehand else ShotType.BACKHAND


def _infer_actor(observation: FrameObservation, tracked_side: TrackedPlayerSide, thresholds: Thresholds) -> PlayerActor:
    if observation.ball_court is None:
        return PlayerActor.UNKNOWN
    if distance(observation.ball_court, observation.tracked_player_feet_court) <= thresholds.tracked_contact_distance_court:
        return PlayerActor.TRACKED
    if point_on_opponent_side(observation.ball_court, tracked_side):
        return PlayerActor.OPPONENT
    return PlayerActor.UNKNOWN


def _has_ball_triplet(first: FrameObservation, second: FrameObservation, third: FrameObservation) -> bool:
    return first.ball_px is not None and second.ball_px is not None and third.ball_px is not None
