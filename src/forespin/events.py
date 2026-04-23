from __future__ import annotations

from bisect import bisect_left, bisect_right

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
    point_on_opponent_side,
    subtract,
)


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
    bounces: list[BounceEvent] = []
    for index in range(1, len(observations) - 1):
        previous = observations[index - 1]
        current = observations[index]
        following = observations[index + 1]
        if not _has_ball_triplet(previous, current, following):
            continue
        if any(abs(current.frame_index - hit_frame) <= thresholds.bounce_suppress_frames_after_hit for hit_frame in hit_frames):
            continue

        v1 = subtract(current.ball_px, previous.ball_px)
        v2 = subtract(following.ball_px, current.ball_px)
        angle_change = angle_change_degrees(v1, v2)
        speed1 = magnitude(v1)
        speed2 = magnitude(v2)
        if speed1 <= 1e-6:
            continue
        speed_ratio = speed2 / speed1
        if angle_change < thresholds.bounce_angle_change_deg and speed_ratio > thresholds.bounce_speed_drop_ratio:
            continue

        if bounces and current.frame_index - bounces[-1].frame_index <= thresholds.hit_suppress_window_frames:
            continue

        in_bounds = point_in_court(current.ball_court, tolerance=0.03)
        confidence = min(1.0, current.ball_confidence + (angle_change / 45.0) * 0.5 + max(0.0, 1.0 - speed_ratio) * 0.5)
        bounces.append(
            BounceEvent(
                frame_index=current.frame_index,
                timestamp_s=current.timestamp_s,
                ball_court=current.ball_court,
                in_bounds=in_bounds,
                confidence=confidence,
            )
        )

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
        best_index = None
        best_score = -1.0
        for frame_index in range(window_start + 1, exit_index):
            prev_obs = observations[frame_index - 1]
            curr_obs = observations[frame_index]
            next_obs = observations[frame_index + 1]
            if not _has_ball_triplet(prev_obs, curr_obs, next_obs):
                continue
            v1 = subtract(curr_obs.ball_px, prev_obs.ball_px)
            v2 = subtract(next_obs.ball_px, curr_obs.ball_px)
            score = angle_change_degrees(v1, v2) + (magnitude(subtract(v2, v1)) * 0.05)
            if score > best_score:
                best_score = score
                best_index = frame_index
        if best_index is None or best_index in existing_frames:
            continue
        observation = observations[best_index]
        bounces.append(
            BounceEvent(
                frame_index=observation.frame_index,
                timestamp_s=observation.timestamp_s,
                ball_court=observation.ball_court,
                in_bounds=point_in_court(observation.ball_court, tolerance=0.03)
                and point_on_opponent_side(observation.ball_court, input_config.tracked_player_side),
                confidence=min(0.7, observation.ball_confidence + 0.2),
                inferred_from_fallback=True,
            )
        )
        existing_frames.add(best_index)

    bounces.sort(key=lambda bounce: bounce.frame_index)
    return bounces


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

