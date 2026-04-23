from __future__ import annotations

from statistics import mean

from forespin.config import Thresholds
from forespin.domain import (
    BounceEvent,
    FrameObservation,
    HitEvent,
    InputConfig,
    PlayerActor,
    PointEvent,
    PointTermination,
    ShotOutcome,
)
from forespin.geometry import point_on_opponent_side, point_on_tracked_side


def assemble_points(
    observations: list[FrameObservation],
    hits: list[HitEvent],
    bounces: list[BounceEvent],
    input_config: InputConfig,
    thresholds: Thresholds,
) -> list[PointEvent]:
    if not hits:
        return []

    points: list[PointEvent] = []
    current_hits: list[HitEvent] = []
    for hit in hits:
        if current_hits and hit.frame_index - current_hits[-1].frame_index > thresholds.dead_ball_gap_frames:
            points.append(_finalize_point(points, current_hits, bounces, observations, input_config))
            current_hits = [hit]
        else:
            current_hits.append(hit)
    if current_hits:
        points.append(_finalize_point(points, current_hits, bounces, observations, input_config))
    return points


def _finalize_point(
    existing_points: list[PointEvent],
    point_hits: list[HitEvent],
    all_bounces: list[BounceEvent],
    observations: list[FrameObservation],
    input_config: InputConfig,
) -> PointEvent:
    point_start = point_hits[0].frame_index
    point_end = point_hits[-1].frame_index
    point_bounces = [bounce for bounce in all_bounces if point_start <= bounce.frame_index <= point_end + 1]
    if point_bounces:
        point_end = max(point_end, point_bounces[-1].frame_index)

    tracked_player_lost, winner, termination = _resolve_point_outcome(point_hits, point_bounces, input_config)
    final_location = _last_tracked_location(observations, point_end, point_start)
    confidence = mean(hit.confidence for hit in point_hits) if point_hits else 0.0
    return PointEvent(
        point_index=len(existing_points) + 1,
        start_frame=point_start,
        end_frame=point_end,
        hit_events=point_hits,
        bounce_events=point_bounces,
        tracked_player_lost=tracked_player_lost,
        winner=winner,
        terminal_reason=termination,
        tracked_player_final_location=final_location,
        confidence=confidence,
    )


def _resolve_point_outcome(
    point_hits: list[HitEvent],
    point_bounces: list[BounceEvent],
    input_config: InputConfig,
) -> tuple[bool, PlayerActor, PointTermination]:
    last_hit = point_hits[-1]
    post_hit_bounce = next((bounce for bounce in point_bounces if bounce.frame_index > last_hit.frame_index), None)

    if last_hit.actor == PlayerActor.TRACKED:
        if last_hit.result == ShotOutcome.OUT:
            return True, PlayerActor.OPPONENT, PointTermination.TRACKED_ERROR
        if post_hit_bounce is not None and post_hit_bounce.in_bounds and point_on_opponent_side(post_hit_bounce.ball_court, input_config.tracked_player_side):
            return False, PlayerActor.TRACKED, PointTermination.TRACKED_WINNER
        if last_hit.result == ShotOutcome.IN:
            return False, PlayerActor.TRACKED, PointTermination.OPPONENT_ERROR
        return True, PlayerActor.OPPONENT, PointTermination.DEAD_BALL

    if last_hit.actor == PlayerActor.OPPONENT:
        if post_hit_bounce is not None and post_hit_bounce.in_bounds and point_on_tracked_side(post_hit_bounce.ball_court, input_config.tracked_player_side):
            return True, PlayerActor.OPPONENT, PointTermination.OPPONENT_WINNER
        if post_hit_bounce is not None and not post_hit_bounce.in_bounds:
            return False, PlayerActor.TRACKED, PointTermination.OPPONENT_ERROR
        return True, PlayerActor.OPPONENT, PointTermination.OPPONENT_WINNER

    return True, PlayerActor.OPPONENT, PointTermination.DEAD_BALL


def _last_tracked_location(
    observations: list[FrameObservation],
    point_end: int,
    point_start: int,
):
    for observation in reversed(observations[point_start : point_end + 1]):
        if observation.tracked_player_feet_court is not None:
            return observation.tracked_player_feet_court
    return None

