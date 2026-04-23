from __future__ import annotations

from forespin.config import Thresholds
from forespin.domain import HitEvent, InputConfig, MetricBucket, MetricsReport, PointEvent, ShotMetric, ShotOutcome, ShotType
from forespin.geometry import classify_depth_zone, classify_lateral_zone, safe_rate


SHOT_TYPES = (ShotType.SERVE, ShotType.FOREHAND, ShotType.BACKHAND, ShotType.VOLLEY)


def compute_metrics(
    hits: list[HitEvent],
    points: list[PointEvent],
    input_config: InputConfig,
    thresholds: Thresholds,
) -> MetricsReport:
    shot_stats = {shot_type.value: ShotMetric() for shot_type in SHOT_TYPES}
    lost_depth = {name: MetricBucket() for name in ("baseline", "net", "other")}
    lost_lateral = {name: MetricBucket() for name in ("left", "right")}
    ignored = 0
    warnings: list[str] = []

    for hit in hits:
        if hit.shot_type not in SHOT_TYPES:
            continue
        if hit.result == ShotOutcome.UNKNOWN:
            ignored += 1
            continue
        metric = shot_stats[hit.shot_type.value]
        metric.attempts += 1
        if hit.result == ShotOutcome.IN:
            metric.in_count += 1

    for metric in shot_stats.values():
        metric.in_rate = safe_rate(metric.in_count, metric.attempts)

    lost_points = [point for point in points if point.tracked_player_lost]
    for point in lost_points:
        depth_key = classify_depth_zone(point.tracked_player_final_location, input_config.tracked_player_side, thresholds)
        lateral_key = classify_lateral_zone(point.tracked_player_final_location)
        lost_depth[depth_key].count += 1
        lost_lateral[lateral_key].count += 1

    for bucket in lost_depth.values():
        bucket.percentage = safe_rate(bucket.count, len(lost_points))
    for bucket in lost_lateral.values():
        bucket.percentage = safe_rate(bucket.count, len(lost_points))

    if ignored:
        warnings.append(f"Ignored {ignored} tracked shots because the continuation could not be resolved.")
    if not lost_points:
        warnings.append("No lost points were resolved with sufficient confidence.")

    return MetricsReport(
        shot_stats=shot_stats,
        lost_point_depth=lost_depth,
        lost_point_lateral=lost_lateral,
        lost_point_count=len(lost_points),
        ignored_tracked_shots=ignored,
        warnings=warnings,
    )

