from __future__ import annotations

import math
from collections.abc import Iterable

from forespin.config import Thresholds
from forespin.domain import Handedness, Point2D, TrackedPlayerSide

SINGLES_LEFT_X = 4.5 / 36.0
SINGLES_RIGHT_X = 31.5 / 36.0
FAR_SERVICE_Y = 18.0 / 78.0
NEAR_SERVICE_Y = 60.0 / 78.0


def distance(a: Point2D | None, b: Point2D | None) -> float:
    if a is None or b is None:
        return math.inf
    return math.hypot(a.x - b.x, a.y - b.y)


def vector(a: Point2D, b: Point2D) -> Point2D:
    return Point2D(b.x - a.x, b.y - a.y)


def magnitude(point: Point2D | None) -> float:
    if point is None:
        return 0.0
    return math.hypot(point.x, point.y)


def dot(a: Point2D, b: Point2D) -> float:
    return a.x * b.x + a.y * b.y


def subtract(a: Point2D, b: Point2D) -> Point2D:
    return Point2D(a.x - b.x, a.y - b.y)


def add(a: Point2D, b: Point2D) -> Point2D:
    return Point2D(a.x + b.x, a.y + b.y)


def scale(point: Point2D, factor: float) -> Point2D:
    return Point2D(point.x * factor, point.y * factor)


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def average(points: Iterable[Point2D]) -> Point2D | None:
    points_list = list(points)
    if not points_list:
        return None
    return Point2D(
        sum(point.x for point in points_list) / len(points_list),
        sum(point.y for point in points_list) / len(points_list),
    )


def angle_change_degrees(v1: Point2D, v2: Point2D) -> float:
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot(v1, v2) / (mag1 * mag2)))
    return math.degrees(math.acos(cosine))


def transform_point(point: Point2D | None, homography: list[list[float]]) -> Point2D | None:
    if point is None:
        return None
    x = point.x
    y = point.y
    w = (homography[2][0] * x) + (homography[2][1] * y) + homography[2][2]
    if w == 0.0:
        return None
    tx = ((homography[0][0] * x) + (homography[0][1] * y) + homography[0][2]) / w
    ty = ((homography[1][0] * x) + (homography[1][1] * y) + homography[1][2]) / w
    return Point2D(tx, ty)


def point_in_court(point: Point2D | None, tolerance: float = 0.03) -> bool:
    if point is None:
        return False
    return (-tolerance <= point.x <= 1.0 + tolerance) and (-tolerance <= point.y <= 1.0 + tolerance)


def point_in_singles_court(point: Point2D | None, tolerance: float = 0.03) -> bool:
    if point is None:
        return False
    return (
        SINGLES_LEFT_X - tolerance <= point.x <= SINGLES_RIGHT_X + tolerance
        and -tolerance <= point.y <= 1.0 + tolerance
    )


def point_on_opponent_side(point: Point2D | None, tracked_side: TrackedPlayerSide) -> bool:
    if point is None:
        return False
    return point.y < 0.5 if tracked_side == TrackedPlayerSide.NEAR else point.y > 0.5


def point_on_tracked_side(point: Point2D | None, tracked_side: TrackedPlayerSide) -> bool:
    if point is None:
        return False
    return not point_on_opponent_side(point, tracked_side)


def classify_depth_zone(
    point: Point2D | None,
    tracked_side: TrackedPlayerSide,
    thresholds: Thresholds,
) -> str:
    if point is None:
        return "other"
    baseline_y = 1.0 if tracked_side == TrackedPlayerSide.NEAR else 0.0
    if abs(point.y - baseline_y) <= thresholds.baseline_zone_margin:
        return "baseline"
    if abs(point.y - 0.5) <= thresholds.net_zone_margin:
        return "net"
    return "other"


def classify_lateral_zone(point: Point2D | None) -> str:
    if point is None:
        return "right"
    return "left" if point.x < 0.5 else "right"


def inside_service_line(
    point: Point2D | None,
    tracked_side: TrackedPlayerSide,
    thresholds: Thresholds,
) -> bool:
    if point is None:
        return False
    if tracked_side == TrackedPlayerSide.NEAR:
        return point.y <= thresholds.service_line_near_y + 0.04
    return point.y >= thresholds.service_line_far_y - 0.04


def expected_forehand_is_ball_right_of_player(
    tracked_side: TrackedPlayerSide,
    handedness: Handedness,
) -> bool:
    if handedness == Handedness.RIGHT:
        return tracked_side == TrackedPlayerSide.NEAR
    return tracked_side == TrackedPlayerSide.FAR
