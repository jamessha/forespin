from __future__ import annotations

import math
from pathlib import Path

from forespin.deps import require_vision_stack
from forespin.domain import AnalysisResult, BounceEvent, HitEvent, PlayerActor, Point2D


def render_overlay_video(result: AnalysisResult, output_path: str | Path) -> Path:
    cv2, _ = require_vision_stack()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(result.config.video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to reopen video for overlay rendering: {result.config.video_path}")

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        result.metadata.fps or 30.0,
        (result.metadata.width, result.metadata.height),
    )

    hits_by_frame = {hit.frame_index: hit for hit in result.hits}
    bounces_by_frame: dict[int, list[BounceEvent]] = {}
    for bounce in result.bounces:
        bounces_by_frame.setdefault(bounce.frame_index, []).append(bounce)
    bounces_so_far: list[BounceEvent] = []

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            _draw_calibrated_court(frame, result.court.corners_px, cv2)
            observation = result.observations[frame_index] if frame_index < len(result.observations) else None
            if observation is not None:
                _draw_point(frame, observation.ball_px, (0, 0, 255), 7)
                _draw_point(frame, observation.tracked_player_feet_px, (0, 255, 0), 8)
                if observation.tracked_player_bbox_px is not None:
                    bbox = observation.tracked_player_bbox_px
                    cv2.rectangle(frame, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (0, 200, 0), 2)
            if frame_index in hits_by_frame:
                _draw_hit_banner(frame, hits_by_frame[frame_index], cv2)
            for bounce in bounces_by_frame.get(frame_index, []):
                bounces_so_far.append(bounce)
                _draw_bounce_banner(frame, bounce, cv2)
            _draw_virtual_court(frame, bounces_so_far, cv2)
            _draw_summary(frame, result, cv2)
            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        capture.release()
    return output


def _draw_summary(frame, result: AnalysisResult, cv2) -> None:
    y = 26
    cv2.putText(frame, f"Status: {result.status.value}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    y += 26
    for shot_name, metric in result.metrics.shot_stats.items():
        label = f"{shot_name}: {metric.in_count}/{metric.attempts} ({metric.in_rate:.0%})"
        cv2.putText(frame, label, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 22


def _draw_hit_banner(frame, hit: HitEvent, cv2) -> None:
    actor = "tracked" if hit.actor == PlayerActor.TRACKED else "opponent"
    label = f"HIT {actor} {hit.shot_type.value}"
    cv2.putText(frame, label, (max(16, frame.shape[1] // 2 - 180), frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def _draw_bounce_banner(frame, bounce: BounceEvent, cv2) -> None:
    label = "BOUNCE in" if bounce.in_bounds else "BOUNCE out"
    cv2.putText(frame, label, (frame.shape[1] - 180, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)


def _draw_virtual_court(frame, bounces: list[BounceEvent], cv2) -> None:
    height, width = frame.shape[:2]
    margin = 18
    padding = 12
    court_width = min(180, max(90, int(width * 0.11)))
    court_height = int(court_width * (78.0 / 27.0))
    max_height = int(height * 0.42)
    if court_height > max_height:
        court_height = max_height
        court_width = int(court_height * (27.0 / 78.0))

    panel_width = court_width + padding * 2
    panel_height = court_height + padding * 2 + 22
    panel_x = margin
    panel_y = max(margin, height - margin - panel_height)
    court_x = panel_x + padding
    court_y = panel_y + padding + 22

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (12, 24, 18),
        -1,
    )
    cv2.putText(
        frame,
        "Bounce map",
        (panel_x + padding, panel_y + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (235, 245, 235),
        1,
    )

    def to_px(point: Point2D) -> tuple[int, int]:
        return (
            int(round(court_x + point.x * court_width)),
            int(round(court_y + point.y * court_height)),
        )

    line_color = (235, 235, 225)
    cv2.rectangle(frame, (court_x, court_y), (court_x + court_width, court_y + court_height), line_color, 2)

    # Regulation singles court proportions: service lines are 21 ft from the net on a 78 ft court.
    net_y = 0.5
    far_service_y = 18.0 / 78.0
    near_service_y = 60.0 / 78.0
    center_x = 0.5
    for y in (net_y, far_service_y, near_service_y):
        start = to_px(Point2D(0.0, y))
        end = to_px(Point2D(1.0, y))
        cv2.line(frame, start, end, line_color, 1)
    cv2.line(frame, to_px(Point2D(center_x, far_service_y)), to_px(Point2D(center_x, near_service_y)), line_color, 1)

    recent_frame = bounces[-1].frame_index if bounces else None
    for bounce in bounces[-80:]:
        if bounce.ball_court is None:
            continue
        clamped = _bounce_map_point(bounce.ball_court)
        point = to_px(clamped)
        color = (0, 220, 80) if bounce.in_bounds else (0, 80, 255)
        radius = 4 if bounce.frame_index == recent_frame else 3
        cv2.circle(frame, point, radius, color, -1)
        if bounce.frame_index == recent_frame:
            cv2.circle(frame, point, radius + 4, color, 1)


def _bounce_map_point(point: Point2D) -> Point2D:
    return Point2D(
        min(max(1.0 - point.x, 0.0), 1.0),
        min(max(point.y, 0.0), 1.0),
    )


def _draw_calibrated_court(frame, corners: list[Point2D], cv2) -> None:
    if len(corners) != 4:
        return
    clipped = [_clip_point_to_frame(point, frame.shape[1], frame.shape[0]) for point in corners]
    if any(point is None for point in clipped):
        return
    points = [point for point in clipped if point is not None]
    overlay = frame.copy()
    for start, end in zip(points, points[1:] + points[:1]):
        cv2.line(
            overlay,
            (int(start.x), int(start.y)),
            (int(end.x), int(end.y)),
            (255, 80, 0),
            3,
        )
    for index, point in enumerate(points):
        cv2.circle(overlay, (int(point.x), int(point.y)), 5, (255, 160, 0), -1)
        cv2.putText(
            overlay,
            f"C{index}",
            (int(point.x) + 6, int(point.y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 160, 0),
            2,
        )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)


def _clip_point_to_frame(point: Point2D, width: int, height: int) -> Point2D | None:
    if not math.isfinite(point.x) or not math.isfinite(point.y):
        return None
    return Point2D(
        min(max(point.x, 0.0), float(max(0, width - 1))),
        min(max(point.y, 0.0), float(max(0, height - 1))),
    )


def _draw_point(frame, point: Point2D | None, color: tuple[int, int, int], radius: int) -> None:
    if point is None:
        return
    cv2, _ = require_vision_stack()
    cv2.circle(frame, (int(point.x), int(point.y)), radius, color, -1)
