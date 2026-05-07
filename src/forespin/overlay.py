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
    bounces_by_frame = {bounce.frame_index: bounce for bounce in result.bounces}

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
            if frame_index in bounces_by_frame:
                _draw_bounce_banner(frame, bounces_by_frame[frame_index], cv2)
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
    cv2.putText(frame, label, (16, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def _draw_bounce_banner(frame, bounce: BounceEvent, cv2) -> None:
    label = "BOUNCE in" if bounce.in_bounds else "BOUNCE out"
    cv2.putText(frame, label, (frame.shape[1] - 180, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)


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
