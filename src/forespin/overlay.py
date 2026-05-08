from __future__ import annotations

import math
from pathlib import Path

from forespin.deps import require_vision_stack
from forespin.domain import AnalysisResult, BounceEvent, HitEvent, PlayerActor, Point2D
from forespin.geometry import FAR_SERVICE_Y, NEAR_SERVICE_Y, SINGLES_LEFT_X, SINGLES_RIGHT_X

DOUBLES_COURT_WIDTH_FT = 36.0
COURT_LENGTH_FT = 78.0
COURT_CENTER_X = 0.5


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
            _draw_calibrated_court(frame, result.court.homography, cv2)
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
    court_height = int(court_width * (COURT_LENGTH_FT / DOUBLES_COURT_WIDTH_FT))
    max_height = int(height * 0.42)
    if court_height > max_height:
        court_height = max_height
        court_width = int(court_height * (DOUBLES_COURT_WIDTH_FT / COURT_LENGTH_FT))

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

    net_y = 0.5
    cv2.line(frame, to_px(Point2D(0.0, net_y)), to_px(Point2D(1.0, net_y)), line_color, 1)
    for x in (SINGLES_LEFT_X, SINGLES_RIGHT_X):
        cv2.line(frame, to_px(Point2D(x, 0.0)), to_px(Point2D(x, 1.0)), line_color, 1)
    for y in (FAR_SERVICE_Y, NEAR_SERVICE_Y):
        start = to_px(Point2D(SINGLES_LEFT_X, y))
        end = to_px(Point2D(SINGLES_RIGHT_X, y))
        cv2.line(frame, start, end, line_color, 1)
    cv2.line(frame, to_px(Point2D(COURT_CENTER_X, FAR_SERVICE_Y)), to_px(Point2D(COURT_CENTER_X, NEAR_SERVICE_Y)), line_color, 1)

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


def _draw_calibrated_court(frame, homography: list[list[float]], cv2) -> None:
    if len(homography) != 3:
        return
    _, np = require_vision_stack()
    try:
        normalized_to_image = np.linalg.inv(np.array(homography, dtype=np.float64))
    except np.linalg.LinAlgError:
        return
    overlay = frame.copy()
    outer = [Point2D(0.0, 0.0), Point2D(1.0, 0.0), Point2D(1.0, 1.0), Point2D(0.0, 1.0)]
    for start, end in zip(outer, outer[1:] + outer[:1]):
        _draw_projected_line(overlay, start, end, normalized_to_image, cv2, thickness=3)
    for x in (SINGLES_LEFT_X, SINGLES_RIGHT_X):
        _draw_projected_line(overlay, Point2D(x, 0.0), Point2D(x, 1.0), normalized_to_image, cv2, thickness=2)
    _draw_projected_line(overlay, Point2D(0.0, 0.5), Point2D(1.0, 0.5), normalized_to_image, cv2, thickness=2)
    for y in (FAR_SERVICE_Y, NEAR_SERVICE_Y):
        _draw_projected_line(overlay, Point2D(SINGLES_LEFT_X, y), Point2D(SINGLES_RIGHT_X, y), normalized_to_image, cv2, thickness=2)
    _draw_projected_line(
        overlay,
        Point2D(COURT_CENTER_X, FAR_SERVICE_Y),
        Point2D(COURT_CENTER_X, NEAR_SERVICE_Y),
        normalized_to_image,
        cv2,
        thickness=2,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)


def _draw_projected_line(frame, start: Point2D, end: Point2D, normalized_to_image, cv2, *, thickness: int) -> None:
    projected_start = _project_normalized_point(start, normalized_to_image)
    projected_end = _project_normalized_point(end, normalized_to_image)
    if projected_start is None or projected_end is None:
        return
    cv2.line(
        frame,
        (int(round(projected_start.x)), int(round(projected_start.y))),
        (int(round(projected_end.x)), int(round(projected_end.y))),
        (255, 80, 0),
        thickness,
    )


def _project_normalized_point(point: Point2D, normalized_to_image) -> Point2D | None:
    denominator = (
        normalized_to_image[2][0] * point.x
        + normalized_to_image[2][1] * point.y
        + normalized_to_image[2][2]
    )
    if abs(denominator) <= 1e-8:
        return None
    x = (
        normalized_to_image[0][0] * point.x
        + normalized_to_image[0][1] * point.y
        + normalized_to_image[0][2]
    ) / denominator
    y = (
        normalized_to_image[1][0] * point.x
        + normalized_to_image[1][1] * point.y
        + normalized_to_image[1][2]
    ) / denominator
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return Point2D(float(x), float(y))


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
