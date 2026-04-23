from __future__ import annotations

from pathlib import Path
from typing import Any

from forespin.deps import load_optional_module
from forespin.domain import BBox, PlayerTrack, Point2D, TrackedPlayerSide
from forespin.model_weights import MissingModelWeightsError


class PlayerTrackerProtocol:
    def track(self, frame: Any) -> PlayerTrack:
        raise NotImplementedError


class YOLO26PosePlayerTracker(PlayerTrackerProtocol):
    def __init__(self, tracked_player_side: TrackedPlayerSide, weights_path: str) -> None:
        self.tracked_player_side = tracked_player_side
        self.weights_path = Path(weights_path)
        self._ultralytics = load_optional_module("ultralytics")
        self._model: Any | None = None

    def track(self, frame: Any) -> PlayerTrack:
        if self._ultralytics is None:
            raise RuntimeError("Ultralytics is required for YOLO26 pose inference.")
        model = self._load_model()
        results = model.predict(frame, verbose=False, conf=0.2, classes=[0])
        if not results:
            return PlayerTrack(bbox_px=None, feet_px=None, torso_px=None, confidence=0.0)
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return PlayerTrack(bbox_px=None, feet_px=None, torso_px=None, confidence=0.0)

        candidates: list[tuple[float, PlayerTrack]] = []
        frame_mid_y = frame.shape[0] / 2.0
        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        keypoints = result.keypoints.xy.cpu().tolist() if result.keypoints is not None else [None] * len(boxes)
        for box_values, confidence, person_keypoints in zip(boxes, confidences, keypoints):
            bbox = BBox(*[float(value) for value in box_values])
            feet = self._feet_anchor(person_keypoints, bbox)
            if self.tracked_player_side == TrackedPlayerSide.NEAR and feet.y < frame_mid_y:
                continue
            if self.tracked_player_side == TrackedPlayerSide.FAR and feet.y >= frame_mid_y:
                continue
            torso = self._torso_anchor(person_keypoints, bbox)
            candidates.append(
                (
                    float(confidence),
                    PlayerTrack(bbox_px=bbox, feet_px=feet, torso_px=torso, confidence=float(confidence)),
                )
            )

        if not candidates:
            return PlayerTrack(bbox_px=None, feet_px=None, torso_px=None, confidence=0.0)

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _load_model(self) -> Any:
        if self._model is None:
            if not self.weights_path.exists():
                raise FileNotFoundError(f"YOLO26 pose weights not found: {self.weights_path}")
            self._model = self._ultralytics.YOLO(str(self.weights_path))
        return self._model

    @staticmethod
    def _feet_anchor(keypoints: Any, bbox: BBox) -> Point2D:
        if not keypoints:
            return bbox.bottom_center
        ankles = [point for index, point in enumerate(keypoints) if index in (15, 16) and point[0] > 0 and point[1] > 0]
        if not ankles:
            return bbox.bottom_center
        return Point2D(sum(point[0] for point in ankles) / len(ankles), max(point[1] for point in ankles))

    @staticmethod
    def _torso_anchor(keypoints: Any, bbox: BBox) -> Point2D:
        if not keypoints:
            return bbox.center
        torso_points = [point for index, point in enumerate(keypoints) if index in (5, 6, 11, 12) and point[0] > 0 and point[1] > 0]
        if not torso_points:
            return bbox.center
        return Point2D(
            sum(point[0] for point in torso_points) / len(torso_points),
            sum(point[1] for point in torso_points) / len(torso_points),
        )


def create_player_tracker(tracked_player_side: TrackedPlayerSide, weights_path: str | None) -> PlayerTrackerProtocol:
    if not weights_path:
        raise MissingModelWeightsError("YOLO26 pose weights are required.")
    return YOLO26PosePlayerTracker(tracked_player_side=tracked_player_side, weights_path=weights_path)
