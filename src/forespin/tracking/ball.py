from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from forespin.deps import load_optional_module, require_vision_stack
from forespin.domain import BallTrack, Point2D
from forespin.model_weights import MissingModelWeightsError


class BallTrackerProtocol:
    def track(self, frame: Any) -> BallTrack:
        raise NotImplementedError


class TrackNetV2BallTracker(BallTrackerProtocol):
    def __init__(self, weights_path: str) -> None:
        self.weights_path = Path(weights_path)
        self._torch = load_optional_module("torch")
        self._model: Any | None = None
        self._frame_history: deque[Any] = deque(maxlen=3)

    def track(self, frame: Any) -> BallTrack:
        if self._torch is None:
            raise RuntimeError("Torch is required for TrackNetV2 inference.")
        cv2, np = require_vision_stack()
        model = self._load_model()
        resized = cv2.resize(frame, (640, 360))
        self._frame_history.append(resized)
        if len(self._frame_history) < 3:
            return BallTrack(position_px=None, confidence=0.0)

        stacked = np.concatenate(list(self._frame_history), axis=2)
        tensor = self._torch.from_numpy(stacked).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with self._torch.no_grad():
            output = model(tensor)
        heatmap = output.squeeze()
        confidence = float(heatmap.max().item())
        if confidence <= 0.05:
            return BallTrack(position_px=None, confidence=confidence)
        flat_index = int(heatmap.argmax().item())
        height, width = heatmap.shape[-2], heatmap.shape[-1]
        heatmap_y = flat_index // width
        heatmap_x = flat_index % width
        center = Point2D(
            (heatmap_x / max(1, width - 1)) * frame.shape[1],
            (heatmap_y / max(1, height - 1)) * frame.shape[0],
        )
        return BallTrack(position_px=center, confidence=confidence)

    def _load_model(self) -> Any:
        if self._model is None:
            if not self.weights_path.exists():
                raise FileNotFoundError(f"TrackNetV2 weights not found: {self.weights_path}")
            self._model = self._torch.jit.load(str(self.weights_path), map_location="cpu")
            self._model.eval()
        return self._model


def create_ball_tracker(weights_path: str | None) -> BallTrackerProtocol:
    if not weights_path:
        raise MissingModelWeightsError("TrackNetV2 weights are required.")
    return TrackNetV2BallTracker(weights_path)
