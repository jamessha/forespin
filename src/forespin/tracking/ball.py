from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forespin.deps import load_optional_module, require_vision_stack
from forespin.domain import BallTrack, Point2D
from forespin.model_weights import MissingModelWeightsError


class BallTrackerProtocol:
    def track(self, frame: Any) -> BallTrack:
        raise NotImplementedError


class TrackNetV2BallTracker(BallTrackerProtocol):
    MODEL_INPUT_WIDTH = 640
    MODEL_INPUT_HEIGHT = 360

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
        resized = cv2.resize(frame, (self.MODEL_INPUT_WIDTH, self.MODEL_INPUT_HEIGHT))
        self._frame_history.append(resized)
        if len(self._frame_history) < 3:
            return BallTrack(position_px=None, confidence=0.0)

        history = list(self._frame_history)
        # yastrebksv/TrackNet trains on [current, previous, pre-previous] RGB frames.
        stacked = np.concatenate([history[-1], history[-2], history[-3]], axis=2)
        tensor = self._torch.from_numpy(stacked).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with self._torch.no_grad():
            output = model(tensor)
        return self._track_from_output(output, frame, cv2, np)

    def _track_from_output(self, output: Any, frame: Any, cv2: Any, np: Any) -> BallTrack:
        output = output.detach().cpu()
        if len(output.shape) == 3 and int(output.shape[1]) == 256:
            return self._track_from_yastrebksv_logits(output, frame, cv2, np)
        return self._track_from_heatmap(output, frame)

    def _track_from_heatmap(self, output: Any, frame: Any) -> BallTrack:
        heatmap = output.squeeze()
        confidence = float(heatmap.max().item())
        if confidence <= 0.05:
            return BallTrack(position_px=None, confidence=confidence)
        flat_index = int(heatmap.argmax().item())
        height, width = int(heatmap.shape[-2]), int(heatmap.shape[-1])
        heatmap_y = flat_index // width
        heatmap_x = flat_index % width
        center = Point2D(
            (heatmap_x / max(1, width - 1)) * frame.shape[1],
            (heatmap_y / max(1, height - 1)) * frame.shape[0],
        )
        return BallTrack(position_px=center, confidence=confidence)

    def _track_from_yastrebksv_logits(self, output: Any, frame: Any, cv2: Any, np: Any) -> BallTrack:
        class_map = output.argmax(dim=1).reshape((self.MODEL_INPUT_HEIGHT, self.MODEL_INPUT_WIDTH)).numpy().astype(np.uint8)
        peak = int(class_map.max())
        confidence = peak / 255.0
        if confidence <= 0.05:
            return BallTrack(position_px=None, confidence=confidence)

        threshold = max(8, int(peak * 0.55))
        mask = class_map >= threshold
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            flat_index = int(class_map.argmax())
            heatmap_x = flat_index % self.MODEL_INPUT_WIDTH
            heatmap_y = flat_index // self.MODEL_INPUT_WIDTH
        else:
            weights = class_map[ys, xs].astype(np.float64)
            total = float(weights.sum()) or 1.0
            heatmap_x = float((xs * weights).sum() / total)
            heatmap_y = float((ys * weights).sum() / total)

        center = Point2D(
            (heatmap_x / max(1, self.MODEL_INPUT_WIDTH - 1)) * frame.shape[1],
            (heatmap_y / max(1, self.MODEL_INPUT_HEIGHT - 1)) * frame.shape[0],
        )
        return BallTrack(position_px=center, confidence=confidence)

    def _load_model(self) -> Any:
        if self._model is None:
            if not self.weights_path.exists():
                raise FileNotFoundError(f"TrackNetV2 weights not found: {self.weights_path}")
            self._model = self._load_torchscript_or_state_dict()
        return self._model

    def _load_torchscript_or_state_dict(self) -> Any:
        jit_error: Exception | None = None
        try:
            model = self._torch.jit.load(str(self.weights_path), map_location="cpu")
            model.eval()
            return model
        except Exception as exc:
            jit_error = exc

        try:
            loaded = self._torch.load(str(self.weights_path), map_location="cpu")
            state_dict = self._extract_state_dict(loaded)
            model = self._build_yastrebksv_tracknet(state_dict)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            return model
        except Exception as state_error:
            raise RuntimeError(
                "Unable to load TrackNet weights as either TorchScript or a yastrebksv/TrackNet state_dict. "
                f"TorchScript error: {jit_error}. state_dict error: {state_error}"
            ) from state_error

    @staticmethod
    def _extract_state_dict(loaded: Any) -> Mapping[str, Any]:
        if isinstance(loaded, Mapping):
            for key in ("state_dict", "model_state_dict"):
                candidate = loaded.get(key)
                if isinstance(candidate, Mapping):
                    return {
                        str(name).removeprefix("module."): value
                        for name, value in candidate.items()
                    }
            if "conv1.block.0.weight" in loaded:
                return {
                    str(name).removeprefix("module."): value
                    for name, value in loaded.items()
                }
        raise RuntimeError("checkpoint does not contain a supported TrackNet state_dict")

    def _build_yastrebksv_tracknet(self, state_dict: Mapping[str, Any]) -> Any:
        first_weight = state_dict.get("conv1.block.0.weight")
        final_weight = state_dict.get("conv18.block.0.weight")
        if first_weight is None or final_weight is None:
            raise RuntimeError("missing TrackNet conv1/conv18 weights")
        in_channels = int(first_weight.shape[1])
        out_channels = int(final_weight.shape[0])
        if in_channels != 9:
            raise RuntimeError(f"expected yastrebksv TrackNet to use 9 input channels, got {in_channels}")
        if out_channels != 256:
            raise RuntimeError(f"expected yastrebksv TrackNet to use 256 output classes, got {out_channels}")

        nn = self._torch.nn

        class ConvBlock(nn.Module):
            def __init__(
                self,
                in_channels: int,
                out_channels: int,
                kernel_size: int = 3,
                pad: int = 1,
                stride: int = 1,
                bias: bool = True,
            ) -> None:
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                )

            def forward(self, x: Any) -> Any:
                return self.block(x)

        class YastrebksvTrackNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = ConvBlock(in_channels=9, out_channels=64)
                self.conv2 = ConvBlock(in_channels=64, out_channels=64)
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv3 = ConvBlock(in_channels=64, out_channels=128)
                self.conv4 = ConvBlock(in_channels=128, out_channels=128)
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv5 = ConvBlock(in_channels=128, out_channels=256)
                self.conv6 = ConvBlock(in_channels=256, out_channels=256)
                self.conv7 = ConvBlock(in_channels=256, out_channels=256)
                self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv8 = ConvBlock(in_channels=256, out_channels=512)
                self.conv9 = ConvBlock(in_channels=512, out_channels=512)
                self.conv10 = ConvBlock(in_channels=512, out_channels=512)
                self.ups1 = nn.Upsample(scale_factor=2)
                self.conv11 = ConvBlock(in_channels=512, out_channels=256)
                self.conv12 = ConvBlock(in_channels=256, out_channels=256)
                self.conv13 = ConvBlock(in_channels=256, out_channels=256)
                self.ups2 = nn.Upsample(scale_factor=2)
                self.conv14 = ConvBlock(in_channels=256, out_channels=128)
                self.conv15 = ConvBlock(in_channels=128, out_channels=128)
                self.ups3 = nn.Upsample(scale_factor=2)
                self.conv16 = ConvBlock(in_channels=128, out_channels=64)
                self.conv17 = ConvBlock(in_channels=64, out_channels=64)
                self.conv18 = ConvBlock(in_channels=64, out_channels=256)

            def forward(self, x: Any) -> Any:
                batch_size = x.size(0)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.pool1(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.pool2(x)
                x = self.conv5(x)
                x = self.conv6(x)
                x = self.conv7(x)
                x = self.pool3(x)
                x = self.conv8(x)
                x = self.conv9(x)
                x = self.conv10(x)
                x = self.ups1(x)
                x = self.conv11(x)
                x = self.conv12(x)
                x = self.conv13(x)
                x = self.ups2(x)
                x = self.conv14(x)
                x = self.conv15(x)
                x = self.ups3(x)
                x = self.conv16(x)
                x = self.conv17(x)
                x = self.conv18(x)
                return x.reshape(batch_size, 256, -1)

        return YastrebksvTrackNet()


def create_ball_tracker(weights_path: str | None) -> BallTrackerProtocol:
    if not weights_path:
        raise MissingModelWeightsError("TrackNetV2 weights are required.")
    return TrackNetV2BallTracker(weights_path)
