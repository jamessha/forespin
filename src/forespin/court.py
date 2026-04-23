from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol

from forespin.config import Thresholds
from forespin.court_reference import CourtReference
from forespin.deps import load_optional_module, require_vision_stack
from forespin.domain import CourtCalibration, Point2D


class CourtCalibrationError(RuntimeError):
    pass


class CourtCalibrationBackend(Protocol):
    def evaluate_frame(self, frame: Any) -> dict[str, Any]:
        ...

    def write_debug_artifacts(
        self,
        debug_dir: Path,
        frame_index: int,
        frame: Any,
        evaluation: dict[str, Any],
    ) -> None:
        ...


class CourtCalibrator:
    def __init__(self, thresholds: Thresholds, weights_path: str | None = None) -> None:
        self.thresholds = thresholds
        if weights_path:
            self.backend: CourtCalibrationBackend = LearnedCourtCalibratorBackend(
                thresholds=thresholds,
                weights_path=weights_path,
                plausibility_check=self._corners_plausible_for_baseline_view,
            )
        else:
            self.backend = HeuristicCourtCalibratorBackend(
                thresholds=thresholds,
                plausibility_check=self._corners_plausible_for_baseline_view,
            )

    def calibrate_video(
        self,
        video_path: str | Path,
        max_scan_frames: int = 30,
        debug_dir: str | Path | None = None,
    ) -> CourtCalibration:
        cv2, _ = require_vision_stack()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise CourtCalibrationError(f"Unable to open video: {video_path}")
        debug_path = Path(debug_dir).expanduser().resolve() if debug_dir else None
        debug_summaries: list[dict[str, Any]] = []
        try:
            scanned = 0
            while scanned < max_scan_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                evaluation = self.backend.evaluate_frame(frame)
                if debug_path is not None:
                    self.backend.write_debug_artifacts(debug_path, scanned, frame, evaluation)
                    debug_summaries.append(evaluation["summary"])
                scanned += 1
                if evaluation["calibration"] is not None:
                    if debug_path is not None:
                        self._write_debug_summary(debug_path, video_path, debug_summaries, selected_frame=scanned - 1)
                    return evaluation["calibration"]
        finally:
            capture.release()
        message = "Unable to find a usable baseline-view court calibration in the opening frames."
        if debug_path is not None:
            self._write_debug_summary(debug_path, video_path, debug_summaries, selected_frame=None)
            message += f" Debug artifacts saved to {debug_path}."
        raise CourtCalibrationError(message)

    def maybe_refresh(self, frame_index: int, frame: Any, current: CourtCalibration) -> CourtCalibration:
        if frame_index == 0 or frame_index % self.thresholds.refresh_court_every_n_frames != 0:
            return current
        try:
            candidate = self.calibrate_frame(frame)
        except CourtCalibrationError:
            return current
        if candidate.confidence >= current.confidence * 0.8:
            return candidate
        return current

    def calibrate_frame(self, frame: Any) -> CourtCalibration:
        evaluation = self.backend.evaluate_frame(frame)
        calibration = evaluation["calibration"]
        if calibration is None:
            raise CourtCalibrationError(evaluation["summary"]["reason"])
        return calibration

    @staticmethod
    def _write_debug_summary(
        debug_dir: Path,
        video_path: str | Path,
        summaries: list[dict[str, Any]],
        selected_frame: int | None,
    ) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_path": str(video_path),
            "selected_frame": selected_frame,
            "frames": [
                {
                    "frame_index": index,
                    **summary,
                }
                for index, summary in enumerate(summaries)
            ],
        }
        (debug_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    @staticmethod
    def _corners_plausible_for_baseline_view(corners: list[Point2D], width: int, height: int) -> bool:
        if len(corners) != 4:
            return False

        top_left, top_right, bottom_right, bottom_left = corners
        if not (top_left.x < top_right.x and bottom_left.x < bottom_right.x):
            return False

        top_y_limit_low = -height * 0.35
        top_y_limit_high = height * 0.75
        bottom_y_limit_low = height * 0.20
        bottom_y_limit_high = height * 2.75
        wide_left_limit = -width * 1.75
        wide_right_limit = width * 2.75

        top_corners = (top_left, top_right)
        bottom_corners = (bottom_left, bottom_right)

        for corner in top_corners:
            if corner.x < wide_left_limit or corner.x > wide_right_limit:
                return False
            if corner.y < top_y_limit_low or corner.y > top_y_limit_high:
                return False

        for corner in bottom_corners:
            if corner.x < wide_left_limit or corner.x > wide_right_limit:
                return False
            if corner.y < bottom_y_limit_low or corner.y > bottom_y_limit_high:
                return False

        top_width = top_right.x - top_left.x
        bottom_width = bottom_right.x - bottom_left.x
        left_height = bottom_left.y - top_left.y
        right_height = bottom_right.y - top_right.y

        if top_width <= width * 0.10:
            return False
        if bottom_width <= top_width * 0.80:
            return False
        if left_height <= height * 0.15 or right_height <= height * 0.15:
            return False

        return True


class LearnedCourtCalibratorBackend:
    MODEL_INPUT_WIDTH = 640
    MODEL_INPUT_HEIGHT = 360
    OUTPUT_KEYPOINTS = 14
    OUTPUT_CHANNELS = 15
    MIN_VISIBLE_KEYPOINTS = 4
    MIN_PEAK_CONFIDENCE = 0.15

    def __init__(
        self,
        *,
        thresholds: Thresholds,
        weights_path: str,
        plausibility_check: Any,
    ) -> None:
        self.thresholds = thresholds
        self.weights_path = Path(weights_path)
        self._plausibility_check = plausibility_check
        self._torch = load_optional_module("torch")
        self._model: Any | None = None
        self.reference = CourtReference()

    def evaluate_frame(self, frame: Any) -> dict[str, Any]:
        cv2, np = require_vision_stack()
        heatmaps = self._predict_heatmaps(frame, np)
        visible_keypoints, confidences = self._decode_keypoints(
            heatmaps=heatmaps,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            np=np,
        )
        visible_count = sum(point is not None for point in visible_keypoints)
        mean_visible_confidence = (
            sum(confidence for point, confidence in zip(visible_keypoints, confidences) if point is not None) / visible_count
            if visible_count
            else 0.0
        )

        calibration: CourtCalibration | None = None
        reason: str | None = None
        reference_to_image: Any | None = None
        reconstructed_keypoints: list[Point2D | None] = [None] * self.OUTPUT_KEYPOINTS
        corners: list[Point2D | None] = []
        reprojection_error_px: float | None = None

        if visible_count < self.MIN_VISIBLE_KEYPOINTS:
            reason = "The learned calibrator did not find enough court keypoints."
        else:
            best_fit = self._fit_reference_to_image(visible_keypoints, np, cv2)
            if best_fit is None:
                reason = "Unable to fit a court homography from the detected keypoints."
            else:
                reference_to_image, reprojection_error_px = best_fit
                reconstructed_keypoints = self._project_reference_points(reference_to_image, self.reference.key_points, np, cv2)
                corners = self._project_reference_points(reference_to_image, self.reference.border_points, np, cv2)
                if any(corner is None for corner in corners):
                    reason = "The learned homography could not reconstruct the outer court corners."
                elif not self._plausibility_check(corners, frame.shape[1], frame.shape[0]):
                    reason = "The reconstructed court corners were implausible for a baseline-view court."
                else:
                    confidence = self._confidence(
                        visible_count=visible_count,
                        mean_visible_confidence=mean_visible_confidence,
                        reprojection_error_px=reprojection_error_px,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    )
                    if confidence < self.thresholds.min_court_confidence:
                        reason = "The learned court calibration confidence fell below the acceptance threshold."
                    else:
                        try:
                            homography = self._image_to_normalized_homography(reference_to_image, cv2, np).tolist()
                        except np.linalg.LinAlgError:
                            reason = "The learned court homography was numerically unstable."
                        else:
                            calibration = CourtCalibration(
                                corners_px=[corner for corner in corners if corner is not None],
                                homography=homography,
                                confidence=confidence,
                                source="learned_court",
                            )

        summary = {
            "mode": "learned",
            "reason": reason or "accepted",
            "visible_keypoints": visible_count,
            "mean_visible_keypoint_confidence": mean_visible_confidence,
            "reprojection_error_px": reprojection_error_px,
            "confidence": calibration.confidence if calibration else 0.0,
            "corners": [self._point_summary(point) for point in corners],
            "keypoints": [
                {
                    "index": index,
                    "visible": point is not None,
                    "confidence": confidences[index],
                    "predicted": self._point_summary(point),
                    "reconstructed": self._point_summary(reconstructed_keypoints[index]),
                }
                for index, point in enumerate(visible_keypoints)
            ],
        }
        return {
            "calibration": calibration,
            "heatmaps": heatmaps,
            "predicted_keypoints": visible_keypoints,
            "reconstructed_keypoints": reconstructed_keypoints,
            "corners": corners,
            "reference_to_image": reference_to_image.tolist() if reference_to_image is not None else None,
            "summary": summary,
        }

    def write_debug_artifacts(
        self,
        debug_dir: Path,
        frame_index: int,
        frame: Any,
        evaluation: dict[str, Any],
    ) -> None:
        cv2, _ = require_vision_stack()
        debug_dir.mkdir(parents=True, exist_ok=True)
        raw_path = debug_dir / f"frame_{frame_index:03d}_raw.jpg"
        overlay_path = debug_dir / f"frame_{frame_index:03d}_overlay.jpg"
        cv2.imwrite(str(raw_path), frame)

        overlay = frame.copy()
        predicted = evaluation["predicted_keypoints"]
        reconstructed = evaluation["reconstructed_keypoints"]
        corners = evaluation["corners"]

        for point in reconstructed:
            if point is None:
                continue
            cv2.circle(overlay, (int(point.x), int(point.y)), 5, (255, 120, 0), -1)
        for index, point in enumerate(predicted):
            if point is None:
                continue
            cv2.circle(overlay, (int(point.x), int(point.y)), 8, (0, 255, 0), -1)
            cv2.putText(
                overlay,
                f"K{index}",
                (int(point.x) + 6, int(point.y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                2,
            )

        if len(corners) == 4 and all(point is not None for point in corners):
            ordered = [point for point in corners if point is not None]
            for start, end in zip(ordered, ordered[1:] + ordered[:1]):
                cv2.line(
                    overlay,
                    (int(start.x), int(start.y)),
                    (int(end.x), int(end.y)),
                    (255, 0, 0),
                    3,
                )

        cv2.putText(
            overlay,
            evaluation["summary"]["reason"],
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            (
                f"visible={evaluation['summary']['visible_keypoints']} "
                f"confidence={evaluation['summary']['confidence']:.2f}"
            ),
            (18, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(str(overlay_path), overlay)

    def _predict_heatmaps(self, frame: Any, np: Any) -> Any:
        if self._torch is None:
            raise RuntimeError("Torch is required for learned court calibration.")
        model = self._load_model()
        cv2, _ = require_vision_stack()
        resized = cv2.resize(frame, (self.MODEL_INPUT_WIDTH, self.MODEL_INPUT_HEIGHT))
        tensor = self._torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0)
        with self._torch.no_grad():
            output = model(tensor.float())[0]
        return self._torch.sigmoid(output[: self.OUTPUT_KEYPOINTS]).detach().cpu().numpy()

    def _load_model(self) -> Any:
        if self._torch is None:
            raise RuntimeError("Torch is required for learned court calibration.")
        if self._model is not None:
            return self._model
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Learned court detector weights not found: {self.weights_path}")

        nn = self._torch.nn

        class ConvBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pad: int = 1, stride: int = 1, bias: bool = True) -> None:
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                )

            def forward(self, x: Any) -> Any:
                return self.block(x)

        class BallTrackerNet(nn.Module):
            def __init__(self, out_channels: int = 14) -> None:
                super().__init__()
                self.out_channels = out_channels
                self.conv1 = ConvBlock(in_channels=3, out_channels=64)
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
                self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

            def forward(self, x: Any) -> Any:
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
                return self.conv18(x)

        model = BallTrackerNet(out_channels=self.OUTPUT_CHANNELS)
        loaded = self._torch.load(str(self.weights_path), map_location="cpu")
        if isinstance(loaded, dict) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            loaded = loaded["model_state_dict"]
        if isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], dict):
            loaded = loaded["model"]
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Unsupported learned court detector checkpoint format: {self.weights_path}")
        state_dict = {
            key.removeprefix("module."): value
            for key, value in loaded.items()
            if not key.startswith("optimizer")
        }
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self._model = model
        return model

    def _decode_keypoints(
        self,
        *,
        heatmaps: Any,
        frame_width: int,
        frame_height: int,
        np: Any,
    ) -> tuple[list[Point2D | None], list[float]]:
        keypoints: list[Point2D | None] = []
        confidences: list[float] = []
        for heatmap in heatmaps:
            peak = float(heatmap.max())
            confidences.append(peak)
            if peak < self.MIN_PEAK_CONFIDENCE:
                keypoints.append(None)
                continue
            mask = heatmap >= max(self.MIN_PEAK_CONFIDENCE, peak * 0.55)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                flat_index = int(heatmap.argmax())
                heatmap_width = int(heatmap.shape[1])
                y_center = flat_index // heatmap_width
                x_center = flat_index % heatmap_width
            else:
                weights = heatmap[ys, xs]
                total = float(weights.sum()) or 1.0
                x_center = float((xs * weights).sum() / total)
                y_center = float((ys * weights).sum() / total)
            keypoints.append(
                Point2D(
                    x=(x_center / max(1, int(heatmap.shape[1]) - 1)) * (frame_width - 1),
                    y=(y_center / max(1, int(heatmap.shape[0]) - 1)) * (frame_height - 1),
                )
            )
        return keypoints, confidences

    def _fit_reference_to_image(self, points: list[Point2D | None], np: Any, cv2: Any) -> tuple[Any, float] | None:
        visible_indices = [index for index, point in enumerate(points) if point is not None]
        if len(visible_indices) < self.MIN_VISIBLE_KEYPOINTS:
            return None

        best_matrix: Any | None = None
        best_error = math.inf
        for indices in self.reference.configuration_indices():
            if any(points[index] is None for index in indices):
                continue
            source = np.array([self.reference.key_points[index] for index in indices], dtype=np.float32)
            destination = np.array(
                [[points[index].x, points[index].y] for index in indices if points[index] is not None],
                dtype=np.float32,
            )
            matrix, _ = cv2.findHomography(source, destination, method=0)
            if matrix is None:
                continue
            projected = self._project_reference_points(matrix, self.reference.key_points, np, cv2)
            errors: list[float] = []
            for index in visible_indices:
                observed = points[index]
                predicted = projected[index]
                if observed is None or predicted is None:
                    continue
                errors.append(math.hypot(observed.x - predicted.x, observed.y - predicted.y))
            if not errors:
                continue
            mean_error = sum(errors) / len(errors)
            if mean_error < best_error:
                best_matrix = matrix
                best_error = mean_error

        if best_matrix is None:
            return None
        return best_matrix, best_error

    @staticmethod
    def _project_reference_points(matrix: Any, reference_points: list[tuple[float, float]], np: Any, cv2: Any) -> list[Point2D | None]:
        projected = cv2.perspectiveTransform(np.array(reference_points, dtype=np.float32).reshape((-1, 1, 2)), matrix)
        return [Point2D(float(point[0][0]), float(point[0][1])) for point in projected]

    def _image_to_normalized_homography(self, reference_to_image: Any, cv2: Any, np: Any) -> Any:
        reference_border = np.array(self.reference.border_points, dtype=np.float32)
        normalized_border = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        reference_to_normalized = cv2.getPerspectiveTransform(reference_border, normalized_border)
        image_to_reference = np.linalg.inv(reference_to_image)
        image_to_normalized = reference_to_normalized @ image_to_reference
        scale = image_to_normalized[2][2] if abs(image_to_normalized[2][2]) > 1e-8 else 1.0
        return image_to_normalized / scale

    @staticmethod
    def _confidence(
        *,
        visible_count: int,
        mean_visible_confidence: float,
        reprojection_error_px: float | None,
        frame_width: int,
        frame_height: int,
    ) -> float:
        visible_ratio = min(1.0, visible_count / 14.0)
        scale = max(frame_width, frame_height)
        if reprojection_error_px is None:
            consistency = 0.0
        else:
            consistency = max(0.0, 1.0 - (reprojection_error_px / max(scale * 0.08, 1.0)))
        return max(
            0.0,
            min(
                1.0,
                (visible_ratio * 0.45) + (mean_visible_confidence * 0.35) + (consistency * 0.20),
            ),
        )

    @staticmethod
    def _point_summary(point: Point2D | None) -> dict[str, float] | None:
        if point is None:
            return None
        return {"x": point.x, "y": point.y}


class HeuristicCourtCalibratorBackend:
    def __init__(self, *, thresholds: Thresholds, plausibility_check: Any) -> None:
        self.thresholds = thresholds
        self._plausibility_check = plausibility_check

    def evaluate_frame(self, frame: Any) -> dict[str, Any]:
        cv2, np = require_vision_stack()
        mask = self._court_line_mask(frame, cv2)
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=60,
            minLineLength=max(30, int(frame.shape[1] * 0.16)),
            maxLineGap=20,
        )
        candidates = [self._segment_from_line(line[0]) for line in lines] if lines is not None else []
        top = self._pick_horizontal(candidates, top=True)
        bottom = self._pick_horizontal(candidates, top=False)
        left = self._pick_side(candidates, left=True, frame_width=frame.shape[1])
        right = self._pick_side(candidates, left=False, frame_width=frame.shape[1])

        corners: list[Point2D | None] = []
        if all([top, bottom, left, right]):
            corners = [
                self._intersection(top, left),
                self._intersection(top, right),
                self._intersection(bottom, right),
                self._intersection(bottom, left),
            ]
        valid_corners = [corner for corner in corners if corner is not None]
        confidence = self._confidence(top, bottom, left, right, frame.shape[1], frame.shape[0]) if all([top, bottom, left, right]) else 0.0

        reason: str | None = None
        calibration: CourtCalibration | None = None
        if lines is None or len(lines) < 4:
            reason = "Not enough candidate court lines."
        elif not all([top, bottom, left, right]):
            reason = "Could not isolate the court boundaries."
        elif any(corner is None for corner in corners):
            reason = "Failed to intersect court boundaries into corners."
        elif not self._plausibility_check(valid_corners, frame.shape[1], frame.shape[0]):
            reason = "Detected court corners were implausible for a baseline-view court."
        elif confidence < self.thresholds.min_court_confidence:
            reason = "Court detection confidence fell below the acceptance threshold."
        else:
            src = np.array([[corner.x, corner.y] for corner in valid_corners], dtype=np.float32)
            dst = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
            homography = cv2.getPerspectiveTransform(src, dst)
            calibration = CourtCalibration(
                corners_px=valid_corners,
                homography=homography.tolist(),
                confidence=confidence,
                source="auto_hough",
            )

        summary = {
            "mode": "hough",
            "reason": reason or "accepted",
            "line_count": len(candidates),
            "selected_lines": {
                "top": self._line_summary(top),
                "bottom": self._line_summary(bottom),
                "left": self._line_summary(left),
                "right": self._line_summary(right),
            },
            "corners": [self._point_summary(corner) for corner in corners],
            "confidence": confidence,
        }
        return {
            "calibration": calibration,
            "mask": mask,
            "edges": edges,
            "candidates": candidates,
            "selected_lines": {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
            },
            "corners": corners,
            "summary": summary,
        }

    def write_debug_artifacts(
        self,
        debug_dir: Path,
        frame_index: int,
        frame: Any,
        evaluation: dict[str, Any],
    ) -> None:
        cv2, _ = require_vision_stack()
        debug_dir.mkdir(parents=True, exist_ok=True)
        raw_path = debug_dir / f"frame_{frame_index:03d}_raw.jpg"
        mask_path = debug_dir / f"frame_{frame_index:03d}_mask.png"
        edges_path = debug_dir / f"frame_{frame_index:03d}_edges.png"
        lines_path = debug_dir / f"frame_{frame_index:03d}_lines.jpg"
        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(str(mask_path), evaluation["mask"])
        cv2.imwrite(str(edges_path), evaluation["edges"])
        overlay = frame.copy()
        for segment in evaluation["candidates"]:
            self._draw_segment(overlay, segment, color=(180, 180, 255), thickness=1)
        selected_colors = {
            "top": (0, 255, 255),
            "bottom": (0, 180, 255),
            "left": (0, 255, 0),
            "right": (255, 0, 0),
        }
        for name, segment in evaluation["selected_lines"].items():
            if segment is None:
                continue
            self._draw_segment(overlay, segment, color=selected_colors[name], thickness=3)
        for index, corner in enumerate(evaluation["corners"]):
            if corner is None:
                continue
            cv2.circle(overlay, (int(corner.x), int(corner.y)), 8, (255, 255, 0), -1)
            cv2.putText(
                overlay,
                f"C{index}",
                (int(corner.x) + 6, int(corner.y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
        cv2.putText(
            overlay,
            evaluation["summary"]["reason"],
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"lines={evaluation['summary']['line_count']} confidence={evaluation['summary']['confidence']:.2f}",
            (18, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(str(lines_path), overlay)

    @staticmethod
    def _court_line_mask(frame: Any, cv2: Any) -> Any:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = (0, 0, 150)
        upper = (180, 70, 255)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    @staticmethod
    def _segment_from_line(line: list[int]) -> dict[str, float]:
        x1, y1, x2, y2 = [float(value) for value in line]
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dy, dx))
        length = math.hypot(dx, dy)
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "dx": dx,
            "dy": dy,
            "angle": angle,
            "length": length,
            "center_x": (x1 + x2) / 2.0,
            "center_y": (y1 + y2) / 2.0,
        }

    @staticmethod
    def _pick_horizontal(segments: list[dict[str, float]], top: bool) -> dict[str, float] | None:
        horizontal = [segment for segment in segments if abs(segment["angle"]) <= 25.0]
        if not horizontal:
            return None
        key = (lambda item: (item["center_y"], -item["length"])) if top else (lambda item: (-item["center_y"], -item["length"]))
        return sorted(horizontal, key=key)[0]

    @staticmethod
    def _pick_side(segments: list[dict[str, float]], left: bool, frame_width: int) -> dict[str, float] | None:
        side_segments = [
            segment
            for segment in segments
            if abs(segment["angle"]) > 20.0 and abs(segment["angle"]) < 85.0
        ]
        if not side_segments:
            return None
        if left:
            candidates = [segment for segment in side_segments if segment["center_x"] < frame_width / 2.0]
            if not candidates:
                return None
            return sorted(candidates, key=lambda item: (item["center_x"], -item["length"]))[0]
        candidates = [segment for segment in side_segments if segment["center_x"] >= frame_width / 2.0]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (-item["center_x"], -item["length"]))[0]

    @staticmethod
    def _intersection(first: dict[str, float], second: dict[str, float]) -> Point2D | None:
        denominator = (first["x1"] - first["x2"]) * (second["y1"] - second["y2"]) - (first["y1"] - first["y2"]) * (second["x1"] - second["x2"])
        if abs(denominator) < 1e-6:
            return None
        numerator_x = ((first["x1"] * first["y2"] - first["y1"] * first["x2"]) * (second["x1"] - second["x2"])) - (
            (first["x1"] - first["x2"]) * (second["x1"] * second["y2"] - second["y1"] * second["x2"])
        )
        numerator_y = ((first["x1"] * first["y2"] - first["y1"] * first["x2"]) * (second["y1"] - second["y2"])) - (
            (first["y1"] - first["y2"]) * (second["x1"] * second["y2"] - second["y1"] * second["x2"])
        )
        return Point2D(numerator_x / denominator, numerator_y / denominator)

    @staticmethod
    def _confidence(
        top: dict[str, float],
        bottom: dict[str, float],
        left: dict[str, float],
        right: dict[str, float],
        width: int,
        height: int,
    ) -> float:
        horizontal_coverage = min(1.0, (top["length"] + bottom["length"]) / (width * 1.4))
        vertical_coverage = min(1.0, (left["length"] + right["length"]) / (height * 1.4))
        angle_separation = min(1.0, abs(left["angle"] - right["angle"]) / 120.0)
        return max(0.0, min(1.0, (horizontal_coverage * 0.4) + (vertical_coverage * 0.4) + (angle_separation * 0.2)))

    @staticmethod
    def _draw_segment(frame: Any, segment: dict[str, float], color: tuple[int, int, int], thickness: int) -> None:
        cv2, _ = require_vision_stack()
        cv2.line(
            frame,
            (int(segment["x1"]), int(segment["y1"])),
            (int(segment["x2"]), int(segment["y2"])),
            color,
            thickness,
        )

    @staticmethod
    def _point_summary(point: Point2D | None) -> dict[str, float] | None:
        if point is None:
            return None
        return {"x": point.x, "y": point.y}

    @staticmethod
    def _line_summary(segment: dict[str, float] | None) -> dict[str, float] | None:
        if segment is None:
            return None
        return {
            "x1": segment["x1"],
            "y1": segment["y1"],
            "x2": segment["x2"],
            "y2": segment["y2"],
            "angle": segment["angle"],
            "length": segment["length"],
            "center_x": segment["center_x"],
            "center_y": segment["center_y"],
        }
