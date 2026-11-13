from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.config import Settings
from app.schemas import TrackedDetection

logger = logging.getLogger("neurovision.detector")


class BaseDetector(Protocol):
    """Abstract protocol for object detection and tracking."""

    def track_frame(
        self,
        frame_bgr: np.ndarray,
        persist: bool = True,
        conf: float = 0.25,
        classes: list[int] | None = None,
    ) -> list[TrackedDetection]: ...

    def get_supported_classes(self) -> dict[int, str]: ...

    def resolve_class_indices(self, class_names: list[str] | None) -> list[int] | None: ...

    def reset_tracker(self) -> None: ...


class YoloDetector:
    """Ultralytics YOLO + ByteTrack detector implementation."""

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu") -> None:
        from ultralytics import YOLO

        self.model_path = model_path
        self.device = device
        logger.info("Initializing YOLO model '%s' on device '%s'", model_path, device)
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception as exc:
            logger.warning("Could not move YOLO model to device '%s': %s. Falling back to default.", device, exc)

        self._names: dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}
        self._name_to_id: dict[str, int] = {v.lower(): k for k, v in self._names.items()}

    def get_supported_classes(self) -> dict[int, str]:
        return dict(self._names)

    def resolve_class_indices(self, class_names: list[str] | None) -> list[int] | None:
        if not class_names:
            return None
        indices: list[int] = []
        for name in class_names:
            clean = name.strip().lower()
            if clean in self._name_to_id:
                indices.append(self._name_to_id[clean])
        return sorted(list(set(indices))) if indices else None

    def reset_tracker(self) -> None:
        """Reset ByteTrack tracker state."""
        try:
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                if hasattr(self.model.predictor, "trackers"):
                    self.model.predictor.trackers = None
        except Exception as exc:
            logger.debug("Failed to reset predictor trackers: %s", exc)

    def track_frame(
        self,
        frame_bgr: np.ndarray,
        persist: bool = True,
        conf: float = 0.25,
        classes: list[int] | None = None,
    ) -> list[TrackedDetection]:
        detections: list[TrackedDetection] = []
        try:
            results = self.model.track(
                source=frame_bgr,
                persist=persist,
                tracker="bytetrack.yaml",
                conf=conf,
                classes=classes,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            logger.warning("YOLO track() raised an exception on frame: %s", exc)
            return detections

        if not results:
            return detections

        result = results[0]
        if result is None:
            return detections

        boxes = result.boxes
        if boxes is None:
            return detections

        # Defensive len() — ultralytics 8.4.x boxes.data can be None on empty frames
        try:
            n = len(boxes)
        except TypeError:
            return detections
        if n == 0:
            return detections

        try:
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.int().cpu().numpy()
            track_ids = boxes.id.int().cpu().numpy() if boxes.id is not None else None
        except Exception as exc:
            logger.warning("Failed to extract box tensors: %s", exc)
            return detections

        for i in range(len(coords)):
            try:
                track_id = int(track_ids[i]) if track_ids is not None else None
                cls_id = int(clss[i])
                cls_name = self._names.get(cls_id, f"class_{cls_id}")
                confidence = float(confs[i])
                box = [float(round(c, 2)) for c in coords[i]]
                detections.append(
                    TrackedDetection(
                        track_id=track_id,
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=round(confidence, 4),
                        bbox=box,
                    )
                )
            except Exception as exc:
                logger.debug("Skipping detection at index %d: %s", i, exc)
                continue

        return detections


_detector_instance: BaseDetector | None = None


def get_detector(settings: Settings) -> BaseDetector:
    global _detector_instance
    if _detector_instance is None:
        model_name = settings.yolo_model
        # Check if model exists in backend root or load by name
        _detector_instance = YoloDetector(
            model_path=model_name,
            device=settings.resolved_device,
        )
    return _detector_instance


def set_detector(detector: BaseDetector | None) -> None:
    """Allows injecting mock detector for testing."""
    global _detector_instance
    _detector_instance = detector
