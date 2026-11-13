from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from app.schemas import TrackedDetection

# Distinct, professional color palette (BGR format for OpenCV)
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (72, 190, 72),      # Emerald Green
    "car": (240, 150, 40),        # Vibrant Blue
    "truck": (210, 100, 30),      # Deep Blue
    "bus": (255, 120, 50),        # Cerulean
    "motorcycle": (20, 180, 240), # Golden Amber
    "bicycle": (50, 210, 200),    # Lime Yellow
    "traffic light": (50, 50, 240),# Crimson
    "stop sign": (0, 0, 220),     # Red
}

DEFAULT_COLOR = (200, 100, 180)  # Purple


def get_color_for_class(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(class_name.lower(), DEFAULT_COLOR)


def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[TrackedDetection],
) -> np.ndarray:
    """Draw bounding boxes, labels, and tracking IDs onto a video frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = get_color_for_class(det.class_name)

        # Draw main bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Build label text: e.g. "#3 car 0.88" or "car 0.88"
        if det.track_id is not None:
            label = f"#{det.track_id} {det.class_name} {det.confidence:.2f}"
        else:
            label = f"{det.class_name} {det.confidence:.2f}"

        # Calculate label size
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, min(0.6, w / 1200))
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)

        # Background pill above bbox
        pill_y2 = max(y1, text_h + 6)
        pill_y1 = max(0, pill_y2 - text_h - 6)
        pill_x1 = x1
        pill_x2 = min(w, x1 + text_w + 8)

        # Filled badge background
        cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)

        # Text in dark or white depending on contrast
        cv2.putText(
            annotated,
            label,
            (pill_x1 + 4, pill_y2 - 4),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def save_evidence_frame(frame: np.ndarray, dest_path: Path) -> None:
    """Save an evidence frame as JPEG."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
