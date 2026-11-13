from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.exceptions import bad_request


@dataclass(frozen=True)
class VideoProbe:
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    preview_bgr: np.ndarray


def _open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise bad_request(
            "UNREADABLE_VIDEO",
            "The file could not be opened as a video. Filename and MIME type are not enough; the contents must be a readable video.",
        )
    return capture


def _read_first_frame(capture: cv2.VideoCapture) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        raise bad_request(
            "NO_FRAMES",
            "The file opened but no frames could be read. It is not a usable video.",
        )
    return frame


def probe_video(path: Path) -> VideoProbe:
    capture = _open_capture(path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        preview = _read_first_frame(capture)

        if width <= 0 or height <= 0:
            height, width = preview.shape[:2]

        if fps <= 0:
            fps = 0.0

        if frame_count <= 0:
            frame_count = 1

        duration = frame_count / fps if fps > 0 else 0.0

        return VideoProbe(
            duration_seconds=round(duration, 3),
            fps=round(fps, 3),
            frame_count=frame_count,
            width=int(width),
            height=int(height),
            preview_bgr=preview,
        )
    finally:
        capture.release()


def write_preview_jpeg(frame_bgr: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise bad_request("PREVIEW_FAILED", "A preview frame could not be encoded.")
    dest.write_bytes(encoded.tobytes())
