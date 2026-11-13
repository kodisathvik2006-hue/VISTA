from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.services.detector import YoloDetector, set_detector
from tests.conftest import write_sample_video


def test_real_yolo_detector_direct_inference() -> None:
    """Real inference test: directly tests YOLOv8 model loading and inference on synthetic image."""
    settings = get_settings()
    detector = YoloDetector(model_path="yolov8n.pt", device="cpu")

    # Verify classes
    classes = detector.get_supported_classes()
    assert len(classes) == 80
    assert classes[0] == "person"
    assert classes[2] == "car"

    # Synthetic test image
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)

    # Real inference execution
    detections = detector.track_frame(img, persist=True, conf=0.1)
    assert isinstance(detections, list)


def test_real_yolo_bytetrack_job_pipeline(client: TestClient, tmp_path: Path) -> None:
    """Real end-to-end inference and tracking job test using real YOLOv8 and ByteTrack."""
    # Ensure no mock detector is active
    set_detector(None)

    # Generate a short sample video
    video_path = write_sample_video(tmp_path / "real_sample.mp4", frames=6, width=160, height=120, fps=5)

    with video_path.open("rb") as handle:
        upload_resp = client.post(
            "/api/videos/analyze",
            files={"file": ("real_sample.mp4", handle, "video/mp4")},
        )
    assert upload_resp.status_code == 200
    video_id = upload_resp.json()["id"]

    # Submit real AI tracking job with low threshold for synthetic video
    job_resp = client.post(
        "/api/jobs",
        json={
            "video_id": video_id,
            "confidence_threshold": 0.1,
            "save_annotated_video": True,
            "max_frames": 6,
        },
    )
    assert job_resp.status_code == 200
    job_id = job_resp.json()["job_id"]

    # Poll until completed
    for _ in range(60):
        status_resp = client.get(f"/api/jobs/{job_id}")
        assert status_resp.status_code == 200
        if status_resp.json()["status"] in {"completed", "failed"}:
            break
        time.sleep(0.2)

    status_data = client.get(f"/api/jobs/{job_id}").json()
    assert status_data["status"] == "completed", f"Job failed: {status_data.get('error')}"
    assert status_data["progress"] == 100.0
    assert status_data["current_frame"] == 6

    # Verify results endpoint
    results = client.get(f"/api/jobs/{job_id}/results").json()
    assert len(results["frames"]) == 6
    assert "summary" in results

    # Verify annotated video was generated
    video_file_resp = client.get(f"/api/jobs/{job_id}/annotated-video")
    assert video_file_resp.status_code == 200
    assert video_file_resp.headers["content-type"] == "video/mp4"
    assert len(video_file_resp.content) > 0
