from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.schemas import TrackedDetection
from app.services import detector
from app.services.jobs import get_job_manager
from tests.conftest import write_sample_video


class MockDetector:
    """Deterministic mock detector for fast unit testing without PyTorch inference."""

    def __init__(self) -> None:
        self.call_count = 0

    def get_supported_classes(self) -> dict[int, str]:
        return {0: "person", 1: "bicycle", 2: "car", 7: "truck"}

    def resolve_class_indices(self, class_names: list[str] | None) -> list[int] | None:
        if not class_names:
            return None
        mapping = {"person": 0, "bicycle": 1, "car": 2, "truck": 7}
        return [mapping[c.lower()] for c in class_names if c.lower() in mapping]

    def reset_tracker(self) -> None:
        self.call_count = 0

    def track_frame(
        self,
        frame_bgr: np.ndarray,
        persist: bool = True,
        conf: float = 0.25,
        classes: list[int] | None = None,
    ) -> list[TrackedDetection]:
        self.call_count += 1
        # Generate predictable detections
        return [
            TrackedDetection(
                track_id=1,
                class_id=2,
                class_name="car",
                confidence=0.88,
                bbox=[10.0, 10.0, 40.0, 30.0],
            ),
            TrackedDetection(
                track_id=2,
                class_id=0,
                class_name="person",
                confidence=0.75,
                bbox=[5.0, 5.0, 20.0, 35.0],
            ),
        ]


@pytest.fixture(autouse=True)
def inject_mock_detector():
    mock = MockDetector()
    detector.set_detector(mock)
    yield mock
    detector.set_detector(None)


def test_supported_classes_endpoint(client: TestClient) -> None:
    response = client.get("/api/models/classes")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 4
    assert body["classes"]["2"] == "car"
    assert "default_tracking_classes" in body


def test_create_job_invalid_video_id(client: TestClient) -> None:
    response = client.post("/api/jobs", json={"video_id": "not-a-uuid"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_ID"


def test_create_job_missing_video(client: TestClient) -> None:
    response = client.post("/api/jobs", json={"video_id": "00000000-0000-0000-0000-000000000000"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "VIDEO_NOT_FOUND"


def test_job_lifecycle_mocked(client: TestClient, tmp_path: Path) -> None:
    # First upload a video via Phase 1 analyze endpoint
    video_path = write_sample_video(tmp_path / "scene.mp4", frames=10)
    with video_path.open("rb") as handle:
        analyzed = client.post(
            "/api/videos/analyze",
            files={"file": ("scene.mp4", handle, "video/mp4")},
        )
    assert analyzed.status_code == 200
    video_id = analyzed.json()["id"]

    # Create tracking job
    job_resp = client.post(
        "/api/jobs",
        json={
            "video_id": video_id,
            "confidence_threshold": 0.3,
            "classes": ["car", "person"],
            "save_annotated_video": True,
        },
    )
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    job_id = job_data["job_id"]
    assert job_data["video_id"] == video_id
    assert job_data["status"] in {"queued", "processing", "completed"}

    # Poll status until completed (max 5 seconds)
    for _ in range(50):
        status_resp = client.get(f"/api/jobs/{job_id}")
        assert status_resp.status_code == 200
        if status_resp.json()["status"] == "completed":
            break
        time.sleep(0.1)

    completed = client.get(f"/api/jobs/{job_id}").json()
    assert completed["status"] == "completed"
    assert completed["progress"] == 100.0
    assert completed["summary"]["total_detections"] == 20  # 10 frames * 2 detections
    assert completed["summary"]["unique_tracks"] == 2
    assert completed["summary"]["class_counts"]["car"] == 10
    assert completed["summary"]["class_counts"]["person"] == 10

    # Verify results endpoint
    results_resp = client.get(f"/api/jobs/{job_id}/results")
    assert results_resp.status_code == 200
    results_data = results_resp.json()
    assert len(results_data["frames"]) == 10
    assert results_data["frames"][0]["detections"][0]["track_id"] == 1

    # Verify annotated video retrieval
    annotated_resp = client.get(f"/api/jobs/{job_id}/annotated-video")
    assert annotated_resp.status_code == 200
    assert annotated_resp.headers["content-type"] == "video/mp4"

    # Verify evidence frame retrieval
    evidence_resp = client.get(f"/api/jobs/{job_id}/evidence/0")
    assert evidence_resp.status_code == 200
    assert evidence_resp.headers["content-type"] == "image/jpeg"

    # Test deletion and cleanup
    delete_resp = client.delete(f"/api/jobs/{job_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["status"] == "deleted"

    # Verify it is gone
    after_del = client.get(f"/api/jobs/{job_id}")
    assert after_del.status_code == 404
