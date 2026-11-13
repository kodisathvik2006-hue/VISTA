from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.schemas import (
    AccidentDetectionConfig,
    InvestigationCreateRequest,
    InvestigationMode,
    SpecificTargetConfig,
    SuspiciousActivityConfig,
    TargetSelectionCriteria,
    TrackedDetection,
    ZonePolygon,
)
from app.services import detector
from app.services.investigations.color_extractor import extract_dominant_color
from app.services.investigations.investigation_manager import get_investigation_manager
from tests.conftest import write_sample_video


class MockInvestigationDetector:
    """Mock detector for predictable testing of investigation modes."""

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
        # Target car moving across frames
        return [
            TrackedDetection(
                track_id=10,
                class_id=2,
                class_name="car",
                confidence=0.92,
                bbox=[20.0 + self.call_count * 5.0, 30.0, 80.0 + self.call_count * 5.0, 70.0],
            ),
            TrackedDetection(
                track_id=20,
                class_id=0,
                class_name="person",
                confidence=0.81,
                bbox=[150.0, 150.0, 180.0, 210.0],
            ),
        ]


@pytest.fixture(autouse=True)
def inject_mock_detector():
    mock = MockInvestigationDetector()
    detector.set_detector(mock)
    yield mock
    detector.set_detector(None)


def test_color_extractor():
    # Create pure red crop (BGR: 0, 0, 255)
    red_crop = np.zeros((60, 60, 3), dtype=np.uint8)
    red_crop[:, :] = [0, 0, 255]
    assert extract_dominant_color(red_crop) == "red"

    # Create pure blue crop (BGR: 255, 0, 0)
    blue_crop = np.zeros((60, 60, 3), dtype=np.uint8)
    blue_crop[:, :] = [255, 0, 0]
    assert extract_dominant_color(blue_crop) == "blue"


def test_specific_target_investigation(client: TestClient, tmp_path):
    video_path = tmp_path / "target_test.mp4"
    write_sample_video(video_path, frames=15, width=320, height=240, fps=10)

    with video_path.open("rb") as f:
        res = client.post("/api/videos/analyze", files={"file": ("target_test.mp4", f, "video/mp4")})
    assert res.status_code == 200
    video_id = res.json()["id"]

    # 1. Test Candidate Extraction
    cand_res = client.get(f"/api/videos/{video_id}/candidates?frame_index=0")
    assert cand_res.status_code == 200
    candidates = cand_res.json()["candidates"]
    assert len(candidates) >= 1
    assert any(c["class_name"] == "car" for c in candidates)

    # 2. Launch Specific Target Investigation (Target: Car)
    req = InvestigationCreateRequest(
        video_id=video_id,
        mode=InvestigationMode.SPECIFIC_TARGET,
        target_config=SpecificTargetConfig(
            target_criteria=TargetSelectionCriteria(
                target_class="car",
                seed_bbox=[20.0, 30.0, 80.0, 70.0],
            ),
            camera_zone_label="Main Gate Area",
        ),
        max_frames=10,
    )
    inv_res = client.post("/api/investigations", json=req.model_dump())
    assert inv_res.status_code == 200
    job_id = inv_res.json()["job_id"]
    assert inv_res.json()["mode"] == "specific_target"

    # Wait for completion
    import time
    for _ in range(30):
        detail_res = client.get(f"/api/investigations/{job_id}/results")
        if detail_res.json()["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert detail_res.status_code == 200
    data = detail_res.json()
    assert data["status"] == "completed"
    assert data["summary"]["target_telemetry"] is not None
    telemetry = data["summary"]["target_telemetry"]
    assert telemetry["target_class"] == "car"
    assert telemetry["assigned_track_id"] == 10
    assert telemetry["total_visible_duration_seconds"] > 0


def test_suspicious_activity_investigation(client: TestClient, tmp_path):
    video_path = tmp_path / "suspicious_test.mp4"
    write_sample_video(video_path, frames=12, width=320, height=240, fps=10)

    with video_path.open("rb") as f:
        res = client.post("/api/videos/analyze", files={"file": ("suspicious_test.mp4", f, "video/mp4")})
    assert res.status_code == 200
    video_id = res.json()["id"]

    req = InvestigationCreateRequest(
        video_id=video_id,
        mode=InvestigationMode.SUSPICIOUS_ACTIVITY,
        suspicious_config=SuspiciousActivityConfig(
            restricted_zones=[
                ZonePolygon(
                    zone_id="restricted_dock",
                    name="Restricted Loading Dock",
                    points=[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
                    is_restricted=True,
                )
            ],
            stop_duration_threshold_seconds=1.0,
        ),
        max_frames=10,
    )
    inv_res = client.post("/api/investigations", json=req.model_dump())
    assert inv_res.status_code == 200
    job_id = inv_res.json()["job_id"]

    import time
    for _ in range(30):
        detail_res = client.get(f"/api/investigations/{job_id}/results")
        if detail_res.json()["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert detail_res.status_code == 200
    data = detail_res.json()
    assert data["status"] == "completed"

    # Test updating event review status
    if len(data["events"]) > 0:
        event_id = data["events"][0]["event_id"]
        patch_res = client.patch(
            f"/api/investigations/{job_id}/events/{event_id}",
            json={"review_status": "Verified Incident", "notes": "Investigator confirmed entry."},
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["review_status"] == "Verified Incident"
