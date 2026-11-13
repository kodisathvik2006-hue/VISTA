from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services.jobs import reset_job_manager


@pytest.fixture
def tmp_data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    uploads = tmp_path / "uploads"
    frames = tmp_path / "frames"
    jobs = tmp_path / "jobs"
    evidence = tmp_path / "evidence"
    uploads.mkdir()
    frames.mkdir()
    jobs.mkdir()
    evidence.mkdir()
    monkeypatch.setenv("UPLOAD_DIR", str(uploads))
    monkeypatch.setenv("FRAMES_DIR", str(frames))
    monkeypatch.setenv("JOBS_DIR", str(jobs))
    monkeypatch.setenv("EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("MAX_UPLOAD_MB", "2")
    get_settings.cache_clear()
    reset_job_manager()
    return tmp_path


@pytest.fixture
def client(tmp_data_dirs: Path) -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
    reset_job_manager()


def write_sample_video(path: Path, frames: int = 12, width: int = 64, height: int = 48, fps: int = 10) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter could not create a sample video in this environment.")
    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (index * 18 % 255, 90, 160)
        cv2.rectangle(frame, (index, 10), (index + 12, 30), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path
