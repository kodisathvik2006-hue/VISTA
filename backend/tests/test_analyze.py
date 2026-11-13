from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import write_sample_video


def test_analyze_real_video(client: TestClient, tmp_path: Path) -> None:
    video_path = write_sample_video(tmp_path / "clip.mp4")
    with video_path.open("rb") as handle:
        response = client.post(
            "/api/videos/analyze",
            files={"file": ("traffic.mp4", handle, "application/octet-stream")},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["original_filename"] == "traffic.mp4"
    assert body["size_bytes"] > 0
    assert body["metadata"]["width"] == 64
    assert body["metadata"]["height"] == 48
    assert body["metadata"]["frame_count"] >= 1
    assert body["preview_url"].startswith("/api/videos/")
    preview = client.get(body["preview_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/jpeg")
    assert preview.content[:2] == b"\xff\xd8"


def test_reject_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/videos/analyze",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_FILE"


def test_reject_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/api/videos/analyze",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_reject_non_video_with_video_extension(client: TestClient) -> None:
    response = client.post(
        "/api/videos/analyze",
        files={"file": ("fake.mp4", b"this is not a video", "video/mp4")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] in {"UNREADABLE_VIDEO", "NO_FRAMES"}


def test_reject_oversize(client: TestClient) -> None:
    huge = b"0" * (2 * 1024 * 1024 + 50)
    response = client.post(
        "/api/videos/analyze",
        files={"file": ("big.mp4", huge, "video/mp4")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_missing_file_field(client: TestClient) -> None:
    response = client.post("/api/videos/analyze")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
