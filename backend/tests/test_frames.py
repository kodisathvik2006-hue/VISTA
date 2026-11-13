from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import write_sample_video


def test_preview_rejects_path_traversal(client: TestClient) -> None:
    response = client.get("/api/videos/../secret/preview")
    assert response.status_code in {400, 404, 422}


def test_preview_rejects_non_uuid(client: TestClient) -> None:
    response = client.get("/api/videos/not-a-uuid/preview")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_ID"


def test_preview_missing(client: TestClient) -> None:
    response = client.get("/api/videos/11111111-1111-1111-1111-111111111111/preview")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "PREVIEW_NOT_FOUND"


def test_preview_does_not_expose_upload_path(client: TestClient, tmp_path: Path) -> None:
    video_path = write_sample_video(tmp_path / "clip.mp4")
    with video_path.open("rb") as handle:
        analyzed = client.post(
            "/api/videos/analyze",
            files={"file": ("scene.mov", handle, "video/quicktime")},
        )
    video_id = analyzed.json()["id"]
    response = client.get(f"/api/videos/{video_id}/preview")
    assert response.status_code == 200
    assert "content-disposition" in {key.lower() for key in response.headers}
    assert ".." not in response.headers.get("content-disposition", "")
