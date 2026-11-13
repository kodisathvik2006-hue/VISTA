from __future__ import annotations

import re
import uuid
from pathlib import Path

from app.config import Settings
from app.exceptions import bad_request, not_found

SAFE_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def new_video_id() -> str:
    return str(uuid.uuid4())


def sanitize_display_name(filename: str | None) -> str:
    name = Path(filename or "upload").name
    name = name.replace("\x00", "").strip() or "upload"
    return name[:255]


def extension_for(filename: str, allowed: set[str]) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise bad_request(
            "UNSUPPORTED_FORMAT",
            f"Unsupported file extension '{suffix or '(none)'}'. Allowed: {allowed_list}.",
        )
    return suffix


def is_empty_file(path: Path) -> bool:
    return not path.exists() or path.stat().st_size == 0


def ensure_within(base: Path, candidate: Path) -> Path:
    resolved_base = base.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise bad_request("INVALID_PATH", "Requested file path is not allowed.")
    return resolved


def preview_path(settings: Settings, video_id: str) -> Path:
    if not SAFE_ID.match(video_id):
        raise bad_request("INVALID_ID", "Video id is not a valid identifier.")
    path = ensure_within(settings.frames_path, settings.frames_path / f"{video_id}.jpg")
    if not path.is_file():
        raise not_found("PREVIEW_NOT_FOUND", "Preview frame was not found for this video.")
    return path


def remove_if_exists(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except OSError:
        pass
