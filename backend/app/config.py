from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Neurovision"
    app_version: str = "0.2.0"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 200
    upload_dir: str = "data/uploads"
    frames_dir: str = "data/frames"
    jobs_dir: str = "data/jobs"
    evidence_dir: str = "data/evidence"
    allowed_extensions: str = ".mp4,.avi,.mov,.mkv,.webm"

    # Phase 2 AI & Tracking Configuration
    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    tracker_config: str = "bytetrack.yaml"
    device: str = "auto"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allowed_extension_set(self) -> set[str]:
        return {
            ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
            for ext in self.allowed_extensions.split(",")
            if ext.strip()
        }

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def resolve_dir(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    @property
    def uploads_path(self) -> Path:
        return self.resolve_dir(self.upload_dir)

    @property
    def frames_path(self) -> Path:
        return self.resolve_dir(self.frames_dir)

    @property
    def jobs_path(self) -> Path:
        return self.resolve_dir(self.jobs_dir)

    @property
    def evidence_path(self) -> Path:
        return self.resolve_dir(self.evidence_dir)

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return self.device


@lru_cache
def get_settings() -> Settings:
    return Settings()
