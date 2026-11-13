from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "neurovision-api"
    version: str
    opencv: bool = True
    ai_tracking: bool = True
    device: str = "cpu"


class VideoMetadata(BaseModel):
    duration_seconds: float = Field(ge=0)
    fps: float = Field(ge=0)
    frame_count: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class VideoAnalyzeResponse(BaseModel):
    id: str
    original_filename: str
    stored_filename: str
    size_bytes: int
    metadata: VideoMetadata
    preview_url: str


# Phase 2 Schemas
class TrackedDetection(BaseModel):
    track_id: int | None = None
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]


class FrameDetectionRecord(BaseModel):
    frame_index: int
    timestamp_seconds: float
    detections: list[TrackedDetection] = []


class JobSummary(BaseModel):
    total_frames_processed: int = 0
    total_detections: int = 0
    unique_tracks: int = 0
    class_counts: dict[str, int] = {}
    peak_objects_in_frame: int = 0
    processing_time_seconds: float = 0.0
    fps_processed: float = 0.0


class JobCreateRequest(BaseModel):
    video_id: str
    confidence_threshold: float | None = Field(default=None, ge=0.01, le=1.0)
    classes: list[str] | None = None
    save_annotated_video: bool = True
    max_frames: int | None = Field(default=None, ge=1)


class JobResponse(BaseModel):
    job_id: str
    video_id: str
    status: str  # "queued" | "processing" | "completed" | "failed" | "cancelled"
    progress: float = 0.0  # 0.0 to 100.0
    current_frame: int = 0
    total_frames: int = 0
    error: str | None = None
    created_at: str
    completed_at: str | None = None
    summary: JobSummary | None = None
    annotated_video_url: str | None = None


class JobDetailResponse(JobResponse):
    frames: list[FrameDetectionRecord] = []


# =========================================================================
# Investigation Platform Schemas (Investigation-Based Intelligence)
# =========================================================================

from enum import Enum
from typing import Any


class InvestigationMode(str, Enum):
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SPECIFIC_TARGET = "specific_target"
    ACCIDENT_DETECTION = "accident_detection"
    ALTERCATION_DETECTION = "altercation_detection"


class ZonePolygon(BaseModel):
    zone_id: str
    name: str
    points: list[list[float]] = []  # Normalized [[x, y], ...] in 0.0-1.0
    is_restricted: bool = True


class SuspiciousActivityConfig(BaseModel):
    restricted_zones: list[ZonePolygon] = []
    stop_duration_threshold_seconds: float = Field(default=4.0, ge=1.0, le=60.0)
    repeated_crossing_threshold: int = Field(default=3, ge=2, le=20)
    confidence_threshold: float = Field(default=0.25, ge=0.05, le=0.95)
    classes: list[str] = ["person", "car", "motorcycle", "truck", "bus"]


class TargetSelectionCriteria(BaseModel):
    candidate_id: int | None = None
    target_class: str | None = None  # e.g. "car", "motorcycle", "truck", "bus", "person"
    target_color: str | None = None  # e.g. "white", "black", "silver/gray", "red", "blue"
    target_plate: str | None = None  # Optional OCR registration query
    seed_bbox: list[float] | None = None  # [x1, y1, x2, y2]
    seed_frame_index: int = 0


class SpecificTargetConfig(BaseModel):
    target_criteria: TargetSelectionCriteria
    confidence_threshold: float = Field(default=0.25, ge=0.05, le=0.95)
    camera_zone_label: str = "Monitored Zone A"


class AccidentDetectionConfig(BaseModel):
    confidence_threshold: float = Field(default=0.25, ge=0.05, le=0.95)
    ttc_threshold_seconds: float = Field(default=1.5, ge=0.3, le=5.0)
    min_speed_delta_ratio: float = Field(default=0.35, ge=0.1, le=0.9)


class AltercationDetectionConfig(BaseModel):
    confidence_threshold: float = Field(default=0.25, ge=0.05, le=0.95)
    proximity_threshold_pixels: float = Field(default=75.0, ge=20.0, le=250.0)
    motion_energy_threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    target_model: str = "VideoMAE-Temporal-Interface"


class InvestigationCreateRequest(BaseModel):
    video_id: str
    mode: InvestigationMode
    suspicious_config: SuspiciousActivityConfig | None = None
    target_config: SpecificTargetConfig | None = None
    accident_config: AccidentDetectionConfig | None = None
    altercation_config: AltercationDetectionConfig | None = None
    save_annotated_video: bool = True
    max_frames: int | None = Field(default=None, ge=1)


class CandidateDetection(BaseModel):
    candidate_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]
    dominant_color: str


class CandidatesResponse(BaseModel):
    video_id: str
    frame_index: int
    timestamp_seconds: float
    candidates: list[CandidateDetection] = []
    preview_url: str


class InvestigationEvent(BaseModel):
    event_id: str
    event_type: str  # e.g. "restricted_zone_entry", "unusual_stop", "collision_course", etc.
    label: str
    description: str
    start_time: float
    end_time: float
    frame_index: int
    confidence: float
    uncertainty_note: str
    review_status: str = "Pending Human Review"  # "Pending Human Review", "Verified Incident", "False Positive - Dismissed"
    evidence_frame_url: str | None = None
    clip_url: str | None = None
    related_track_id: int | None = None
    metadata: dict[str, Any] = {}


class TargetTelemetry(BaseModel):
    first_seen_timestamp: float | None = None
    last_seen_timestamp: float | None = None
    total_visible_duration_seconds: float = 0.0
    observed_stopping_intervals: list[dict[str, float]] = []  # [{"start": s, "end": e, "duration": d}]
    zones_entered: list[str] = []
    average_confidence: float = 0.0
    lost_visibility_intervals: list[dict[str, float]] = []  # [{"start": s, "end": e, "duration": d}]
    assigned_track_id: int | None = None
    target_class: str | None = None
    target_color: str | None = None
    calibration_note: str = "No camera calibration configured. Spatial metrics are uncalibrated estimates."


class InvestigationSummary(BaseModel):
    mode: InvestigationMode
    total_events: int = 0
    events_by_type: dict[str, int] = {}
    review_pending_count: int = 0
    target_telemetry: TargetTelemetry | None = None
    processing_time_seconds: float = 0.0
    total_frames_processed: int = 0
    fps_processed: float = 0.0
    notes: list[str] = []


class InvestigationJobResponse(BaseModel):
    job_id: str
    video_id: str
    mode: InvestigationMode
    status: str  # "queued", "processing", "completed", "failed", "cancelled"
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    error: str | None = None
    created_at: str
    completed_at: str | None = None
    summary: InvestigationSummary | None = None
    annotated_video_url: str | None = None


class InvestigationJobDetailResponse(InvestigationJobResponse):
    events: list[InvestigationEvent] = []
    evidence_frame_urls: list[str] = []


class EventReviewUpdateRequest(BaseModel):
    review_status: str = Field(pattern="^(Pending Human Review|Verified Incident|False Positive - Dismissed)$")
    notes: str | None = None
