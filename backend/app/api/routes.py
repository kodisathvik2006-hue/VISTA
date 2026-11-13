from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import Settings, get_settings
from app.exceptions import bad_request, payload_too_large
from app.schemas import (
    CandidatesResponse,
    EventReviewUpdateRequest,
    HealthResponse,
    InvestigationCreateRequest,
    InvestigationEvent,
    InvestigationJobDetailResponse,
    InvestigationJobResponse,
    JobCreateRequest,
    JobDetailResponse,
    JobResponse,
    VideoAnalyzeResponse,
    VideoMetadata,
)
from app.services import detector, storage, video
from app.services.investigations.investigation_manager import get_investigation_manager
from app.services.jobs import get_job_manager

logger = logging.getLogger("neurovision")

router = APIRouter()


# -------------------------------------------------------------------------
# Phase 1: Health & Ingestion Probing
# -------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    opencv_ok = True
    ai_ok = True
    try:
        import cv2  # noqa: F401
    except Exception:
        opencv_ok = False
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except Exception:
        ai_ok = False

    return HealthResponse(
        version=settings.app_version,
        opencv=opencv_ok,
        ai_tracking=ai_ok,
        device=settings.resolved_device,
    )


@router.post("/videos/analyze", response_model=VideoAnalyzeResponse)
async def analyze_video(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> VideoAnalyzeResponse:
    original = storage.sanitize_display_name(file.filename)
    suffix = storage.extension_for(original, settings.allowed_extension_set)

    video_id = storage.new_video_id()
    stored_name = f"{video_id}{suffix}"
    upload_path = settings.uploads_path / stored_name
    preview = settings.frames_path / f"{video_id}.jpg"

    content_length = file.headers.get("content-length")
    if content_length:
        try:
            declared = int(content_length)
        except ValueError:
            declared = 0
        if declared > settings.max_upload_bytes:
            raise payload_too_large(
                f"File exceeds the {settings.max_upload_mb} MB upload limit."
            )

    size = 0
    try:
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        with upload_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise payload_too_large(
                        f"File exceeds the {settings.max_upload_mb} MB upload limit."
                    )
                out.write(chunk)

        if size == 0 or storage.is_empty_file(upload_path):
            raise bad_request("EMPTY_FILE", "The uploaded file is empty.")

        probe = video.probe_video(upload_path)
        video.write_preview_jpeg(probe.preview_bgr, preview)

        return VideoAnalyzeResponse(
            id=video_id,
            original_filename=original,
            stored_filename=stored_name,
            size_bytes=size,
            metadata=VideoMetadata(
                duration_seconds=probe.duration_seconds,
                fps=probe.fps,
                frame_count=probe.frame_count,
                width=probe.width,
                height=probe.height,
            ),
            preview_url=f"/api/videos/{video_id}/preview",
        )
    except Exception:
        storage.remove_if_exists(upload_path)
        storage.remove_if_exists(preview)
        raise
    finally:
        await file.close()


@router.get("/videos/{video_id}/preview")
def get_preview(
    video_id: str,
    candidate_frame: int | None = None,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    if candidate_frame is not None:
        cand_path = settings.frames_path / f"{video_id}_candidates_{candidate_frame}.jpg"
        if cand_path.is_file():
            return FileResponse(cand_path, media_type="image/jpeg", filename=f"{video_id}_candidates_{candidate_frame}.jpg")
    path: Path = storage.preview_path(settings, video_id)
    return FileResponse(path, media_type="image/jpeg", filename=f"{video_id}.jpg")


@router.get("/videos/{video_id}/candidates", response_model=CandidatesResponse)
def get_candidates(
    video_id: str,
    frame_index: int = 0,
    confidence: float = 0.25,
    settings: Settings = Depends(get_settings),
) -> CandidatesResponse:
    manager = get_investigation_manager(settings)
    return manager.get_candidates_on_frame(
        video_id=video_id,
        frame_index=frame_index,
        confidence=confidence,
    )


# -------------------------------------------------------------------------
# Phase 2: AI Detection & Tracking Jobs
# -------------------------------------------------------------------------

@router.get("/models/classes")
def get_supported_classes(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    active_detector = detector.get_detector(settings)
    classes = active_detector.get_supported_classes()
    return {
        "count": len(classes),
        "classes": classes,
        "default_tracking_classes": ["person", "car", "truck", "bus", "motorcycle", "bicycle"],
    }


@router.post("/jobs", response_model=JobResponse)
def create_job(
    request: JobCreateRequest,
    settings: Settings = Depends(get_settings),
) -> JobResponse:
    manager = get_job_manager(settings)
    return manager.create_job(request)


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(settings: Settings = Depends(get_settings)) -> list[JobResponse]:
    manager = get_job_manager(settings)
    return manager.list_jobs()


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, settings: Settings = Depends(get_settings)) -> JobResponse:
    manager = get_job_manager(settings)
    return manager.get_job(job_id)


@router.get("/jobs/{job_id}/results", response_model=JobDetailResponse)
def get_job_results(job_id: str, settings: Settings = Depends(get_settings)) -> JobDetailResponse:
    manager = get_job_manager(settings)
    return manager.get_job_detail(job_id)


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str, settings: Settings = Depends(get_settings)) -> dict[str, str]:
    manager = get_job_manager(settings)
    manager.cancel_or_delete_job(job_id)
    return {"status": "deleted", "job_id": job_id}


@router.get("/jobs/{job_id}/annotated-video")
def get_annotated_video(job_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    manager = get_job_manager(settings)
    video_path = manager.get_annotated_video_path(job_id)
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{job_id}_annotated.mp4",
    )


@router.get("/jobs/{job_id}/evidence/{frame_index}")
def get_evidence_frame(
    job_id: str,
    frame_index: int,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    manager = get_job_manager(settings)
    frame_path = manager.get_evidence_frame_path(job_id, frame_index)
    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        filename=f"evidence_{job_id}_{frame_index}.jpg",
    )


# -------------------------------------------------------------------------
# Investigation Platform Routes
# -------------------------------------------------------------------------

@router.post("/investigations", response_model=InvestigationJobResponse)
def create_investigation(
    request: InvestigationCreateRequest,
    settings: Settings = Depends(get_settings),
) -> InvestigationJobResponse:
    manager = get_investigation_manager(settings)
    return manager.create_investigation(request)


@router.get("/investigations/{job_id}", response_model=InvestigationJobResponse)
def get_investigation(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> InvestigationJobResponse:
    manager = get_investigation_manager(settings)
    return manager.get_investigation(job_id)


@router.get("/investigations/{job_id}/results", response_model=InvestigationJobDetailResponse)
def get_investigation_results(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> InvestigationJobDetailResponse:
    manager = get_investigation_manager(settings)
    return manager.get_investigation_detail(job_id)


@router.patch("/investigations/{job_id}/events/{event_id}", response_model=InvestigationEvent)
def update_investigation_event_review(
    job_id: str,
    event_id: str,
    update: EventReviewUpdateRequest,
    settings: Settings = Depends(get_settings),
) -> InvestigationEvent:
    manager = get_investigation_manager(settings)
    return manager.update_event_review(
        job_id=job_id,
        event_id=event_id,
        review_status=update.review_status,
        notes=update.notes,
    )


@router.get("/investigations/{job_id}/annotated-video")
def get_investigation_video(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    video_path = settings.jobs_path / f"inv_{job_id}_annotated.mp4"
    if not video_path.is_file():
        from app.exceptions import not_found
        raise not_found("VIDEO_NOT_FOUND", "Annotated investigation video not found.")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"investigation_{job_id}.mp4",
    )


@router.get("/investigations/{job_id}/events/{event_id}/clip")
def get_investigation_event_clip(
    job_id: str,
    event_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    manager = get_investigation_manager(settings)
    clip_path = manager.get_event_clip_path(job_id, event_id)
    return FileResponse(
        clip_path,
        media_type="video/mp4",
        filename=f"event_clip_{event_id[:8]}.mp4",
    )


@router.get("/investigations/{job_id}/evidence/{frame_index}")
def get_investigation_evidence_frame(
    job_id: str,
    frame_index: int,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    frame_path = settings.evidence_path / f"inv_{job_id}" / f"frame_{frame_index:06d}.jpg"
    if not frame_path.is_file():
        from app.exceptions import not_found
        raise not_found("FRAME_NOT_FOUND", f"Evidence frame {frame_index} not found.")
    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        filename=f"evidence_{job_id}_{frame_index}.jpg",
    )


@router.delete("/investigations/{job_id}")
def delete_investigation(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    manager = get_investigation_manager(settings)
    manager.cancel_or_delete_investigation(job_id)
    return {"status": "deleted", "job_id": job_id}
