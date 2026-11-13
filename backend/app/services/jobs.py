from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.exceptions import bad_request, not_found
from app.schemas import (
    FrameDetectionRecord,
    JobCreateRequest,
    JobDetailResponse,
    JobResponse,
    JobSummary,
    TrackedDetection,
)
from app.services import annotator, detector, storage

logger = logging.getLogger("neurovision.jobs")

SAFE_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass
class JobState:
    job_id: str
    video_id: str
    status: str = "queued"  # queued, processing, completed, failed, cancelled
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    summary: JobSummary | None = None
    frames: list[FrameDetectionRecord] = field(default_factory=list)
    annotated_video_path: Path | None = None
    should_cancel: bool = False


class JobManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create_job(self, request: JobCreateRequest) -> JobResponse:
        # Validate video_id
        if not SAFE_ID.match(request.video_id):
            raise bad_request("INVALID_ID", "Video ID must be a valid UUID.")

        # Find uploaded video file
        upload_matches = list(self.settings.uploads_path.glob(f"{request.video_id}.*"))
        if not upload_matches:
            raise not_found("VIDEO_NOT_FOUND", "Uploaded video file not found for the given video_id.")
        video_path = upload_matches[0]

        job_id = str(uuid.uuid4())
        job_state = JobState(job_id=job_id, video_id=request.video_id)

        with self._lock:
            self._jobs[job_id] = job_state

        # Launch background processing thread
        thread = threading.Thread(
            target=self._process_job,
            args=(job_id, video_path, request),
            daemon=True,
            name=f"job-{job_id[:8]}",
        )
        thread.start()

        return self._to_response(job_state)

    def get_job(self, job_id: str) -> JobResponse:
        state = self._lookup_job(job_id)
        return self._to_response(state)

    def get_job_detail(self, job_id: str) -> JobDetailResponse:
        state = self._lookup_job(job_id)
        resp = self._to_response(state)
        return JobDetailResponse(
            **resp.model_dump(),
            frames=state.frames,
        )

    def list_jobs(self) -> list[JobResponse]:
        with self._lock:
            states = list(self._jobs.values())
        return [self._to_response(s) for s in sorted(states, key=lambda x: x.created_at, reverse=True)]

    def cancel_or_delete_job(self, job_id: str) -> bool:
        state = self._lookup_job(job_id)
        state.should_cancel = True

        # Clean up files
        annotated_path = self.settings.jobs_path / f"{job_id}_annotated.mp4"
        results_json = self.settings.jobs_path / f"{job_id}_results.json"
        storage.remove_if_exists(annotated_path)
        storage.remove_if_exists(results_json)

        evidence_dir = self.settings.evidence_path / job_id
        if evidence_dir.exists() and evidence_dir.is_dir():
            for f in evidence_dir.glob("*"):
                storage.remove_if_exists(f)
            try:
                evidence_dir.rmdir()
            except OSError:
                pass

        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]

        return True

    def get_annotated_video_path(self, job_id: str) -> Path:
        if not SAFE_ID.match(job_id):
            raise bad_request("INVALID_ID", "Job ID must be a valid UUID.")
        path = storage.ensure_within(
            self.settings.jobs_path,
            self.settings.jobs_path / f"{job_id}_annotated.mp4",
        )
        if not path.is_file():
            raise not_found("VIDEO_NOT_FOUND", "Annotated video not found for this job.")
        return path

    def get_evidence_frame_path(self, job_id: str, frame_index: int) -> Path:
        if not SAFE_ID.match(job_id):
            raise bad_request("INVALID_ID", "Job ID must be a valid UUID.")
        evidence_dir = storage.ensure_within(
            self.settings.evidence_path,
            self.settings.evidence_path / job_id,
        )
        frame_file = storage.ensure_within(
            evidence_dir,
            evidence_dir / f"frame_{frame_index:06d}.jpg",
        )
        if not frame_file.is_file():
            raise not_found("FRAME_NOT_FOUND", f"Evidence frame {frame_index} not found.")
        return frame_file

    def _lookup_job(self, job_id: str) -> JobState:
        if not SAFE_ID.match(job_id):
            raise bad_request("INVALID_ID", "Job ID must be a valid UUID.")
        with self._lock:
            state = self._jobs.get(job_id)
        if state is None:
            # Check if persisted on disk
            results_path = self.settings.jobs_path / f"{job_id}_results.json"
            if results_path.is_file():
                state = self._load_persisted_job(results_path)
                with self._lock:
                    self._jobs[job_id] = state
            else:
                raise not_found("JOB_NOT_FOUND", "Analysis job not found.")
        return state

    def _load_persisted_job(self, path: Path) -> JobState:
        data = json.loads(path.read_text(encoding="utf-8"))
        summary_data = data.get("summary")
        summary = JobSummary(**summary_data) if summary_data else None
        frames = [FrameDetectionRecord(**f) for f in data.get("frames", [])]
        return JobState(
            job_id=data["job_id"],
            video_id=data["video_id"],
            status=data.get("status", "completed"),
            progress=data.get("progress", 100.0),
            current_frame=data.get("current_frame", 0),
            total_frames=data.get("total_frames", 0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            completed_at=data.get("completed_at"),
            summary=summary,
            frames=frames,
            annotated_video_path=self.settings.jobs_path / f"{data['job_id']}_annotated.mp4",
        )

    def _to_response(self, state: JobState) -> JobResponse:
        video_url = None
        if state.status == "completed" and state.annotated_video_path and state.annotated_video_path.is_file():
            video_url = f"/api/jobs/{state.job_id}/annotated-video"

        return JobResponse(
            job_id=state.job_id,
            video_id=state.video_id,
            status=state.status,
            progress=round(state.progress, 1),
            current_frame=state.current_frame,
            total_frames=state.total_frames,
            error=state.error,
            created_at=state.created_at,
            completed_at=state.completed_at,
            summary=state.summary,
            annotated_video_url=video_url,
        )

    def _process_job(self, job_id: str, video_path: Path, req: JobCreateRequest) -> None:
        state = self._jobs[job_id]
        state.status = "processing"
        start_time = time.time()

        annotated_out: cv2.VideoWriter | None = None
        annotated_path = self.settings.jobs_path / f"{job_id}_annotated.mp4"
        evidence_dir = self.settings.evidence_path / job_id

        try:
            active_detector = detector.get_detector(self.settings)
            active_detector.reset_tracker()

            # Resolve class indices
            class_indices = active_detector.resolve_class_indices(req.classes)
            conf_thresh = req.confidence_threshold or self.settings.confidence_threshold

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError("Could not open video file for tracking inference.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            if fps <= 0:
                fps = 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

            if req.max_frames and req.max_frames < total_frames:
                total_frames = req.max_frames

            state.total_frames = max(1, total_frames)

            # Initialize annotated video writer if requested
            if req.save_annotated_video:
                self.settings.jobs_path.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                annotated_out = cv2.VideoWriter(
                    str(annotated_path),
                    fourcc,
                    fps,
                    (width, height),
                )
                if not annotated_out.isOpened():
                    logger.warning("Could not initialize VideoWriter with mp4v codec.")
                    annotated_out = None

            frame_idx = 0
            unique_tracks: set[int] = set()
            class_counts: dict[str, int] = {}
            total_detections = 0
            peak_objects = 0
            evidence_count = 0
            max_evidence = 20

            while cap.isOpened():
                if state.should_cancel:
                    state.status = "cancelled"
                    break

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                if req.max_frames and frame_idx >= req.max_frames:
                    break

                # Run YOLO + ByteTrack
                timestamp = round(frame_idx / fps, 3)
                detections = active_detector.track_frame(
                    frame_bgr=frame,
                    persist=True,
                    conf=conf_thresh,
                    classes=class_indices,
                )

                # Collect metrics
                total_detections += len(detections)
                if len(detections) > peak_objects:
                    peak_objects = len(detections)

                for d in detections:
                    if d.track_id is not None:
                        unique_tracks.add(d.track_id)
                    class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

                # Record frame detections
                state.frames.append(
                    FrameDetectionRecord(
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp,
                        detections=detections,
                    )
                )

                # Annotate & write frame
                if annotated_out is not None:
                    annotated_frame = annotator.annotate_frame(frame, detections)
                    annotated_out.write(annotated_frame)

                    # Save evidence frame if there are detections and within evidence cap
                    if detections and evidence_count < max_evidence and (frame_idx % max(1, int(fps)) == 0 or evidence_count == 0):
                        evidence_file = evidence_dir / f"frame_{frame_idx:06d}.jpg"
                        annotator.save_evidence_frame(annotated_frame, evidence_file)
                        evidence_count += 1

                frame_idx += 1
                state.current_frame = frame_idx
                state.progress = min(99.0, (frame_idx / state.total_frames) * 100.0)

            cap.release()
            if annotated_out is not None:
                annotated_out.release()
                if annotated_path.is_file() and annotated_path.stat().st_size > 0:
                    state.annotated_video_path = annotated_path

            if not state.should_cancel:
                elapsed = max(0.001, time.time() - start_time)
                state.status = "completed"
                state.progress = 100.0
                state.completed_at = datetime.now(timezone.utc).isoformat()
                state.summary = JobSummary(
                    total_frames_processed=frame_idx,
                    total_detections=total_detections,
                    unique_tracks=len(unique_tracks),
                    class_counts=class_counts,
                    peak_objects_in_frame=peak_objects,
                    processing_time_seconds=round(elapsed, 2),
                    fps_processed=round(frame_idx / elapsed, 1),
                )

                # Persist results to disk
                self._persist_job(state)

        except Exception as exc:
            logger.exception("Error processing video job %s: %s", job_id, exc)
            state.status = "failed"
            state.error = str(exc)
            state.completed_at = datetime.now(timezone.utc).isoformat()
            if annotated_out is not None:
                annotated_out.release()
            storage.remove_if_exists(annotated_path)

    def _persist_job(self, state: JobState) -> None:
        try:
            self.settings.jobs_path.mkdir(parents=True, exist_ok=True)
            results_file = self.settings.jobs_path / f"{state.job_id}_results.json"
            data = {
                "job_id": state.job_id,
                "video_id": state.video_id,
                "status": state.status,
                "progress": state.progress,
                "current_frame": state.current_frame,
                "total_frames": state.total_frames,
                "created_at": state.created_at,
                "completed_at": state.completed_at,
                "summary": state.summary.model_dump() if state.summary else None,
                "frames": [f.model_dump() for f in state.frames],
            }
            results_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to persist job %s results: %s", state.job_id, exc)


_job_manager: JobManager | None = None


def get_job_manager(settings: Settings | None = None) -> JobManager:
    global _job_manager
    if settings is None:
        settings = get_settings()
    if _job_manager is None:
        _job_manager = JobManager(settings)
    else:
        _job_manager.settings = settings
    return _job_manager


def reset_job_manager() -> None:
    global _job_manager
    _job_manager = None
