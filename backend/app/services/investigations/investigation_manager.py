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
    CandidateDetection,
    CandidatesResponse,
    InvestigationCreateRequest,
    InvestigationEvent,
    InvestigationJobDetailResponse,
    InvestigationJobResponse,
    InvestigationMode,
    InvestigationSummary,
    TargetTelemetry,
    TrackedDetection,
)
from app.services import detector, storage
from app.services.investigations.color_extractor import extract_dominant_color
from app.services.investigations.mode_accident import AccidentDetector
from app.services.investigations.mode_altercation import AltercationActionRecognitionInterface
from app.services.investigations.mode_suspicious import SuspiciousActivityDetector
from app.services.investigations.mode_target import SpecificTargetTracker

logger = logging.getLogger("neurovision.investigations.manager")

SAFE_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass
class InvestigationJobState:
    job_id: str
    video_id: str
    mode: InvestigationMode
    status: str = "queued"  # queued, processing, completed, failed, cancelled
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    summary: InvestigationSummary | None = None
    events: list[InvestigationEvent] = field(default_factory=list)
    evidence_frame_urls: list[str] = field(default_factory=list)
    annotated_video_path: Path | None = None
    should_cancel: bool = False


class InvestigationManager:
    """Manages investigation jobs, execution pipelines, evidence files, and persistence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: dict[str, InvestigationJobState] = {}
        self._lock = threading.Lock()

    def get_candidates_on_frame(
        self,
        video_id: str,
        frame_index: int = 0,
        confidence: float = 0.25,
    ) -> CandidatesResponse:
        """Extract candidate detections with dominant colors on a reference frame."""
        if not SAFE_ID.match(video_id):
            raise bad_request("INVALID_ID", "Video ID must be a valid UUID.")

        upload_matches = list(self.settings.uploads_path.glob(f"{video_id}.*"))
        if not upload_matches:
            raise not_found("VIDEO_NOT_FOUND", "Uploaded video file not found.")
        video_path = upload_matches[0]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise bad_request("UNREADABLE_VIDEO", "Could not open video file.")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        target_idx = max(0, min(frame_index, total_frames - 1))

        if target_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise bad_request("FRAME_READ_FAILED", f"Could not read frame {target_idx} from video.")

        # Detect candidates using YOLO
        active_detector = detector.get_detector(self.settings)
        active_detector.reset_tracker()
        dets = active_detector.track_frame(
            frame_bgr=frame,
            persist=False,
            conf=confidence,
            classes=None,
        )

        candidates: list[CandidateDetection] = []
        annotated = frame.copy()
        h, w = frame.shape[:2]

        for idx, d in enumerate(dets):
            cand_id = idx + 1
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            color = extract_dominant_color(crop)

            candidates.append(
                CandidateDetection(
                    candidate_id=cand_id,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    bbox=d.bbox,
                    dominant_color=color,
                )
            )

            # Draw visual candidate badge on reference preview
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 255), 2)
            label = f"#{cand_id} {d.class_name} ({color})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 230, 255), -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Save preview with candidates
        self.settings.frames_path.mkdir(parents=True, exist_ok=True)
        preview_filename = f"{video_id}_candidates_{target_idx}.jpg"
        preview_path = self.settings.frames_path / preview_filename
        cv2.imwrite(str(preview_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        timestamp = round(target_idx / max(1.0, fps), 2)
        return CandidatesResponse(
            video_id=video_id,
            frame_index=target_idx,
            timestamp_seconds=timestamp,
            candidates=candidates,
            preview_url=f"/api/videos/{video_id}/preview?candidate_frame={target_idx}",
        )

    def create_investigation(self, request: InvestigationCreateRequest) -> InvestigationJobResponse:
        """Create and launch an asynchronous investigation job."""
        if not SAFE_ID.match(request.video_id):
            raise bad_request("INVALID_ID", "Video ID must be a valid UUID.")

        upload_matches = list(self.settings.uploads_path.glob(f"{request.video_id}.*"))
        if not upload_matches:
            raise not_found("VIDEO_NOT_FOUND", "Uploaded video file not found for the given video_id.")
        video_path = upload_matches[0]

        job_id = str(uuid.uuid4())
        job_state = InvestigationJobState(
            job_id=job_id,
            video_id=request.video_id,
            mode=request.mode,
        )

        with self._lock:
            self._jobs[job_id] = job_state

        thread = threading.Thread(
            target=self._process_investigation,
            args=(job_id, video_path, request),
            daemon=True,
            name=f"investigation-{job_id[:8]}",
        )
        thread.start()

        return self._to_response(job_state)

    def get_investigation(self, job_id: str) -> InvestigationJobResponse:
        state = self._lookup_job(job_id)
        return self._to_response(state)

    def get_investigation_detail(self, job_id: str) -> InvestigationJobDetailResponse:
        state = self._lookup_job(job_id)
        resp = self._to_response(state)
        return InvestigationJobDetailResponse(
            **resp.model_dump(),
            events=state.events,
            evidence_frame_urls=state.evidence_frame_urls,
        )

    def update_event_review(
        self,
        job_id: str,
        event_id: str,
        review_status: str,
        notes: str | None = None,
    ) -> InvestigationEvent:
        """Update investigator review status of a specific event."""
        state = self._lookup_job(job_id)
        found_event = None
        for ev in state.events:
            if ev.event_id == event_id:
                ev.review_status = review_status
                if notes:
                    ev.metadata["investigator_notes"] = notes
                found_event = ev
                break

        if not found_event:
            raise not_found("EVENT_NOT_FOUND", f"Event {event_id} not found in investigation {job_id}.")

        # Re-tally summary review pending count
        if state.summary:
            pending = sum(1 for e in state.events if e.review_status == "Pending Human Review")
            state.summary.review_pending_count = pending

        self._persist_job(state)
        return found_event

    def get_event_clip_path(self, job_id: str, event_id: str) -> Path:
        """Extract and return a focused video snippet around a flagged incident."""
        state = self._lookup_job(job_id)
        clip_path = self.settings.jobs_path / f"inv_{job_id}_clip_{event_id}.mp4"
        if clip_path.is_file() and clip_path.stat().st_size > 0:
            return clip_path

        found_ev = None
        for ev in state.events:
            if ev.event_id == event_id:
                found_ev = ev
                break
        if not found_ev:
            raise not_found("EVENT_NOT_FOUND", f"Event {event_id} not found.")

        # Source video: annotated video if exists, else uploaded video
        src_video = self.settings.jobs_path / f"inv_{job_id}_annotated.mp4"
        if not src_video.is_file():
            matches = list(self.settings.uploads_path.glob(f"{state.video_id}.*"))
            if matches:
                src_video = matches[0]
            else:
                raise not_found("VIDEO_NOT_FOUND", "Source video not found for clip extraction.")

        cap = cv2.VideoCapture(str(src_video))
        if not cap.isOpened():
            raise bad_request("VIDEO_ERROR", "Could not open video for clip extraction.")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        start_frame = max(0, int((found_ev.start_time - 1.5) * fps))
        end_frame = max(start_frame + int(fps * 2), int((found_ev.end_time + 1.5) * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.settings.jobs_path.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))

        curr = start_frame
        while cap.isOpened() and curr <= end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            curr += 1

        cap.release()
        writer.release()
        return clip_path

    def cancel_or_delete_investigation(self, job_id: str) -> bool:
        state = self._lookup_job(job_id)
        state.should_cancel = True

        annotated_path = self.settings.jobs_path / f"inv_{job_id}_annotated.mp4"
        results_json = self.settings.jobs_path / f"inv_{job_id}_results.json"
        storage.remove_if_exists(annotated_path)
        storage.remove_if_exists(results_json)

        evidence_dir = self.settings.evidence_path / f"inv_{job_id}"
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

    def _lookup_job(self, job_id: str) -> InvestigationJobState:
        if not SAFE_ID.match(job_id):
            raise bad_request("INVALID_ID", "Job ID must be a valid UUID.")
        with self._lock:
            state = self._jobs.get(job_id)
        if state is None:
            results_path = self.settings.jobs_path / f"inv_{job_id}_results.json"
            if results_path.is_file():
                state = self._load_persisted_job(results_path)
                with self._lock:
                    self._jobs[job_id] = state
            else:
                raise not_found("INVESTIGATION_NOT_FOUND", "Investigation job not found.")
        return state

    def _load_persisted_job(self, path: Path) -> InvestigationJobState:
        data = json.loads(path.read_text(encoding="utf-8"))
        summary_data = data.get("summary")
        summary = InvestigationSummary(**summary_data) if summary_data else None
        events = [InvestigationEvent(**e) for e in data.get("events", [])]
        return InvestigationJobState(
            job_id=data["job_id"],
            video_id=data["video_id"],
            mode=InvestigationMode(data["mode"]),
            status=data.get("status", "completed"),
            progress=data.get("progress", 100.0),
            current_frame=data.get("current_frame", 0),
            total_frames=data.get("total_frames", 0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            completed_at=data.get("completed_at"),
            summary=summary,
            events=events,
            evidence_frame_urls=data.get("evidence_frame_urls", []),
            annotated_video_path=self.settings.jobs_path / f"inv_{data['job_id']}_annotated.mp4",
        )

    def _to_response(self, state: InvestigationJobState) -> InvestigationJobResponse:
        video_url = None
        if state.status == "completed" and state.annotated_video_path and state.annotated_video_path.is_file():
            video_url = f"/api/investigations/{state.job_id}/annotated-video"

        return InvestigationJobResponse(
            job_id=state.job_id,
            video_id=state.video_id,
            mode=state.mode,
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

    def _process_investigation(
        self,
        job_id: str,
        video_path: Path,
        req: InvestigationCreateRequest,
    ) -> None:
        state = self._jobs[job_id]
        state.status = "processing"
        start_time = time.time()

        annotated_out: cv2.VideoWriter | None = None
        annotated_path = self.settings.jobs_path / f"inv_{job_id}_annotated.mp4"
        evidence_dir = self.settings.evidence_path / f"inv_{job_id}"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        try:
            active_detector = detector.get_detector(self.settings)
            active_detector.reset_tracker()

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError("Could not open video file for investigation.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            if fps <= 0:
                fps = 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

            if req.max_frames and req.max_frames < total_frames:
                total_frames = req.max_frames
            state.total_frames = max(1, total_frames)

            # Initialize annotated video writer
            if req.save_annotated_video:
                self.settings.jobs_path.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                annotated_out = cv2.VideoWriter(
                    str(annotated_path),
                    fourcc,
                    fps,
                    (width, height),
                )

            # Initialize Mode-Specific Processor
            mode = req.mode
            target_tracker: SpecificTargetTracker | None = None
            suspicious_detector: SuspiciousActivityDetector | None = None
            accident_detector: AccidentDetector | None = None
            altercation_detector: AltercationActionRecognitionInterface | None = None

            conf_thresh = 0.25
            class_filter: list[int] | None = None

            if mode == InvestigationMode.SPECIFIC_TARGET:
                cfg = req.target_config
                if cfg is None:
                    from app.schemas import SpecificTargetConfig, TargetSelectionCriteria
                    cfg = SpecificTargetConfig(target_criteria=TargetSelectionCriteria())
                target_tracker = SpecificTargetTracker(cfg, fps, width, height)
                conf_thresh = cfg.confidence_threshold
            elif mode == InvestigationMode.SUSPICIOUS_ACTIVITY:
                cfg = req.suspicious_config
                if cfg is None:
                    from app.schemas import SuspiciousActivityConfig
                    cfg = SuspiciousActivityConfig()
                suspicious_detector = SuspiciousActivityDetector(cfg, fps, width, height)
                conf_thresh = cfg.confidence_threshold
                class_filter = active_detector.resolve_class_indices(cfg.classes)
            elif mode == InvestigationMode.ACCIDENT_DETECTION:
                cfg = req.accident_config
                if cfg is None:
                    from app.schemas import AccidentDetectionConfig
                    cfg = AccidentDetectionConfig()
                accident_detector = AccidentDetector(cfg, fps, width, height)
                conf_thresh = cfg.confidence_threshold
                class_filter = active_detector.resolve_class_indices(["car", "truck", "bus", "motorcycle"])
            elif mode == InvestigationMode.ALTERCATION_DETECTION:
                cfg = req.altercation_config
                if cfg is None:
                    from app.schemas import AltercationDetectionConfig
                    cfg = AltercationDetectionConfig()
                altercation_detector = AltercationActionRecognitionInterface(cfg, fps, width, height)
                conf_thresh = cfg.confidence_threshold
                class_filter = active_detector.resolve_class_indices(["person"])

            frame_idx = 0
            evidence_count = 0
            max_evidence = 25

            while cap.isOpened():
                if state.should_cancel:
                    state.status = "cancelled"
                    break

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                if req.max_frames and frame_idx >= req.max_frames:
                    break

                timestamp = round(frame_idx / fps, 3)

                # Run YOLO + ByteTrack
                detections = active_detector.track_frame(
                    frame_bgr=frame,
                    persist=True,
                    conf=conf_thresh,
                    classes=class_filter,
                )

                # Run mode-specific processor
                annotated_frame = frame
                is_key_evidence = False

                if target_tracker is not None:
                    _, annotated_frame, is_key_evidence = target_tracker.process_frame(
                        frame_idx, timestamp, frame, detections
                    )
                elif suspicious_detector is not None:
                    annotated_frame, is_key_evidence = suspicious_detector.process_frame(
                        frame_idx, timestamp, frame, detections
                    )
                elif accident_detector is not None:
                    annotated_frame, is_key_evidence = accident_detector.process_frame(
                        frame_idx, timestamp, frame, detections
                    )
                elif altercation_detector is not None:
                    annotated_frame, is_key_evidence = altercation_detector.process_frame(
                        frame_idx, timestamp, frame, detections
                    )

                # Write annotated video frame
                if annotated_out is not None:
                    annotated_out.write(annotated_frame)

                # Save key evidence frame
                if is_key_evidence and evidence_count < max_evidence:
                    evidence_file = evidence_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(evidence_file), annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    state.evidence_frame_urls.append(f"/api/investigations/{job_id}/evidence/{frame_idx}")
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

                # Collect events from active mode
                if target_tracker is not None:
                    state.events = target_tracker.events
                    target_telemetry = target_tracker.finalize_telemetry(frame_idx)
                elif suspicious_detector is not None:
                    state.events = suspicious_detector.events
                    target_telemetry = None
                elif accident_detector is not None:
                    state.events = accident_detector.events
                    target_telemetry = None
                elif altercation_detector is not None:
                    state.events = altercation_detector.events
                    target_telemetry = None
                else:
                    state.events = []
                    target_telemetry = None

                # Summary tally
                events_by_type: dict[str, int] = {}
                for ev in state.events:
                    events_by_type[ev.event_type] = events_by_type.get(ev.event_type, 0) + 1

                # Link evidence frame & clip URLs to events
                for ev in state.events:
                    ev.evidence_frame_url = f"/api/investigations/{job_id}/evidence/{ev.frame_index}"
                    ev.clip_url = f"/api/investigations/{job_id}/events/{ev.event_id}/clip"

                summary_notes = [
                    f"Investigation executed in mode '{mode.value}'.",
                    "All flagged indicators are preliminary optical findings requiring investigator verification.",
                ]
                if mode == InvestigationMode.SPECIFIC_TARGET:
                    summary_notes.append("Persistent tracking was restricted exclusively to the selected target.")
                elif mode == InvestigationMode.ALTERCATION_DETECTION:
                    summary_notes.append("Temporal action interface active; heuristic proximity & kinematic turbulence evaluated.")

                state.summary = InvestigationSummary(
                    mode=mode,
                    total_events=len(state.events),
                    events_by_type=events_by_type,
                    review_pending_count=len(state.events),
                    target_telemetry=target_telemetry,
                    processing_time_seconds=round(elapsed, 2),
                    total_frames_processed=frame_idx,
                    fps_processed=round(frame_idx / elapsed, 1),
                    notes=summary_notes,
                )

                self._persist_job(state)

        except Exception as exc:
            logger.exception("Error processing investigation job %s: %s", job_id, exc)
            state.status = "failed"
            state.error = str(exc)
            state.completed_at = datetime.now(timezone.utc).isoformat()
            if annotated_out is not None:
                annotated_out.release()
            storage.remove_if_exists(annotated_path)

    def _persist_job(self, state: InvestigationJobState) -> None:
        try:
            self.settings.jobs_path.mkdir(parents=True, exist_ok=True)
            results_file = self.settings.jobs_path / f"inv_{state.job_id}_results.json"
            data = {
                "job_id": state.job_id,
                "video_id": state.video_id,
                "mode": state.mode.value,
                "status": state.status,
                "progress": state.progress,
                "current_frame": state.current_frame,
                "total_frames": state.total_frames,
                "created_at": state.created_at,
                "completed_at": state.completed_at,
                "summary": state.summary.model_dump() if state.summary else None,
                "events": [e.model_dump() for e in state.events],
                "evidence_frame_urls": state.evidence_frame_urls,
            }
            results_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to persist investigation job %s: %s", state.job_id, exc)


_investigation_manager: InvestigationManager | None = None


def get_investigation_manager(settings: Settings | None = None) -> InvestigationManager:
    global _investigation_manager
    if settings is None:
        settings = get_settings()
    if _investigation_manager is None:
        _investigation_manager = InvestigationManager(settings)
    else:
        _investigation_manager.settings = settings
    return _investigation_manager
