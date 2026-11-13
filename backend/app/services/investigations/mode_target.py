from __future__ import annotations

import logging
import uuid
from typing import Sequence

import cv2
import numpy as np

from app.schemas import (
    InvestigationEvent,
    SpecificTargetConfig,
    TargetTelemetry,
    TrackedDetection,
)
from app.services.investigations.base import compute_iou
from app.services.investigations.color_extractor import extract_dominant_color

logger = logging.getLogger("neurovision.investigations.target")


class SpecificTargetTracker:
    """Specialized investigation tracker that tracks ONLY the confirmed target object."""

    def __init__(self, config: SpecificTargetConfig, fps: float, width: int, height: int) -> None:
        self.config = config
        self.fps = max(1.0, fps)
        self.width = width
        self.height = height

        self.criteria = config.target_criteria
        self.target_track_id: int | None = None
        self.matched_target: bool = False

        # Target telemetry state
        self.first_seen: float | None = None
        self.last_seen: float | None = None
        self.seen_frames = 0
        self.confidence_sum = 0.0

        # Trajectory trail for target
        self.trajectory_points: list[tuple[int, int]] = []

        # Motion & stopping analysis
        self.last_centroid: tuple[float, float] | None = None
        self.current_stop_start: float | None = None
        self.stopping_intervals: list[dict[str, float]] = []

        # Lost visibility tracking
        self.last_visible_time: float | None = None
        self.lost_visibility_intervals: list[dict[str, float]] = []

        # Target appearance metadata
        self.confirmed_class = self.criteria.target_class
        self.confirmed_color = self.criteria.target_color
        self.zones_entered: list[str] = [config.camera_zone_label] if config.camera_zone_label else []

        # Key events
        self.events: list[InvestigationEvent] = []

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        frame_bgr: np.ndarray,
        all_detections: Sequence[TrackedDetection],
    ) -> tuple[TrackedDetection | None, np.ndarray, bool]:
        """
        Filters detections so that ONLY the selected target is tracked.
        Returns:
            (target_detection_or_none, annotated_frame, is_key_evidence)
        """
        target_det: TrackedDetection | None = None
        is_key_evidence = False

        # 1. Match target if not yet locked
        if not self.matched_target:
            target_det = self._match_initial_target(frame_bgr, all_detections)
            if target_det is not None:
                self.matched_target = True
                self.target_track_id = target_det.track_id
                self.first_seen = timestamp
                self.confirmed_class = target_det.class_name
                # Extract color from crop
                crop = self._crop_box(frame_bgr, target_det.bbox)
                detected_color = extract_dominant_color(crop)
                if detected_color != "unknown" and not self.confirmed_color:
                    self.confirmed_color = detected_color

                is_key_evidence = True
                self.events.append(
                    InvestigationEvent(
                        event_id=str(uuid.uuid4()),
                        event_type="target_acquired",
                        label="Target Acquired",
                        description=(
                            f"Confirmed {self.confirmed_class} "
                            f"(Color: {self.confirmed_color or 'unspecified'}) "
                            f"locked into tracking at {timestamp:.2f}s."
                        ),
                        start_time=timestamp,
                        end_time=timestamp,
                        frame_index=frame_idx,
                        confidence=target_det.confidence,
                        uncertainty_note="Visual match verified against target selection criteria.",
                        related_track_id=self.target_track_id,
                        metadata={
                            "target_class": self.confirmed_class,
                            "target_color": self.confirmed_color,
                            "bbox": target_det.bbox,
                        },
                    )
                )
        else:
            # Already matched: find detection matching target_track_id
            if self.target_track_id is not None:
                for det in all_detections:
                    if det.track_id == self.target_track_id:
                        target_det = det
                        break

            # Fallback if tracker lost track_id: spatial proximity to last centroid
            if target_det is None and self.last_centroid is not None:
                target_det = self._match_spatial_fallback(all_detections)
                if target_det is not None and target_det.track_id is not None:
                    # Relock track ID
                    self.target_track_id = target_det.track_id

        # 2. Update telemetry
        if target_det is not None:
            self.seen_frames += 1
            self.last_seen = timestamp
            self.confidence_sum += target_det.confidence

            # Centroid
            cx = (target_det.bbox[0] + target_det.bbox[2]) / 2.0
            cy = (target_det.bbox[1] + target_det.bbox[3]) / 2.0
            self.trajectory_points.append((int(cx), int(cy)))
            if len(self.trajectory_points) > 120:
                self.trajectory_points.pop(0)

            # Check if recovering from lost visibility
            if self.last_visible_time is not None:
                gap = timestamp - self.last_visible_time
                if gap > 1.2:
                    self.lost_visibility_intervals.append(
                        {
                            "start": round(self.last_visible_time, 2),
                            "end": round(timestamp, 2),
                            "duration": round(gap, 2),
                        }
                    )
                    self.events.append(
                        InvestigationEvent(
                            event_id=str(uuid.uuid4()),
                            event_type="visibility_regained",
                            label="Target Re-acquired",
                            description=(
                                f"Target re-acquired after {gap:.1f}s period of occlusion or lost visibility."
                            ),
                            start_time=self.last_visible_time,
                            end_time=timestamp,
                            frame_index=frame_idx,
                            confidence=target_det.confidence,
                            uncertainty_note="Association maintained via spatial continuity.",
                            related_track_id=self.target_track_id,
                        )
                    )
            self.last_visible_time = timestamp

            # Analyze stopping intervals
            self._update_motion(timestamp, cx, cy)
        else:
            # Target not seen in this frame
            if self.current_stop_start is not None and self.last_visible_time is not None:
                stop_duration = self.last_visible_time - self.current_stop_start
                if stop_duration >= 2.0:
                    self.stopping_intervals.append(
                        {
                            "start": round(self.current_stop_start, 2),
                            "end": round(self.last_visible_time, 2),
                            "duration": round(stop_duration, 2),
                        }
                    )
                self.current_stop_start = None

        # 3. Target-focused rendering
        annotated = self._render_target_focus(frame_bgr, target_det, timestamp)

        return target_det, annotated, is_key_evidence

    def finalize_telemetry(self, total_frames: int) -> TargetTelemetry:
        """Calculate final telemetry summary for the selected target."""
        # Close any pending stop interval
        if self.current_stop_start is not None and self.last_seen is not None:
            stop_duration = self.last_seen - self.current_stop_start
            if stop_duration >= 2.0:
                self.stopping_intervals.append(
                    {
                        "start": round(self.current_stop_start, 2),
                        "end": round(self.last_seen, 2),
                        "duration": round(stop_duration, 2),
                    }
                )

        avg_conf = (self.confidence_sum / self.seen_frames) if self.seen_frames > 0 else 0.0
        vis_duration = (self.seen_frames / self.fps) if self.fps > 0 else 0.0

        return TargetTelemetry(
            first_seen_timestamp=round(self.first_seen, 2) if self.first_seen is not None else None,
            last_seen_timestamp=round(self.last_seen, 2) if self.last_seen is not None else None,
            total_visible_duration_seconds=round(vis_duration, 2),
            observed_stopping_intervals=self.stopping_intervals,
            zones_entered=self.zones_entered,
            average_confidence=round(avg_conf, 3),
            lost_visibility_intervals=self.lost_visibility_intervals,
            assigned_track_id=self.target_track_id,
            target_class=self.confirmed_class,
            target_color=self.confirmed_color,
            calibration_note="Camera calibration unconfigured; physical speed and geographic coordinates are uncalibrated estimates.",
        )

    def _match_initial_target(
        self,
        frame_bgr: np.ndarray,
        all_detections: Sequence[TrackedDetection],
    ) -> TrackedDetection | None:
        """Find the target candidate based on seed bbox, class, or color."""
        if not all_detections:
            return None

        # 1. Match by seed bounding box IoU if provided
        if self.criteria.seed_bbox:
            best_det = None
            best_iou = 0.25
            for det in all_detections:
                iou = compute_iou(det.bbox, self.criteria.seed_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
            if best_det is not None:
                return best_det

        # 2. Match by class and color filter
        target_cls = (self.criteria.target_class or "").lower().strip()
        target_col = (self.criteria.target_color or "").lower().strip()

        candidates: list[TrackedDetection] = []
        for det in all_detections:
            if target_cls and det.class_name.lower() != target_cls:
                continue
            candidates.append(det)

        if not candidates:
            return None

        # If color is specified, find candidate matching color
        if target_col and target_col != "unspecified":
            for cand in candidates:
                crop = self._crop_box(frame_bgr, cand.bbox)
                cand_color = extract_dominant_color(crop)
                if target_col in cand_color or cand_color in target_col:
                    return cand

        # Default to highest confidence candidate matching class
        return max(candidates, key=lambda c: c.confidence)

    def _match_spatial_fallback(
        self,
        all_detections: Sequence[TrackedDetection],
    ) -> TrackedDetection | None:
        if self.last_centroid is None:
            return None
        target_cls = (self.confirmed_class or "").lower().strip()
        best_cand = None
        min_dist = 180.0  # max pixel jump allowed for relock

        for det in all_detections:
            if target_cls and det.class_name.lower() != target_cls:
                continue
            cx = (det.bbox[0] + det.bbox[2]) / 2.0
            cy = (det.bbox[1] + det.bbox[3]) / 2.0
            dist = float(np.hypot(cx - self.last_centroid[0], cy - self.last_centroid[1]))
            if dist < min_dist:
                min_dist = dist
                best_cand = det

        return best_cand

    def _update_motion(self, timestamp: float, cx: float, cy: float) -> None:
        if self.last_centroid is not None:
            displacement = np.hypot(cx - self.last_centroid[0], cy - self.last_centroid[1])
            is_stopped = displacement < 3.5  # < 3.5 pixels per frame considered stopped
            if is_stopped:
                if self.current_stop_start is None:
                    self.current_stop_start = timestamp
            else:
                if self.current_stop_start is not None:
                    duration = timestamp - self.current_stop_start
                    if duration >= 2.5:
                        self.stopping_intervals.append(
                            {
                                "start": round(self.current_stop_start, 2),
                                "end": round(timestamp, 2),
                                "duration": round(duration, 2),
                            }
                        )
                    self.current_stop_start = None
        self.last_centroid = (cx, cy)

    def _render_target_focus(
        self,
        frame_bgr: np.ndarray,
        target_det: TrackedDetection | None,
        timestamp: float,
    ) -> np.ndarray:
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        # Draw trajectory breadcrumbs for target
        if len(self.trajectory_points) > 1:
            for i in range(1, len(self.trajectory_points)):
                pt1 = self.trajectory_points[i - 1]
                pt2 = self.trajectory_points[i]
                alpha = int(255 * (i / len(self.trajectory_points)))
                cv2.line(annotated, pt1, pt2, (0, 215, 255), 2, cv2.LINE_AA)

        if target_det is not None:
            x1, y1, x2, y2 = [int(v) for v in target_det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Prominent gold / cyan bounding box for TARGET ONLY
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 3)

            # Corner brackets for tactical focus effect
            corner_len = min(20, int((x2 - x1) * 0.25))
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 4)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 4)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 4)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 4)

            # Label pill: TARGET #ID [CLASS] [COLOR] [CONF]
            cls_name = self.confirmed_class or target_det.class_name
            color_str = f"({self.confirmed_color}) " if self.confirmed_color else ""
            label = f"TARGET #{self.target_track_id or 1} {color_str}{cls_name.upper()} {target_det.confidence:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), (0, 180, 220), -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Upper HUD telemetry banner
        cv2.rectangle(annotated, (10, 10), (380, 60), (20, 20, 20), -1)
        cv2.rectangle(annotated, (10, 10), (380, 60), (0, 180, 220), 1)
        target_info = f"TARGET: {self.confirmed_class or 'Pending'} | TRACK #{self.target_track_id or '---'}"
        status_info = f"TIME: {timestamp:.2f}s | VISIBILITY: {'LOCKED' if target_det else 'LOST / OCCLUDED'}"
        cv2.putText(annotated, target_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            status_info,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200) if target_det else (0, 80, 255),
            1,
            cv2.LINE_AA,
        )

        return annotated

    def _crop_box(self, frame_bgr: np.ndarray, bbox: list[float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame_bgr[y1:y2, x1:x2]
