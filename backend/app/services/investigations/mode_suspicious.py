from __future__ import annotations

import logging
import uuid
from typing import Sequence

import cv2
import numpy as np

from app.schemas import (
    InvestigationEvent,
    SuspiciousActivityConfig,
    TrackedDetection,
    ZonePolygon,
)
from app.services.investigations.base import draw_zone_overlay, point_in_polygon

logger = logging.getLogger("neurovision.investigations.suspicious")


class SuspiciousActivityDetector:
    """
    Evaluates observable spatial-temporal physical indicators:
    - Restricted zone entries
    - Prolonged stationary stops in monitored zones
    - Repeated back-and-forth zone crossings

    Strict ethical constraint: Observational events only. Does not classify persons
    or infer criminal intent. All events require human review.
    """

    def __init__(self, config: SuspiciousActivityConfig, fps: float, width: int, height: int) -> None:
        self.config = config
        self.fps = max(1.0, fps)
        self.width = width
        self.height = height

        self.zones: list[ZonePolygon] = list(config.restricted_zones)
        # Default fallback zones if none provided
        if not self.zones:
            self.zones = [
                ZonePolygon(
                    zone_id="restricted_perimeter",
                    name="Restricted Zone Alpha",
                    points=[[0.05, 0.55], [0.45, 0.55], [0.45, 0.95], [0.05, 0.95]],
                    is_restricted=True,
                )
            ]

        # Per-track state
        self.track_positions: dict[int, list[tuple[float, float, float]]] = {}  # track_id -> [(time, x, y)]
        self.track_current_zone: dict[int, str | None] = {}
        self.track_zone_crossings: dict[int, list[tuple[float, str]]] = {}
        self.stop_tracking: dict[int, float] = {}  # track_id -> stop_start_timestamp

        # Recorded events
        self.events: list[InvestigationEvent] = []
        self._reported_entries: set[tuple[int, str]] = set()  # (track_id, zone_id)
        self._reported_stops: set[int] = set()
        self._reported_crossings: set[int] = set()

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        frame_bgr: np.ndarray,
        all_detections: Sequence[TrackedDetection],
    ) -> tuple[np.ndarray, bool]:
        """Process frame for observable zone & motion rules."""
        is_key_evidence = False
        flagged_detections: dict[int, str] = {}  # track_id -> event label

        for det in all_detections:
            if det.track_id is None:
                continue

            tid = det.track_id
            # Normalized centroid [0..1]
            cx = ((det.bbox[0] + det.bbox[2]) / 2.0) / self.width
            cy = ((det.bbox[1] + det.bbox[3]) / 2.0) / self.height

            if tid not in self.track_positions:
                self.track_positions[tid] = []
            self.track_positions[tid].append((timestamp, cx, cy))
            if len(self.track_positions[tid]) > 90:
                self.track_positions[tid].pop(0)

            # Check which zone centroid is currently inside
            current_zone = None
            for zone in self.zones:
                if point_in_polygon(cx, cy, zone.points):
                    current_zone = zone
                    break

            prev_zone_id = self.track_current_zone.get(tid)
            curr_zone_id = current_zone.zone_id if current_zone else None

            # Rule 1: Restricted zone entry
            if current_zone and current_zone.is_restricted:
                if (tid, current_zone.zone_id) not in self._reported_entries:
                    self._reported_entries.add((tid, current_zone.zone_id))
                    flagged_detections[tid] = "Restricted Entry"
                    is_key_evidence = True
                    self.events.append(
                        InvestigationEvent(
                            event_id=str(uuid.uuid4()),
                            event_type="restricted_zone_entry",
                            label=f"Restricted Zone Entry ({current_zone.name})",
                            description=(
                                f"Observed {det.class_name} entering configured restricted zone "
                                f"'{current_zone.name}' at {timestamp:.2f}s."
                            ),
                            start_time=timestamp,
                            end_time=timestamp,
                            frame_index=frame_idx,
                            confidence=det.confidence,
                            uncertainty_note="Boundary crossing based on optical bounding-box centroid.",
                            related_track_id=tid,
                            metadata={"zone_id": current_zone.zone_id, "class": det.class_name, "bbox": det.bbox},
                        )
                    )

            # Rule 2: Repeated zone crossings
            if curr_zone_id != prev_zone_id and curr_zone_id is not None:
                if tid not in self.track_zone_crossings:
                    self.track_zone_crossings[tid] = []
                self.track_zone_crossings[tid].append((timestamp, curr_zone_id))

                # Count crossings within last 30 seconds
                recent_crossings = [
                    t for t, zid in self.track_zone_crossings[tid] if timestamp - t <= 30.0
                ]
                if (
                    len(recent_crossings) >= self.config.repeated_crossing_threshold
                    and tid not in self._reported_crossings
                ):
                    self._reported_crossings.add(tid)
                    flagged_detections[tid] = "Repeated Crossing"
                    is_key_evidence = True
                    self.events.append(
                        InvestigationEvent(
                            event_id=str(uuid.uuid4()),
                            event_type="repeated_zone_crossing",
                            label="Repeated Zone Crossing Indicator",
                            description=(
                                f"{det.class_name.capitalize()} crossed between monitored zones "
                                f"{len(recent_crossings)} times within 30 seconds."
                            ),
                            start_time=recent_crossings[0],
                            end_time=timestamp,
                            frame_index=frame_idx,
                            confidence=det.confidence,
                            uncertainty_note="Multiple zone transitions detected; check camera angle coverage.",
                            related_track_id=tid,
                            metadata={"crossings_count": len(recent_crossings)},
                        )
                    )

            self.track_current_zone[tid] = curr_zone_id

            # Rule 3: Unusual stationary stopping
            is_stopped = False
            hist = self.track_positions[tid]
            if len(hist) >= 15:
                # Calculate displacement over last 1.0 second
                d_x = (cx - hist[0][1]) * self.width
                d_y = (cy - hist[0][2]) * self.height
                displacement = np.hypot(d_x, d_y)
                if displacement < 10.0:  # < 10 pixels displacement
                    is_stopped = True

            if is_stopped:
                if tid not in self.stop_tracking:
                    self.stop_tracking[tid] = timestamp
                else:
                    duration = timestamp - self.stop_tracking[tid]
                    if (
                        duration >= self.config.stop_duration_threshold_seconds
                        and tid not in self._reported_stops
                    ):
                        self._reported_stops.add(tid)
                        flagged_detections[tid] = "Stationary Stop"
                        is_key_evidence = True
                        self.events.append(
                            InvestigationEvent(
                                event_id=str(uuid.uuid4()),
                                event_type="unusual_stationary_stop",
                                label=f"Unusual Stationary Stop ({duration:.1f}s)",
                                description=(
                                    f"{det.class_name.capitalize()} observed stationary in monitored area for "
                                    f"{duration:.1f}s (threshold: {self.config.stop_duration_threshold_seconds}s)."
                                ),
                                start_time=self.stop_tracking[tid],
                                end_time=timestamp,
                                frame_index=frame_idx,
                                confidence=det.confidence,
                                uncertainty_note="Static bounding box observed without significant displacement.",
                                related_track_id=tid,
                                metadata={"duration_seconds": round(duration, 1)},
                            )
                        )
            else:
                self.stop_tracking.pop(tid, None)

        # Render frame with zone overlays and alert badges
        annotated = draw_zone_overlay(frame_bgr, self.zones)
        for det in all_detections:
            tid = det.track_id
            if tid is not None and tid in flagged_detections:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                alert_text = flagged_detections[tid]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    annotated,
                    f"! {alert_text}",
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        return annotated, is_key_evidence
