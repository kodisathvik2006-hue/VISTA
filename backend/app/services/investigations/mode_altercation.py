from __future__ import annotations

import logging
import uuid
from typing import Sequence

import cv2
import numpy as np

from app.schemas import (
    AltercationDetectionConfig,
    InvestigationEvent,
    TrackedDetection,
)

logger = logging.getLogger("neurovision.investigations.altercation")


class AltercationActionRecognitionInterface:
    """
    Dedicated action-recognition interface.

    Prepared for temporal transformer or 3D CNN models (such as VideoMAE or SlowFast).
    Transparent engineering: YOLO alone detects objects, not human intent or complex
    physical altercations. This pipeline tracks multi-person spatial proximity and
    abrupt bounding-box aspect/centroid jitter as preliminary triggers, and clearly marks
    all findings as potential indicators requiring human verification.
    """

    def __init__(self, config: AltercationDetectionConfig, fps: float, width: int, height: int) -> None:
        self.config = config
        self.fps = max(1.0, fps)
        self.width = width
        self.height = height

        # Person centroids & box dimensions history: track_id -> [(t, cx, cy, w, h)]
        self.person_history: dict[int, list[tuple[float, float, float, float, float]]] = {}
        self.events: list[InvestigationEvent] = []
        self._reported_pairs: set[tuple[int, int]] = set()

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        frame_bgr: np.ndarray,
        all_detections: Sequence[TrackedDetection],
    ) -> tuple[np.ndarray, bool]:
        """Analyze multi-person interactions."""
        is_key_evidence = False
        annotated = frame_bgr.copy()
        flagged_boxes: list[list[float]] = []

        # Filter person detections
        persons = [d for d in all_detections if d.class_name.lower() == "person" and d.track_id is not None]

        # Record person positions
        for p in persons:
            tid = p.track_id
            cx = (p.bbox[0] + p.bbox[2]) / 2.0
            cy = (p.bbox[1] + p.bbox[3]) / 2.0
            bw = p.bbox[2] - p.bbox[0]
            bh = p.bbox[3] - p.bbox[1]

            if tid not in self.person_history:
                self.person_history[tid] = []
            self.person_history[tid].append((timestamp, cx, cy, bw, bh))
            if len(self.person_history[tid]) > 45:
                self.person_history[tid].pop(0)

        # Evaluate pairs of people in close proximity
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                tid1, tid2 = p1.track_id, p2.track_id
                pair_key = (min(tid1, tid2), max(tid1, tid2))

                if pair_key in self._reported_pairs:
                    continue

                cx1 = (p1.bbox[0] + p1.bbox[2]) / 2.0
                cy1 = (p1.bbox[1] + p1.bbox[3]) / 2.0
                cx2 = (p2.bbox[0] + p2.bbox[2]) / 2.0
                cy2 = (p2.bbox[1] + p2.bbox[3]) / 2.0

                dist = np.hypot(cx1 - cx2, cy1 - cy2)

                # Check proximity
                if dist < self.config.proximity_threshold_pixels:
                    # Check kinematic turbulence / jitter across recent frames
                    hist1 = self.person_history.get(tid1, [])
                    hist2 = self.person_history.get(tid2, [])

                    if len(hist1) >= 10 and len(hist2) >= 10:
                        # Compute variance in centroid displacement (rapid erratic movements)
                        v1 = [np.hypot(hist1[k][1] - hist1[k - 1][1], hist1[k][2] - hist1[k - 1][2]) for k in range(1, len(hist1))]
                        v2 = [np.hypot(hist2[k][1] - hist2[k - 1][1], hist2[k][2] - hist2[k - 1][2]) for k in range(1, len(hist2))]
                        energy = float(np.mean(v1) + np.mean(v2))

                        if energy > (self.config.motion_energy_threshold * 12.0):
                            self._reported_pairs.add(pair_key)
                            is_key_evidence = True
                            flagged_boxes.append(p1.bbox)
                            flagged_boxes.append(p2.bbox)

                            self.events.append(
                                InvestigationEvent(
                                    event_id=str(uuid.uuid4()),
                                    event_type="potential_altercation_indicator",
                                    label="Potential Physical Altercation Indicator",
                                    description=(
                                        f"High-energy spatial interaction observed between Person #{tid1} and "
                                        f"Person #{tid2} (distance: {dist:.1f}px, motion index: {energy:.1f})."
                                    ),
                                    start_time=timestamp,
                                    end_time=timestamp + 1.5,
                                    frame_index=frame_idx,
                                    confidence=round(min(0.85, (p1.confidence + p2.confidence) / 2.0), 3),
                                    uncertainty_note=(
                                        "Flagged via spatial proximity and motion variance. YOLO detects objects, "
                                        "not human intent. Requires human verification or temporal action recognition (VideoMAE/SlowFast)."
                                    ),
                                    related_track_id=tid1,
                                    metadata={
                                        "track_ids": [tid1, tid2],
                                        "separation_pixels": round(dist, 1),
                                        "motion_index": round(energy, 2),
                                        "pipeline_status": "Temporal Interface Active (Human Review Required)",
                                    },
                                )
                            )

        # Draw visual indicators
        for box in flagged_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 2)
            cv2.putText(
                annotated,
                "! ACTION REVIEW REQ",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated, is_key_evidence
