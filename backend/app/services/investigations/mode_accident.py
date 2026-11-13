from __future__ import annotations

import logging
import uuid
from typing import Sequence

import cv2
import numpy as np

from app.schemas import (
    AccidentDetectionConfig,
    InvestigationEvent,
    TrackedDetection,
)
from app.services.investigations.base import compute_iou

logger = logging.getLogger("neurovision.investigations.accident")

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


class AccidentDetector:
    """
    Independent accident-analysis pipeline using vehicle detections, trajectories,
    kinematic motion estimates, and Time-To-Collision (TTC) calculations.

    All results are labelled potential incidents requiring human review.
    """

    def __init__(self, config: AccidentDetectionConfig, fps: float, width: int, height: int) -> None:
        self.config = config
        self.fps = max(1.0, fps)
        self.width = width
        self.height = height

        # Trajectory history: track_id -> [(timestamp, cx, cy, vx, vy)]
        self.track_history: dict[int, list[tuple[float, float, float, float, float]]] = {}
        self.events: list[InvestigationEvent] = []
        self._reported_pairs: set[tuple[int, int]] = set()

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        frame_bgr: np.ndarray,
        all_detections: Sequence[TrackedDetection],
    ) -> tuple[np.ndarray, bool]:
        """Analyze vehicle kinematics for collisions and near-misses."""
        is_key_evidence = False
        annotated = frame_bgr.copy()
        flagged_boxes: list[tuple[list[float], str]] = []

        # Filter vehicle detections with track IDs
        vehicles = [d for d in all_detections if d.track_id is not None and d.class_name.lower() in VEHICLE_CLASSES]

        # 1. Update velocities
        current_kinematics: dict[int, tuple[float, float, float, float]] = {}  # tid -> (cx, cy, vx, vy)
        for v in vehicles:
            tid = v.track_id
            cx = (v.bbox[0] + v.bbox[2]) / 2.0
            cy = (v.bbox[1] + v.bbox[3]) / 2.0

            vx, vy = 0.0, 0.0
            if tid in self.track_history and len(self.track_history[tid]) > 0:
                prev_t, prev_cx, prev_cy, _, _ = self.track_history[tid][-1]
                dt = max(0.001, timestamp - prev_t)
                vx = (cx - prev_cx) / dt
                vy = (cy - prev_cy) / dt

            current_kinematics[tid] = (cx, cy, vx, vy)
            if tid not in self.track_history:
                self.track_history[tid] = []
            self.track_history[tid].append((timestamp, cx, cy, vx, vy))
            if len(self.track_history[tid]) > 30:
                self.track_history[tid].pop(0)

        # 2. Pairwise collision / TTC check
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1 = vehicles[i]
                v2 = vehicles[j]
                tid1, tid2 = v1.track_id, v2.track_id
                pair_key = (min(tid1, tid2), max(tid1, tid2))

                if pair_key in self._reported_pairs:
                    continue

                cx1, cy1, vx1, vy1 = current_kinematics[tid1]
                cx2, cy2, vx2, vy2 = current_kinematics[tid2]

                dx = cx2 - cx1
                dy = cy2 - cy1
                dist = np.hypot(dx, dy)

                # Relative velocity vector
                rel_vx = vx1 - vx2
                rel_vy = vy1 - vy2
                closing_speed = (dx * rel_vx + dy * rel_vy) / (dist + 1e-6)

                # Check box overlap (direct physical contact / collision)
                iou = compute_iou(v1.bbox, v2.bbox)

                if iou > 0.08:
                    # Detected bbox collision
                    self._reported_pairs.add(pair_key)
                    is_key_evidence = True
                    flagged_boxes.append((v1.bbox, "POTENTIAL IMPACT"))
                    flagged_boxes.append((v2.bbox, "POTENTIAL IMPACT"))

                    self.events.append(
                        InvestigationEvent(
                            event_id=str(uuid.uuid4()),
                            event_type="potential_collision_impact",
                            label="Potential Collision Impact Flagged",
                            description=(
                                f"Spatial overlap (IoU: {iou:.2f}) observed between {v1.class_name} #{tid1} "
                                f"and {v2.class_name} #{tid2} at {timestamp:.2f}s."
                            ),
                            start_time=timestamp,
                            end_time=timestamp + 1.0,
                            frame_index=frame_idx,
                            confidence=round((v1.confidence + v2.confidence) / 2.0, 3),
                            uncertainty_note="Optical bounding box intersection; 2D perspective overlap can occur without contact. Requires human review.",
                            related_track_id=tid1,
                            metadata={"track_ids": [tid1, tid2], "iou": round(iou, 3)},
                        )
                    )
                elif closing_speed > 25.0 and dist < 120.0:
                    # Time-to-Collision
                    ttc = dist / closing_speed
                    if ttc <= self.config.ttc_threshold_seconds:
                        self._reported_pairs.add(pair_key)
                        is_key_evidence = True
                        flagged_boxes.append((v1.bbox, f"TTC: {ttc:.1f}s"))
                        flagged_boxes.append((v2.bbox, f"TTC: {ttc:.1f}s"))

                        self.events.append(
                            InvestigationEvent(
                                event_id=str(uuid.uuid4()),
                                event_type="near_miss_convergence",
                                label="High-Risk Convergence (Near Miss / Hazard)",
                                description=(
                                    f"Estimated Time-to-Collision (TTC) of {ttc:.2f}s calculated between "
                                    f"{v1.class_name} #{tid1} and {v2.class_name} #{tid2} "
                                    f"(closing speed: {closing_speed:.1f} px/s, dist: {dist:.1f} px)."
                                ),
                                start_time=timestamp,
                                end_time=timestamp + ttc,
                                frame_index=frame_idx,
                                confidence=round((v1.confidence + v2.confidence) / 2.0, 3),
                                uncertainty_note="Kinematic estimate in image coordinates without depth camera calibration.",
                                related_track_id=tid1,
                                metadata={"ttc_seconds": round(ttc, 2), "track_ids": [tid1, tid2]},
                            )
                        )

        # Draw warnings on annotated frame
        for bbox, label in flagged_boxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 69, 255), 3)
            cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + 180, y1), (0, 69, 255), -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated, is_key_evidence
