from __future__ import annotations

from typing import Sequence
import cv2
import numpy as np

from app.schemas import ZonePolygon


def point_in_polygon(x: float, y: float, polygon_points: Sequence[Sequence[float]]) -> bool:
    """Check if point (x, y) is inside polygon using ray casting."""
    if len(polygon_points) < 3:
        return False
    inside = False
    n = len(polygon_points)
    p1x, p1y = polygon_points[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    denom = area1 + area2 - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def draw_zone_overlay(frame: np.ndarray, zones: Sequence[ZonePolygon]) -> np.ndarray:
    """Draw semi-transparent polygon zones onto frame."""
    if not zones:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for zone in zones:
        if len(zone.points) < 3:
            continue
        pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone.points], np.int32)
        color = (30, 30, 220) if zone.is_restricted else (220, 160, 40)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2)
        # Label zone
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(frame, zone.name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
