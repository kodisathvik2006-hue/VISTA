from __future__ import annotations

import cv2
import numpy as np


COLOR_NAMES = [
    "black",
    "white",
    "silver/gray",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
]


def extract_dominant_color(crop_bgr: np.ndarray) -> str:
    """Extract visually dominant color from a bounding-box crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"

    h, w = crop_bgr.shape[:2]
    if h < 5 or w < 5:
        return "unknown"

    # Focus on central 60% of crop to reduce road/sky background influence
    y1 = int(h * 0.2)
    y2 = max(y1 + 1, int(h * 0.8))
    x1 = int(w * 0.2)
    x2 = max(x1 + 1, int(w * 0.8))
    center = crop_bgr[y1:y2, x1:x2]

    # Convert to HSV
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    total_pixels = center.shape[0] * center.shape[1]
    if total_pixels == 0:
        return "unknown"

    # Categorize pixels
    black_mask = v_chan < 50
    white_mask = (v_chan > 190) & (s_chan < 40)
    gray_mask = (s_chan < 45) & (~black_mask) & (~white_mask)

    chroma_mask = ~(black_mask | white_mask | gray_mask)

    red_mask = chroma_mask & ((h_chan < 10) | (h_chan >= 165))
    orange_mask = chroma_mask & ((h_chan >= 10) & (h_chan < 25))
    yellow_mask = chroma_mask & ((h_chan >= 25) & (h_chan < 35))
    green_mask = chroma_mask & ((h_chan >= 35) & (h_chan < 85))
    blue_mask = chroma_mask & ((h_chan >= 85) & (h_chan < 135))
    purple_mask = chroma_mask & ((h_chan >= 135) & (h_chan < 165))

    counts = {
        "black": int(np.count_nonzero(black_mask)),
        "white": int(np.count_nonzero(white_mask)),
        "silver/gray": int(np.count_nonzero(gray_mask)),
        "red": int(np.count_nonzero(red_mask)),
        "orange": int(np.count_nonzero(orange_mask)),
        "yellow": int(np.count_nonzero(yellow_mask)),
        "green": int(np.count_nonzero(green_mask)),
        "blue": int(np.count_nonzero(blue_mask)),
        "purple": int(np.count_nonzero(purple_mask)),
    }

    dominant = max(counts.items(), key=lambda item: item[1])
    # If dominant color is less than 15% of pixels, return unknown
    if dominant[1] < total_pixels * 0.15:
        return "unspecified"

    return dominant[0]
