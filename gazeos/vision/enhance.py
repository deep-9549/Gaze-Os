"""Image enhancement and network input preparation (PRD 9.5)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def apply_clahe(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return np.asarray(_CLAHE.apply(gray), dtype=np.uint8)


def face_to_tensor(face_rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """(H, W, 3) uint8 RGB -> (1, 3, H, W) float32, ImageNet-normalized."""
    x = face_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def eye_to_tensor(eye_gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    """(36, 60) uint8 -> (1, 1, 36, 60) float32 in [-1, 1]."""
    x = eye_gray.astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(x[None, None], dtype=np.float32)


def mean_brightness(image_rgb: NDArray[np.uint8], box: tuple[int, int, int, int] | None = None) -> float:
    """Mean luma of an image region (x0, y0, x1, y1), 0-255."""
    region = image_rgb
    if box is not None:
        x0, y0, x1, y1 = box
        h, w = image_rgb.shape[:2]
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        region = image_rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())
