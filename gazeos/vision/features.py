"""Per-frame face features: Eye Aspect Ratio and the 17-d gesture feature (PRD 13.1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from gazeos.vision import face_model as fm

FloatArray = NDArray[np.float64]

NUM_GESTURE_FEATURES = 2 + len(fm.GESTURE_BLENDSHAPES) + 3  # 17


def eye_aspect_ratio(landmarks: FloatArray, indices: Sequence[int]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|) (Soukupová and Čech, 2016)."""
    p = landmarks[list(indices), :2]
    horizontal = float(np.linalg.norm(p[0] - p[3]))
    if horizontal < 1e-6:
        return 0.0
    vertical = float(np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4]))
    return vertical / (2.0 * horizontal)


def ear_pair(landmarks: FloatArray) -> tuple[float, float]:
    """(left EAR, right EAR) for the subject's left and right eye."""
    return (
        eye_aspect_ratio(landmarks, fm.LEFT_EYE_EAR),
        eye_aspect_ratio(landmarks, fm.RIGHT_EYE_EAR),
    )


def inter_eye_distance_px(landmarks: FloatArray) -> float:
    return float(
        np.linalg.norm(landmarks[fm.LEFT_EYE_OUTER, :2] - landmarks[fm.RIGHT_EYE_OUTER, :2])
    )


def gesture_features(
    landmarks: FloatArray,
    blendshapes: Mapping[str, float],
    head_pyr_delta: Sequence[float],
) -> FloatArray:
    """17 features: 2 EAR + 12 blendshapes + 3 head-pose deltas."""
    left, right = ear_pair(landmarks)
    shapes = [float(blendshapes.get(name, 0.0)) for name in fm.GESTURE_BLENDSHAPES]
    return np.asarray([left, right, *shapes, *head_pyr_delta], dtype=np.float64)


def face_bbox(landmarks: FloatArray) -> tuple[int, int, int, int]:
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
