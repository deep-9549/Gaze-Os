"""MediaPipe Face Mesh landmark indices and a metric 3D face model.

The 3D points are in millimetres, in a head frame aligned with the camera frame
when the face looks straight at the camera (x to image right, y down, z away
from the camera). They are a generic adult face; per-user scale error mostly
affects distance, which normalization removes.
"""

from __future__ import annotations

import numpy as np

# Six stable landmarks used for solvePnP (PRD 9.3).
NOSE_TIP = 1
CHIN = 152
RIGHT_EYE_OUTER = 33  # subject's right eye, image left
RIGHT_EYE_INNER = 133
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
MOUTH_RIGHT = 61
MOUTH_LEFT = 291

PNP_INDICES: tuple[int, ...] = (
    NOSE_TIP,
    CHIN,
    RIGHT_EYE_OUTER,
    LEFT_EYE_OUTER,
    MOUTH_RIGHT,
    MOUTH_LEFT,
)

PNP_MODEL_MM = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip
        [0.0, 66.0, 13.0],  # chin
        [-45.0, -34.0, 27.0],  # right eye outer corner
        [45.0, -34.0, 27.0],  # left eye outer corner
        [-30.0, 30.0, 25.0],  # right mouth corner
        [30.0, 30.0, 25.0],  # left mouth corner
    ],
    dtype=np.float64,
)

# Extra model points used for the face centre and eye centres.
MODEL_POINTS_MM: dict[int, tuple[float, float, float]] = {
    NOSE_TIP: (0.0, 0.0, 0.0),
    RIGHT_EYE_OUTER: (-45.0, -34.0, 27.0),
    RIGHT_EYE_INNER: (-15.0, -33.0, 25.0),
    LEFT_EYE_INNER: (15.0, -33.0, 25.0),
    LEFT_EYE_OUTER: (45.0, -34.0, 27.0),
}

FACE_CENTER_INDICES = (RIGHT_EYE_OUTER, RIGHT_EYE_INNER, LEFT_EYE_INNER, LEFT_EYE_OUTER, NOSE_TIP)


def model_point(idx: int) -> np.ndarray:
    return np.asarray(MODEL_POINTS_MM[idx], dtype=np.float64)


FACE_CENTER_MODEL_MM = np.mean([model_point(i) for i in FACE_CENTER_INDICES], axis=0)
RIGHT_EYE_CENTER_MODEL_MM = (model_point(RIGHT_EYE_OUTER) + model_point(RIGHT_EYE_INNER)) / 2.0
LEFT_EYE_CENTER_MODEL_MM = (model_point(LEFT_EYE_OUTER) + model_point(LEFT_EYE_INNER)) / 2.0

# Six-point eye contours for the Eye Aspect Ratio, ordered p1..p6 (PRD 13.1):
# p1, p4 are the corners; p2, p3 on the upper lid; p5, p6 on the lower lid.
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)

# Iris centres (468-477 are the 10 iris points).
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
NUM_LANDMARKS = 478

# The 12 blendshapes used by the gesture model (PRD 13.1).
GESTURE_BLENDSHAPES: tuple[str, ...] = (
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "jawOpen",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthPucker",
    "cheekPuff",
)
