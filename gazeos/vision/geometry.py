"""Gaze-direction maths shared by runtime, training and evaluation.

Conventions (same as MPIIGaze / ETH-XGaze):
camera frame has x to the image right, y down, z forward out of the lens.
A gaze vector looking straight into the camera points along -z.
pitch > 0 looks up, yaw > 0 looks toward the camera's -x side.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def pitchyaw_to_vector(pitchyaw: ArrayLike) -> FloatArray:
    """(N, 2) or (2,) pitch/yaw in radians -> unit gaze vectors (N, 3) or (3,)."""
    py = np.asarray(pitchyaw, dtype=np.float64)
    single = py.ndim == 1
    py = py.reshape(-1, 2)
    pitch, yaw = py[:, 0], py[:, 1]
    out = np.stack(
        [-np.cos(pitch) * np.sin(yaw), -np.sin(pitch), -np.cos(pitch) * np.cos(yaw)], axis=1
    )
    return out[0] if single else out


def vector_to_pitchyaw(vectors: ArrayLike) -> FloatArray:
    """(N, 3) or (3,) gaze vectors (any length) -> (N, 2) or (2,) pitch/yaw in radians."""
    v = np.asarray(vectors, dtype=np.float64)
    single = v.ndim == 1
    v = v.reshape(-1, 3)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    pitch = np.arcsin(np.clip(-v[:, 1], -1.0, 1.0))
    yaw = np.arctan2(-v[:, 0], -v[:, 2])
    out = np.stack([pitch, yaw], axis=1)
    return out[0] if single else out


def angular_error_deg(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Angle in degrees between gaze vectors (or pitch/yaw pairs) a and b (PRD 17.1)."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.shape[-1] == 2:
        av = pitchyaw_to_vector(av)
    if bv.shape[-1] == 2:
        bv = pitchyaw_to_vector(bv)
    av = av.reshape(-1, 3)
    bv = bv.reshape(-1, 3)
    cos = np.sum(av * bv, axis=1) / (np.linalg.norm(av, axis=1) * np.linalg.norm(bv, axis=1))
    return np.asarray(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), dtype=np.float64)


def rotation_to_euler(R: FloatArray) -> tuple[float, float, float]:
    """Rotation matrix -> (pitch, yaw, roll) in radians, x-y-z intrinsic order."""
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return pitch, yaw, roll


def euler_to_rotation(pitch: float, yaw: float, roll: float) -> FloatArray:
    """Inverse of rotation_to_euler."""
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cz, sz = math.cos(roll), math.sin(roll)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def rotation_angle_deg(R_a: FloatArray, R_b: FloatArray) -> float:
    """Angle of the relative rotation between two rotation matrices, in degrees."""
    R = R_a.T @ R_b
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def deg_to_screen_cm(error_deg: float, distance_cm: float = 60.0) -> float:
    """On-screen error for an angular error at a viewing distance (PRD 3.3)."""
    return distance_cm * math.tan(math.radians(error_deg))


def head_direction_pitchyaw(R: FloatArray) -> tuple[float, float]:
    """Pitch/yaw of the direction the face points (the head's -z axis toward the camera)."""
    forward = -R[:, 2]
    py = vector_to_pitchyaw(forward)
    return float(py[0]), float(py[1])
