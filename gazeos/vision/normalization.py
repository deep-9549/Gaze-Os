"""Data normalization (Zhang, Sugano and Bulling, ETRA 2018; PRD 9.4).

A virtual camera is rotated to look straight at the face centre from a fixed
distance, cancelling head roll and scale. The same code is used to prepare
training data and at runtime, so both see identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from gazeos.vision import face_model as fm
from gazeos.vision.geometry import pitchyaw_to_vector, rotation_to_euler, vector_to_pitchyaw
from gazeos.vision.head_pose import CameraIntrinsics, HeadPose

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]

NORM_DISTANCE_MM = 600.0
NORM_FOCAL_PX = 960.0
FACE_SIZE = 224
EYE_SIZE = (60, 36)  # (width, height), MPIIGaze convention


@dataclass
class NormalizationResult:
    face: UInt8Array  # (224, 224, 3) RGB
    right_eye: UInt8Array  # (36, 60) grayscale
    left_eye: UInt8Array  # (36, 60) grayscale, mirrored so both eyes share weights
    R_n: FloatArray  # (3, 3) rotation of the virtual camera
    S: FloatArray  # (3, 3) scale
    W: FloatArray  # (3, 3) image warp
    head_pyr: tuple[float, float, float]  # normalized head pose (pitch, yaw, roll)

    def normalize_gaze(self, gaze_vector: FloatArray) -> FloatArray:
        return np.asarray(self.R_n @ gaze_vector, dtype=np.float64)

    def denormalize_gaze(self, gaze_vector_n: FloatArray) -> FloatArray:
        return np.asarray(self.R_n.T @ gaze_vector_n, dtype=np.float64)

    def denormalize_pitchyaw(self, pitchyaw_n: tuple[float, float]) -> tuple[float, float]:
        g = self.denormalize_gaze(pitchyaw_to_vector(np.asarray(pitchyaw_n)))
        py = vector_to_pitchyaw(g)
        return float(py[0]), float(py[1])


def normalization_rotation(face_center: FloatArray, head_R: FloatArray) -> FloatArray:
    """R_n whose z-axis points from the camera to the face centre and whose
    x-axis is aligned with the head's x-axis (removes roll)."""
    z = face_center / np.linalg.norm(face_center)
    head_x = head_R[:, 0]
    y = np.cross(z, head_x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    return np.stack([x, y, z], axis=0)


def virtual_camera(size: int = FACE_SIZE, focal: float = NORM_FOCAL_PX) -> FloatArray:
    return np.array(
        [[focal, 0.0, size / 2.0], [0.0, focal, size / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def warp_matrix(
    face_center: FloatArray,
    head_R: FloatArray,
    intrinsics: CameraIntrinsics,
    distance_mm: float = NORM_DISTANCE_MM,
    size: int = FACE_SIZE,
    focal: float = NORM_FOCAL_PX,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Returns (W, R_n, S) with W = C_n S R_n C_r^-1."""
    R_n = normalization_rotation(face_center, head_R)
    S = np.diag([1.0, 1.0, distance_mm / float(np.linalg.norm(face_center))])
    C_n = virtual_camera(size, focal)
    W = C_n @ S @ R_n @ np.linalg.inv(intrinsics.matrix)
    return W, R_n, S


def _warp_points(W: FloatArray, pts: FloatArray) -> FloatArray:
    homo = np.concatenate([pts[:, :2], np.ones((len(pts), 1))], axis=1)
    out = (W @ homo.T).T
    return np.asarray(out[:, :2] / out[:, 2:3], dtype=np.float64)


def _crop_eye(
    gray: UInt8Array, corner_a: FloatArray, corner_b: FloatArray, mirror: bool
) -> UInt8Array:
    center = (corner_a + corner_b) / 2.0
    width = max(float(np.linalg.norm(corner_b - corner_a)) * 1.6, 8.0)
    height = width * EYE_SIZE[1] / EYE_SIZE[0]
    x0, y0 = center[0] - width / 2.0, center[1] - height / 2.0
    # Affine map from the crop window to the 60x36 output.
    sx, sy = EYE_SIZE[0] / width, EYE_SIZE[1] / height
    M = np.array([[sx, 0.0, -x0 * sx], [0.0, sy, -y0 * sy]], dtype=np.float64)
    eye = cv2.warpAffine(gray, M, EYE_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if mirror:
        eye = cv2.flip(eye, 1)
    return np.asarray(eye, dtype=np.uint8)


def normalize(
    image_rgb: UInt8Array,
    landmarks_px: FloatArray,
    pose: HeadPose,
    intrinsics: CameraIntrinsics,
    clahe: bool = True,
) -> NormalizationResult:
    """Produce the normalized 224x224 face and two 36x60 eye crops."""
    W, R_n, S = warp_matrix(pose.face_center, pose.R, intrinsics)
    face = np.asarray(cv2.warpPerspective(image_rgb, W, (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_LINEAR), dtype=np.uint8)
    gray = np.asarray(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
    pts = _warp_points(
        W,
        landmarks_px[
            [fm.RIGHT_EYE_OUTER, fm.RIGHT_EYE_INNER, fm.LEFT_EYE_INNER, fm.LEFT_EYE_OUTER]
        ],
    )
    right = _crop_eye(gray, pts[0], pts[1], mirror=False)
    left = _crop_eye(gray, pts[2], pts[3], mirror=True)
    if clahe:
        from gazeos.vision.enhance import apply_clahe

        right, left = apply_clahe(right), apply_clahe(left)
    head_pyr = rotation_to_euler(R_n @ pose.R)
    return NormalizationResult(
        face=np.asarray(face, dtype=np.uint8),
        right_eye=right,
        left_eye=left,
        R_n=R_n,
        S=S,
        W=W,
        head_pyr=head_pyr,
    )
