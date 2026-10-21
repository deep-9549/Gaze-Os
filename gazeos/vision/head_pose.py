"""Camera model and head-pose estimation (PRD 9.2, 9.3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from gazeos.vision import face_model as fm
from gazeos.vision.geometry import rotation_angle_deg

FloatArray = NDArray[np.float64]


@dataclass
class CameraIntrinsics:
    matrix: FloatArray  # (3, 3)
    dist: FloatArray  # (5,) or empty

    @classmethod
    def default(cls, width: int, height: int) -> CameraIntrinsics:
        """fx = fy = image width, principal point at the centre, no distortion."""
        f = float(width)
        matrix = np.array(
            [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        return cls(matrix=matrix, dist=np.zeros(5, dtype=np.float64))

    @classmethod
    def load(cls, path: str | Path) -> CameraIntrinsics:
        data = json.loads(Path(path).read_text())
        return cls(
            matrix=np.asarray(data["matrix"], dtype=np.float64),
            dist=np.asarray(data["dist"], dtype=np.float64).ravel(),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"matrix": self.matrix.tolist(), "dist": self.dist.tolist()}, indent=2)
        )


@dataclass
class HeadPose:
    R: FloatArray  # (3, 3) head-to-camera rotation
    t: FloatArray  # (3,) head origin (nose tip) in camera coordinates, mm
    reprojection_error_px: float
    low_confidence: bool = False

    def transform(self, model_points_mm: FloatArray) -> FloatArray:
        """Head-model points (N, 3) -> camera coordinates (N, 3)."""
        pts = np.atleast_2d(model_points_mm)
        return np.asarray((self.R @ pts.T).T + self.t, dtype=np.float64)

    @property
    def face_center(self) -> FloatArray:
        return np.asarray(self.transform(fm.FACE_CENTER_MODEL_MM)[0], dtype=np.float64)

    @property
    def eye_centers(self) -> tuple[FloatArray, FloatArray]:
        """(right eye, left eye) centres in camera coordinates, mm."""
        right = np.asarray(self.transform(fm.RIGHT_EYE_CENTER_MODEL_MM)[0], dtype=np.float64)
        left = np.asarray(self.transform(fm.LEFT_EYE_CENTER_MODEL_MM)[0], dtype=np.float64)
        return right, left

    @property
    def gaze_origin(self) -> FloatArray:
        right, left = self.eye_centers
        return (right + left) / 2.0


class HeadPoseEstimator:
    """solvePnP between six landmarks and the metric face model."""

    def __init__(self, intrinsics: CameraIntrinsics, disagreement_deg: float = 10.0) -> None:
        self.intrinsics = intrinsics
        self.disagreement_deg = disagreement_deg
        self._rvec: FloatArray | None = None
        self._tvec: FloatArray | None = None

    def reset(self) -> None:
        self._rvec = None
        self._tvec = None

    def estimate(
        self,
        landmarks_px: FloatArray,
        mediapipe_matrix: FloatArray | None = None,
    ) -> HeadPose | None:
        """landmarks_px: (478, >=2) landmark pixel coordinates."""
        image_points = np.ascontiguousarray(
            landmarks_px[list(fm.PNP_INDICES), :2], dtype=np.float64
        )
        use_guess = self._rvec is not None
        ok, rvec, tvec = cv2.solvePnP(
            fm.PNP_MODEL_MM,
            image_points,
            self.intrinsics.matrix,
            self.intrinsics.dist,
            rvec=self._rvec.copy() if use_guess and self._rvec is not None else None,
            tvec=self._tvec.copy() if use_guess and self._tvec is not None else None,
            useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or tvec[2, 0] <= 0:
            self.reset()
            return None
        self._rvec = np.asarray(rvec, dtype=np.float64)
        self._tvec = np.asarray(tvec, dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        projected, _ = cv2.projectPoints(
            fm.PNP_MODEL_MM, rvec, tvec, self.intrinsics.matrix, self.intrinsics.dist
        )
        err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))
        pose = HeadPose(R=np.asarray(R, dtype=np.float64), t=tvec.ravel().astype(np.float64),
                        reprojection_error_px=err)
        if mediapipe_matrix is not None:
            pose.low_confidence = self.disagrees(pose.R, mediapipe_matrix)
        return pose

    def disagrees(self, R: FloatArray, mediapipe_matrix: FloatArray) -> bool:
        """Cross-check against MediaPipe's transformation matrix (PRD 9.3).

        MediaPipe's canonical frame has y up and z toward the camera, so it is
        converted to our y-down / z-away convention before comparing.
        """
        flip = np.diag([1.0, -1.0, -1.0])
        R_mp = flip @ np.asarray(mediapipe_matrix, dtype=np.float64)[:3, :3] @ flip
        u, _, vt = np.linalg.svd(R_mp)  # strip any scale
        R_mp = u @ vt
        return rotation_angle_deg(R, R_mp) > self.disagreement_deg


def project_points(
    points_cam_mm: FloatArray, intrinsics: CameraIntrinsics
) -> FloatArray:
    """Project camera-frame points (N, 3) to pixels (N, 2) without distortion."""
    pts = np.atleast_2d(points_cam_mm)
    uvw = (intrinsics.matrix @ pts.T).T
    return np.asarray(uvw[:, :2] / uvw[:, 2:3], dtype=np.float64)
