"""Synthetic face landmarks for tests, replay fixtures and the --simulate mode.

Builds MediaPipe-compatible landmark arrays from a head pose and a gaze target,
using the same metric face model as head-pose estimation, with an eyeball model
for the iris. A personal visual-axis offset (kappa) and per-frame noise make it
a fair test bed for calibration and filtering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.vision import face_model as fm
from gazeos.vision.geometry import euler_to_rotation, pitchyaw_to_vector, vector_to_pitchyaw
from gazeos.vision.head_pose import CameraIntrinsics, project_points
from gazeos.vision.landmarks import LandmarkResult

FloatArray = NDArray[np.float64]

EYEBALL_RADIUS_MM = 12.0

# Head-frame 3D positions (mm) for landmarks the pipeline reads.
_EYE_HALF_WIDTH = 15.0


def _eye_points(center: FloatArray, openness: float) -> dict[str, FloatArray]:
    h = 4.5 * openness + 0.3
    cx, cy, cz = center
    return {
        "outer_up": np.array([cx - 7.0, cy - h, cz - 1.0]),
        "inner_up": np.array([cx + 7.0, cy - h, cz - 1.0]),
        "outer_down": np.array([cx - 7.0, cy + h, cz - 1.0]),
        "inner_down": np.array([cx + 7.0, cy + h, cz - 1.0]),
        "mid_up": np.array([cx, cy - h * 1.1, cz - 1.5]),
        "mid_down": np.array([cx, cy + h * 1.1, cz - 1.5]),
    }


@dataclass
class SyntheticFace:
    intrinsics: CameraIntrinsics
    image_size: tuple[int, int] = (640, 480)
    kappa_deg: tuple[float, float] = (1.5, -2.0)  # personal visual-axis offset (pitch, yaw)
    noise_px: float = 0.4
    seed: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def model_points(
        self, gaze_head: FloatArray, open_left: float, open_right: float
    ) -> dict[int, FloatArray]:
        pts: dict[int, FloatArray] = {}
        for idx, p in zip(fm.PNP_INDICES, fm.PNP_MODEL_MM, strict=True):
            pts[idx] = p.copy()
        for idx, p in fm.MODEL_POINTS_MM.items():
            pts[idx] = np.asarray(p, dtype=np.float64)
        for side, center, openness in (
            ("right", fm.RIGHT_EYE_CENTER_MODEL_MM, open_right),
            ("left", fm.LEFT_EYE_CENTER_MODEL_MM, open_left),
        ):
            e = _eye_points(center, openness)
            if side == "right":
                # p1..p6 = 33, 160, 158, 133, 153, 144 (outer is image-left)
                pts[160], pts[158] = e["outer_up"], e["inner_up"]
                pts[153], pts[144] = e["inner_down"], e["outer_down"]
                pts[159], pts[145] = e["mid_up"], e["mid_down"]
                iris_idx = fm.RIGHT_IRIS_CENTER
            else:
                # 362, 385, 387, 263, 373, 380 (outer is image-right)
                mirror = e
                pts[385], pts[387] = mirror["inner_up"], mirror["outer_up"]
                pts[373], pts[380] = mirror["outer_down"], mirror["inner_down"]
                pts[386], pts[374] = mirror["mid_up"], mirror["mid_down"]
                iris_idx = fm.LEFT_IRIS_CENTER
            eyeball = center + np.array([0.0, 0.0, EYEBALL_RADIUS_MM])
            iris = eyeball + EYEBALL_RADIUS_MM * gaze_head
            for k in range(5):
                pts[iris_idx + k] = iris + (np.array([math.cos(k), math.sin(k), 0.0]) * (0 if k == 0 else 5.5))
        return pts

    def landmarks(
        self,
        head_R: FloatArray,
        head_t: FloatArray,
        target_cam: FloatArray | None = None,
        gaze_cam: FloatArray | None = None,
        open_left: float = 1.0,
        open_right: float = 1.0,
    ) -> tuple[LandmarkResult, FloatArray]:
        """Returns the landmark result and the true visual-axis gaze vector (camera frame).

        Give either a 3D target in camera mm or a gaze direction.
        """
        right_c = head_R @ fm.RIGHT_EYE_CENTER_MODEL_MM + head_t
        left_c = head_R @ fm.LEFT_EYE_CENTER_MODEL_MM + head_t
        origin = (right_c + left_c) / 2.0
        if gaze_cam is None:
            if target_cam is None:
                raise ValueError("need a target or a gaze direction")
            gaze_cam = target_cam - origin
        gaze_cam = gaze_cam / np.linalg.norm(gaze_cam)
        # The camera sees the optical axis: visual axis minus the personal kappa offset.
        py = vector_to_pitchyaw(gaze_cam) - np.radians(np.asarray(self.kappa_deg))
        optical_cam = pitchyaw_to_vector(py)
        gaze_head = head_R.T @ optical_cam

        model = self.model_points(gaze_head, open_left, open_right)
        idx = np.array(sorted(model))
        pts_head = np.stack([model[i] for i in idx])
        cam_pts = (head_R @ pts_head.T).T + head_t
        uv = project_points(cam_pts, self.intrinsics)
        uv += self.rng.normal(0.0, self.noise_px, uv.shape)

        lm = np.zeros((fm.NUM_LANDMARKS, 3))
        # Fill unused landmarks on an ellipse around the face for a sane bounding box.
        nose = project_points((head_R @ np.zeros(3) + head_t)[None], self.intrinsics)[0]
        scale = self.intrinsics.matrix[0, 0] / head_t[2]
        angles = np.linspace(0, 2 * np.pi, fm.NUM_LANDMARKS, endpoint=False)
        lm[:, 0] = nose[0] + 75.0 * scale * np.cos(angles)
        lm[:, 1] = nose[1] - 10.0 * scale + 100.0 * scale * np.sin(angles)
        lm[idx, :2] = uv
        lm[idx, 2] = cam_pts[:, 2] - head_t[2]
        shapes = {
            "eyeBlinkLeft": float(np.clip(1.0 - open_left, 0, 1)),
            "eyeBlinkRight": float(np.clip(1.0 - open_right, 0, 1)),
            "browInnerUp": 0.05,
            "jawOpen": 0.02,
        }
        return LandmarkResult(num_faces=1, landmarks_px=lm, blendshapes=shapes, transform=None), gaze_cam


@dataclass
class SimulatedUser:
    """A user sitting in front of a screen, looking at targets."""

    geometry: ScreenGeometry
    face: SyntheticFace
    distance_mm: float = 600.0
    head_pyr_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    head_offset_mm: tuple[float, float] = (0.0, 120.0)  # face below the camera, roughly screen centre

    def head_pose(self) -> tuple[FloatArray, FloatArray]:
        p, y, r = (math.radians(a) for a in self.head_pyr_deg)
        # Face toward the camera, then apply the user's own head rotation.
        R = euler_to_rotation(p, y, r)
        t = np.array([self.head_offset_mm[0], self.head_offset_mm[1], self.distance_mm])
        return R, t

    def look_at(self, x_px: float, y_px: float, open_left: float = 1.0, open_right: float = 1.0) -> LandmarkResult:
        R, t = self.head_pose()
        target = self.geometry.screen_mm_to_camera(*self.geometry.px_to_mm(x_px, y_px))
        lm, _ = self.face.landmarks(R, t, target_cam=target, open_left=open_left, open_right=open_right)
        return lm
