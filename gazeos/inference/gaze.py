"""Gaze estimators used at runtime (PRD 10).

- OnnxGazeEstimator runs an exported GazeNet-GOS / Lite / L2CS model with
  ONNX Runtime on the CPU (4 intra-op threads, PRD 10.8).
- GeometricGazeEstimator is the classical M0 baseline: iris offset from the
  eye-corner midpoint plus head pose. It needs no trained network, so the app
  works end to end from day one; calibration (P2-P3) then corrects its bias.

All estimators return gaze in the camera frame after de-normalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gazeos.vision import face_model as fm
from gazeos.vision.enhance import eye_to_tensor, face_to_tensor
from gazeos.vision.geometry import vector_to_pitchyaw
from gazeos.vision.head_pose import HeadPose
from gazeos.vision.normalization import NormalizationResult

FloatArray = NDArray[np.float64]

NUM_BINS = 28
BIN_WIDTH_DEG = 3.0
BIN_CENTERS_DEG = np.arange(NUM_BINS) * BIN_WIDTH_DEG - 42.0 + BIN_WIDTH_DEG / 2.0


@dataclass
class GazeEstimate:
    pitchyaw: FloatArray  # (2,) camera frame, radians
    pitchyaw_norm: FloatArray  # (2,) normalized frame, radians
    sigma: FloatArray  # (2,) predicted std, radians
    features: FloatArray | None = None
    attention: FloatArray | None = None  # fusion attention weights, for the debug view


def softmax(z: NDArray[np.floating], axis: int = -1) -> NDArray[np.float64]:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / e.sum(axis=axis, keepdims=True), dtype=np.float64)


def decode_bins(logits: NDArray[np.floating], centers_deg: FloatArray = BIN_CENTERS_DEG) -> FloatArray:
    """Classify-then-regress: softmax-weighted average of bin centres, in radians (PRD 10.3)."""
    probs = softmax(logits)
    return np.asarray(np.radians(probs @ centers_deg), dtype=np.float64)


class OnnxGazeEstimator:
    """Wraps any exported gaze ONNX model following the GazeOS I/O contract.

    Inputs (by name, all optional except face): face (1,3,S,S), right_eye and
    left_eye (1,1,36,60), head (1,3). Outputs: pitch_logits and yaw_logits
    (1,K); optional log_var (1,2), features (1,D), attention (1,T,T).
    """

    def __init__(self, path: str | Path, threads: int = 4, bin_width_deg: float | None = None) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name: i for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]
        face = self.input_names["face"]
        self.face_size = int(face.shape[2]) if isinstance(face.shape[2], int) else 224
        meta = self.session.get_modelmeta().custom_metadata_map
        k = int(meta.get("num_bins", NUM_BINS))
        width = float(bin_width_deg or meta.get("bin_width_deg", BIN_WIDTH_DEG))
        lo = float(meta.get("bin_min_deg", -k * width / 2.0))
        self.centers = np.arange(k) * width + lo + width / 2.0

    def estimate(self, norm: NormalizationResult) -> GazeEstimate:
        import cv2

        face = norm.face
        if face.shape[0] != self.face_size:
            face = cv2.resize(face, (self.face_size, self.face_size), interpolation=cv2.INTER_LINEAR)
        feeds: dict[str, NDArray[np.float32]] = {"face": face_to_tensor(face)}
        if "right_eye" in self.input_names:
            feeds["right_eye"] = eye_to_tensor(norm.right_eye)
            feeds["left_eye"] = eye_to_tensor(norm.left_eye)
        if "head" in self.input_names:
            feeds["head"] = np.asarray([norm.head_pyr], dtype=np.float32)
        outs = dict(zip(self.output_names, self.session.run(self.output_names, feeds), strict=True))
        pitch = float(decode_bins(outs["pitch_logits"][0], self.centers))
        yaw = float(decode_bins(outs["yaw_logits"][0], self.centers))
        if "log_var" in outs:
            sigma = np.sqrt(np.exp(np.asarray(outs["log_var"][0], dtype=np.float64)))
        else:
            sigma = np.radians(np.array([3.0, 3.0]))
        py_n = np.array([pitch, yaw])
        cam = np.asarray(norm.denormalize_pitchyaw((pitch, yaw)), dtype=np.float64)
        feats = np.asarray(outs["features"][0], dtype=np.float64) if "features" in outs else None
        attn = np.asarray(outs["attention"][0], dtype=np.float64) if "attention" in outs else None
        return GazeEstimate(cam, py_n, sigma, feats, attn)


class GeometricGazeEstimator:
    """M0: pupil-centre method. Eye-in-head angle from the iris offset, plus head pose.

    The eyeball is modelled as a 12 mm sphere; the iris offset from the
    eye-corner midpoint, as a fraction of half the eye width (about 15 mm),
    gives sin(angle) * 12 / 15.
    """

    EYEBALL_RADIUS_MM = 12.0
    HALF_EYE_WIDTH_MM = 15.0

    def eye_angles(self, landmarks: FloatArray) -> tuple[float, float]:
        """(pitch, yaw) of the eyes relative to the head, averaged over both eyes."""
        out = []
        for outer, inner, iris, top, bottom in (
            (fm.RIGHT_EYE_OUTER, fm.RIGHT_EYE_INNER, fm.RIGHT_IRIS_CENTER, 159, 145),
            (fm.LEFT_EYE_OUTER, fm.LEFT_EYE_INNER, fm.LEFT_IRIS_CENTER, 386, 374),
        ):
            a, b = landmarks[outer, :2], landmarks[inner, :2]
            mid = (a + b) / 2.0
            axis = b - a
            half = float(np.linalg.norm(axis)) / 2.0
            if half < 1e-6:
                continue
            ux = axis / (2 * half)
            # The right eye's outer->inner axis points to image right, the left
            # eye's to image left: flip the left one so ux is always image right
            # and uy (perpendicular) always points down.
            if outer == fm.LEFT_EYE_OUTER:
                ux = -ux
            uy = np.array([-ux[1], ux[0]])
            off_x = float((landmarks[iris, :2] - mid) @ ux)
            # Lids sit higher than the corner line: use the lid midpoint vertically.
            lid_mid = (landmarks[top, :2] + landmarks[bottom, :2]) / 2.0
            off_y = float((landmarks[iris, :2] - lid_mid) @ uy)
            scale = self.HALF_EYE_WIDTH_MM / self.EYEBALL_RADIUS_MM / half
            sx = float(np.clip(off_x * scale, -0.95, 0.95))
            sy = float(np.clip(off_y * scale, -0.95, 0.95))
            # Iris to image right (camera +x) means looking toward camera +x: yaw negative.
            out.append((-math.asin(sy), -math.asin(sx)))
        if not out:
            return 0.0, 0.0
        arr = np.asarray(out)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())

    def estimate(self, landmarks: FloatArray, pose: HeadPose) -> GazeEstimate:
        from gazeos.vision.geometry import pitchyaw_to_vector

        ep, ey = self.eye_angles(landmarks)
        # Eye-in-head direction rotated by the head pose into the camera frame.
        g_head = pitchyaw_to_vector(np.array([ep, ey]))
        g_cam = pose.R @ g_head
        py = vector_to_pitchyaw(g_cam)
        return GazeEstimate(
            pitchyaw=np.asarray(py, dtype=np.float64),
            pitchyaw_norm=np.array([ep, ey]),
            sigma=np.radians(np.array([4.0, 4.0])),
            features=np.array([ep, ey, *vector_to_pitchyaw(-pose.R[:, 2])]),
        )
