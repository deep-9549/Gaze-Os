from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from gazeos.settings import QualitySettings
from gazeos.types import TrackingStatus
from gazeos.vision import face_model as fm
from gazeos.vision.features import ear_pair, eye_aspect_ratio, gesture_features
from gazeos.vision.geometry import (
    angular_error_deg,
    euler_to_rotation,
    pitchyaw_to_vector,
    rotation_to_euler,
    vector_to_pitchyaw,
)
from gazeos.vision.head_pose import HeadPoseEstimator, project_points
from gazeos.vision.normalization import (
    NORM_DISTANCE_MM,
    normalization_rotation,
    normalize,
    warp_matrix,
)
from gazeos.vision.quality import QualityGate

angles = st.floats(min_value=-1.2, max_value=1.2, allow_nan=False)


@given(angles, angles)
def test_pitchyaw_roundtrip(pitch: float, yaw: float) -> None:
    v = pitchyaw_to_vector(np.array([pitch, yaw]))
    assert np.isclose(np.linalg.norm(v), 1.0)
    back = vector_to_pitchyaw(v)
    assert np.allclose(back, [pitch, yaw], atol=1e-9)


def test_looking_at_camera_is_minus_z() -> None:
    assert np.allclose(pitchyaw_to_vector(np.array([0.0, 0.0])), [0, 0, -1])
    # Looking up has a negative y component (y points down in the camera frame).
    assert pitchyaw_to_vector(np.array([0.3, 0.0]))[1] < 0


def test_angular_error() -> None:
    a = np.array([[0.0, 0.0]])
    b = np.array([[0.0, math.radians(10)]])
    assert np.isclose(angular_error_deg(a, b)[0], 10.0)


@given(st.floats(-0.6, 0.6), st.floats(-0.6, 0.6), st.floats(-0.6, 0.6))
def test_euler_roundtrip(p: float, y: float, r: float) -> None:
    R = euler_to_rotation(p, y, r)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.allclose(rotation_to_euler(R), (p, y, r), atol=1e-9)


@settings(max_examples=40)
@given(
    st.floats(-300, 300), st.floats(-200, 300), st.floats(350, 900),
    st.floats(-0.5, 0.5), st.floats(-0.5, 0.5), st.floats(-0.4, 0.4),
    angles, angles,
)
def test_normalize_denormalize_gaze_roundtrip(x, y, z, hp, hy, hr, gp, gy) -> None:
    """PRD 23: normalizing then de-normalizing returns the original gaze vector."""
    c = np.array([x, y, z])
    R = euler_to_rotation(hp, hy, hr)
    R_n = normalization_rotation(c, R)
    assert np.allclose(R_n @ R_n.T, np.eye(3), atol=1e-9)
    # z-axis of the virtual camera points at the face centre.
    assert np.allclose(R_n[2], c / np.linalg.norm(c))
    g = pitchyaw_to_vector(np.array([gp, gy]))
    assert np.allclose(R_n.T @ (R_n @ g), g, atol=1e-9)


def test_normalization_removes_roll(intrinsics) -> None:
    c = np.array([0.0, 0.0, 600.0])
    for roll in (-0.4, 0.0, 0.4):
        R = euler_to_rotation(0.0, 0.0, roll)
        R_n = normalization_rotation(c, R)
        _, _, nroll = rotation_to_euler(R_n @ R)
        assert abs(nroll) < 1e-9


def test_warp_maps_face_centre_to_image_centre(intrinsics) -> None:
    c = np.array([40.0, 60.0, 700.0])
    W, R_n, S = warp_matrix(c, np.eye(3), intrinsics)
    uv = project_points(c[None], intrinsics)[0]
    out = W @ np.array([uv[0], uv[1], 1.0])
    assert np.allclose(out[:2] / out[2], [112, 112], atol=1e-6)
    assert np.isclose(S[2, 2], NORM_DISTANCE_MM / np.linalg.norm(c))


def test_head_pose_recovers_synthetic_pose(user, intrinsics) -> None:
    user.head_pyr_deg = (8.0, -12.0, 5.0)
    lm = user.look_at(900, 500)
    est = HeadPoseEstimator(intrinsics)
    pose = est.estimate(lm.landmarks_px)
    R_true, t_true = user.head_pose()
    assert pose is not None
    assert np.allclose(pose.t, t_true, atol=2.0)
    assert np.allclose(pose.R, R_true, atol=1e-2)
    assert pose.reprojection_error_px < 0.5


def test_normalize_produces_crops(user, intrinsics) -> None:
    lm = user.look_at(960, 540)
    pose = HeadPoseEstimator(intrinsics).estimate(lm.landmarks_px)
    image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    norm = normalize(image, lm.landmarks_px, pose, intrinsics)
    assert norm.face.shape == (224, 224, 3)
    assert norm.right_eye.shape == (36, 60) and norm.left_eye.shape == (36, 60)


def test_ear_open_vs_closed(user) -> None:
    open_l, open_r = ear_pair(user.look_at(960, 540).landmarks_px)
    closed_l, closed_r = ear_pair(user.look_at(960, 540, open_left=0.0, open_right=0.0).landmarks_px)
    assert open_l > 0.25 and open_r > 0.25
    assert closed_l < 0.1 and closed_r < 0.1
    assert eye_aspect_ratio(np.zeros((478, 3)), fm.LEFT_EYE_EAR) == 0.0


def test_gesture_feature_size(user) -> None:
    lm = user.look_at(960, 540)
    f = gesture_features(lm.landmarks_px, lm.blendshapes, (0.0, 0.0, 0.0))
    assert f.shape == (17,)


def test_quality_gates() -> None:
    g = QualityGate(QualitySettings())
    assert g.check(0.0, 1, inter_eye_px=100).ok
    assert g.check(0.0, 2, inter_eye_px=100).status == TrackingStatus.MULTIPLE_FACES
    assert g.check(0.0, 1, inter_eye_px=40).status == TrackingStatus.MOVE_CLOSER
    assert g.check(0.0, 1, inter_eye_px=100, brightness=20).status == TrackingStatus.LOW_LIGHT
    r = g.check(0.0, 1, inter_eye_px=100, head_yaw=math.radians(50))
    assert not r.ok and r.hold_cursor
    assert g.check(0.0, 1, inter_eye_px=100, blink_score=0.9).eye_closed
    # Auto-pause only after the face has been lost for more than a second.
    assert not g.check(0.5, 0).auto_pause
    assert g.check(1.2, 0).auto_pause
