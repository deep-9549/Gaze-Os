from __future__ import annotations

import numpy as np
import pytest

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.calibration.implicit import DriftCorrector, GazeHistory, ImplicitRecalibrator
from gazeos.calibration.personalization import (
    METHODS,
    LastLayerFineTune,
    load_personalizer,
    make_personalizer,
)
from gazeos.calibration.protocol import (
    CalibrationSample,
    grid_targets,
    in_sample_window,
    mad_inliers,
    target_radius,
    validate,
    validation_targets,
)
from gazeos.pipeline import FrameProcessor


def _collect(proc, user, targets, frames=20, t0=0.0):
    t, out = t0, []
    for x, y in targets:
        for _ in range(frames):
            t += 1 / 30
            obs = proc.process_landmarks(user.look_at(x, y), t).observation
            if obs is not None:
                out.append(CalibrationSample(obs, np.array([x, y], dtype=float), t=t))
    return out, t


def test_grid_targets(geometry: ScreenGeometry) -> None:
    pts = grid_targets(geometry, 9)
    assert len(pts) == 9 and pts[0] == (0.1 * geometry.width_px, 0.1 * geometry.height_px)
    assert len(grid_targets(geometry, 16)) == 16
    assert not set(validation_targets(geometry)) & set(pts)
    with pytest.raises(ValueError):
        grid_targets(geometry, 5)


def test_sample_window_and_radius() -> None:
    assert not in_sample_window(0.3) and in_sample_window(0.8) and not in_sample_window(1.6)
    assert target_radius(0.0) > target_radius(1.5)


def test_mad_inliers_drops_outliers(rng) -> None:
    v = rng.normal(0, 1, (40, 2))
    v[5] = [30, 0]
    mask = mad_inliers(v)
    assert not mask[5] and mask.sum() >= 35


def test_screen_geometry_ray(geometry: ScreenGeometry) -> None:
    eye = np.array([0.0, 100.0, 600.0])
    target_mm = (100.0, 50.0)
    target_cam = geometry.screen_mm_to_camera(*target_mm)
    hit = geometry.intersect(eye, target_cam - eye)
    assert hit is not None and np.allclose(hit, target_mm)
    assert geometry.intersect(eye, np.array([0.0, 0.0, 1.0])) is None
    assert np.isclose(geometry.px_to_deg(geometry.deg_to_px(2.0)), 2.0)


def test_validation_score(geometry: ScreenGeometry) -> None:
    res = validate([(100, 100)], [(100, 100)], geometry)
    assert res.score == 100 and res.mean_px == 0
    far = validate([(0, 0)], [(geometry.width_px, geometry.height_px)], geometry)
    assert far.score == 0


@pytest.mark.parametrize("method", sorted(METHODS))
def test_personalization_reduces_error(method, settings, geometry, intrinsics, user) -> None:
    """Every method improves on the raw network on the simulated user with a kappa offset."""
    proc = FrameProcessor(settings, geometry, intrinsics, None)
    train, t = _collect(proc, user, grid_targets(geometry), frames=10)
    test, _ = _collect(proc, user, validation_targets(geometry), frames=5, t0=t)
    p = make_personalizer(method, geometry)
    p.fit(train)
    raw = np.mean([np.linalg.norm(s.obs.screen_px - s.target_px) for s in test])
    err = np.mean([np.linalg.norm(p.predict_px(s.obs) - s.target_px) for s in test])
    if method == "none":
        assert np.isclose(err, raw)
    else:
        assert err < 0.4 * raw, (method, err, raw)  # PRD 3.2: at least 40% better


@pytest.mark.parametrize("method", ["offset", "polynomial", "gp", "finetune"])
def test_personalizer_save_load(tmp_path, method, settings, geometry, intrinsics, user) -> None:
    proc = FrameProcessor(settings, geometry, intrinsics, None)
    train, t = _collect(proc, user, grid_targets(geometry), frames=4)
    p = make_personalizer(method, geometry)
    p.fit(train)
    path = tmp_path / "cal.json"
    p.save(path)
    q = load_personalizer(path, geometry)
    obs = train[3].obs
    assert np.allclose(p.predict_px(obs), q.predict_px(obs), atol=1.0)


def test_finetune_requires_features(geometry, settings, intrinsics, user) -> None:
    proc = FrameProcessor(settings, geometry, intrinsics, None)
    train, _ = _collect(proc, user, grid_targets(geometry)[:3], frames=2)
    train[0].obs.features = None
    with pytest.raises((ValueError, TypeError)):
        LastLayerFineTune(geometry).fit(train)


def test_drift_correction() -> None:
    d = DriftCorrector()
    offset = d.fit([np.array([110.0, 95.0]), np.array([112.0, 97.0])], (100.0, 100.0))
    assert np.allclose(offset, [-11, 4])
    assert np.allclose(d.apply(np.array([111.0, 96.0])), [100, 100])


def test_implicit_recalibration(settings, geometry, intrinsics, user) -> None:
    proc = FrameProcessor(settings, geometry, intrinsics, None)
    base, t = _collect(proc, user, grid_targets(geometry), frames=5)
    p = make_personalizer("polynomial", geometry)
    p.fit(base)
    rec = ImplicitRecalibrator(p, base, refit_every=2, background=False)
    rec.baseline_error_px = 30.0
    hist = GazeHistory()
    target = (700.0, 400.0)
    for _ in range(12):
        t += 1 / 30
        obs = proc.process_landmarks(user.look_at(*target), t).observation
        hist.add(t, obs)
    assert rec.on_click(t, hist, target) > 0
    # A click far from where the user looked is rejected.
    assert rec.on_click(t, hist, (50.0, 50.0)) == 0
    rec.on_click(t, hist, target)
    assert rec.refits == 1
