from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gazeos.calibration.personalization import make_personalizer
from gazeos.calibration.protocol import CalibrationSample, grid_targets
from gazeos.control.engine import ControlEngine
from gazeos.os_backends.base import RecordingBackend
from gazeos.pipeline import FrameProcessor
from gazeos.profiles.logs import ExperimentLogger, HeatmapRecorder, read_log
from gazeos.profiles.store import ProfileStore
from gazeos.recording import RecordedSession, SessionMeta, SessionRecorder, condition_code
from gazeos.settings import Settings
from research.metrics import rms_precision

BASELINE = Path(__file__).parent / "fixtures" / "replay_baseline.json"


def test_profile_lifecycle(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    p = store.create("Ramesh")
    s = p.load_settings()
    s.dwell.dwell_ms = 1200
    p.save_settings(s)
    assert store.get("Ramesh").load_settings().dwell.dwell_ms == 1200
    p.save_calibration_features(target_px=np.zeros((3, 2)))
    assert "target_px" in p.load_calibration_features()
    store.last_used = "Ramesh"
    assert store.last_used == "Ramesh" and store.list() == ["Ramesh"]
    labels = [label for label, _, _ in p.stored_items()]
    assert "Settings" in labels and any("no images" in label for label in labels)
    p.delete_all()
    assert not p.path.exists()
    with pytest.raises(ValueError):
        store.get("../evil")
    with pytest.raises(FileExistsError):
        store.create("x")
        store.create("x")


def test_settings_validation() -> None:
    s = Settings()
    with pytest.raises(ValueError):
        s.dwell.dwell_ms = 100  # below 300 ms
    assert Settings.model_validate_json(s.model_dump_json()) == s


def test_experiment_log_redacts_passwords(tmp_path) -> None:
    log = ExperimentLogger(tmp_path, "s1")
    log.log_text("hunter2", in_password_field=True)
    log.log_text("hello", in_password_field=False)
    log.log_frame(0.1, (10, 20), "idle", "Tracking", {"total": 5.0}, 30.0)
    log.close()
    recs = read_log(log.path)
    assert recs[0]["text"] is None and recs[0]["redacted"] and recs[1]["text"] == "hello"
    assert "hunter2" not in log.path.read_text()


def test_heatmap(tmp_path) -> None:
    h = HeatmapRecorder(320, 240, 16)
    for _ in range(50):
        h.add(100, 100)
    assert h.normalized().max() == pytest.approx(1.0)
    h.save(tmp_path / "hm")
    assert (tmp_path / "hm.npy").exists()


def _record_session(user, settings, geometry, intrinsics, path, rng) -> Path:
    rec = SessionRecorder(path, SessionMeta(screen={"width_px": geometry.width_px, "height_px": geometry.height_px,
                                                    "width_mm": geometry.width_mm, "height_mm": geometry.height_mm},
                                            camera={"width": 640, "height": 480}, participant="P01",
                                            condition={"lighting": "dim", "glasses": "yes", "distance_cm": "65"}))
    t = 0.0
    targets = grid_targets(geometry) + [(float(rng.uniform(200, 1700)), float(rng.uniform(150, 900))) for _ in range(6)]
    for x, y in targets:
        for _ in range(30):
            t += 1 / 30
            rec.add(t, user.look_at(x, y), (x, y))
    return rec.save()


def replay_metrics(path: Path, settings, intrinsics) -> dict[str, float]:
    """Replay a recorded session through the pipeline and engine (PRD 23 'Replay integration')."""
    from gazeos.calibration.geometry import ScreenGeometry

    sess = RecordedSession(path)
    geometry = ScreenGeometry(**{k: sess.meta["screen"][k] for k in ("width_px", "height_px", "width_mm", "height_mm")})
    proc = FrameProcessor(settings, geometry, intrinsics, None)
    frames = list(sess.frames())
    calib_targets = set(grid_targets(geometry))
    samples, test = [], []
    for fr in frames:
        pf = proc.process_landmarks(fr.landmarks, fr.t)
        if pf.observation is None or fr.target_px is None:
            continue
        item = (fr, pf)
        (samples if fr.target_px in calib_targets else test).append(item)
    p = make_personalizer("polynomial", geometry)
    p.fit([CalibrationSample(pf.observation, np.asarray(fr.target_px)) for fr, pf in samples])
    s = settings.model_copy(deep=True)
    s.snap.enabled = False
    eng = ControlEngine(s, geometry, RecordingBackend(), p)
    errs, jitters, cur = [], [], []
    last_target = None
    for fr, pf in test:
        if fr.target_px != last_target and cur:
            jitters.append(rms_precision(cur[len(cur) // 2:]))
            cur = []
        last_target = fr.target_px
        out = eng.update(pf.engine_input)
        if out.cursor is not None:
            cur.append(out.cursor)
            errs.append(float(np.linalg.norm(np.asarray(out.cursor) - fr.target_px)))
    jitters.append(rms_precision(cur[len(cur) // 2:]))
    return {"mean_error_px": float(np.mean(errs)), "jitter_px": float(np.mean(jitters))}


def test_recording_roundtrip_and_replay_regression(tmp_path, settings, geometry, intrinsics) -> None:
    from gazeos.sim import SimulatedUser, SyntheticFace

    rng = np.random.default_rng(0)
    user = SimulatedUser(geometry, SyntheticFace(intrinsics, noise_px=0.2, seed=1))
    path = _record_session(user, settings, geometry, intrinsics, tmp_path / "s.npz", rng)
    sess = RecordedSession(path)
    assert len(sess) == 15 * 30 and sess.meta["condition_code"] == condition_code(
        {"lighting": "dim", "glasses": "yes", "distance_cm": "65"}) == 0b111
    m = replay_metrics(path, settings, intrinsics)
    baseline = json.loads(BASELINE.read_text())
    # Fails CI if recorded accuracy or jitter gets more than 5% worse (PRD 23).
    assert m["mean_error_px"] <= baseline["mean_error_px"] * 1.05, m
    assert m["jitter_px"] <= baseline["jitter_px"] * 1.05, m
