"""Head nudge and the correction learned from it (20 Nov 2026)."""

from __future__ import annotations

import numpy as np

from gazeos.calibration.implicit import CorrectionField, DriftCorrector
from gazeos.calibration.protocol import GazeObservation
from gazeos.control.engine import ControlEngine, EngineInput
from gazeos.control.head_nudge import HeadNudge
from gazeos.control.targets import GazeTarget, TargetRegistry
from gazeos.os_backends.base import RecordingBackend
from gazeos.settings import ControlMode
from gazeos.types import Rect, TrackingStatus
from gazeos.vision.quality import QualityReport

OK = QualityReport(ok=True, status=TrackingStatus.TRACKING)
EYE = 80.0  # eye distance in image pixels
NOSE = (320.0, 260.0)


def _feed_nudge(n: HeadNudge, t0: float, seconds: float, nose=NOSE) -> float:
    t = t0
    for _ in range(int(round(seconds * 30))):
        t += 1 / 30
        n.update(t, nose, EYE)
    return t


def test_nudge_follows_the_head_after_settling() -> None:
    n = HeadNudge(1920, max_px=500, gain=1.5, deadband=0.01, settle_s=0.4)
    t = _feed_nudge(n, 0.0, 1.0)
    assert np.allclose(n.offset, 0.0)
    # The nose moves 4 px down and 4 px to the image's left (the user's right): 0.07 eye distances.
    t = _feed_nudge(n, t, 2.0, nose=(NOSE[0] - 4.0, NOSE[1] + 4.0))
    assert n.offset[0] > 100 and n.offset[1] > 100  # cursor right and down
    assert abs(n.offset[0] - n.offset[1]) < 1.0


def test_sway_inside_the_dead_band_does_not_move_the_cursor() -> None:
    n = HeadNudge(1920, max_px=500, gain=1.5, deadband=0.01, settle_s=0.4)
    t = _feed_nudge(n, 0.0, 1.0)
    rng = np.random.default_rng(0)
    for _ in range(90):
        t += 1 / 30
        n.update(t, (NOSE[0] + rng.normal(0, 0.3), NOSE[1] + rng.normal(0, 0.3)), EYE)
        assert np.linalg.norm(n.offset) < 10.0


def test_head_turn_while_settling_is_not_a_nudge() -> None:
    """The head keeps turning toward a new target just after the eyes land: not a correction."""
    n = HeadNudge(1920, max_px=500, gain=1.5, deadband=0.01, settle_s=0.4)
    t = 0.0
    for i in range(10):  # 0.33 s of head turning, inside the settle time
        t += 1 / 30
        n.update(t, (NOSE[0] - i, NOSE[1]), EYE)
    t = _feed_nudge(n, t, 1.0, nose=(NOSE[0] - 9, NOSE[1]))
    assert np.linalg.norm(n.offset) < 10.0


def test_nudge_is_capped_and_reset() -> None:
    n = HeadNudge(1920, max_px=200, gain=1.5, deadband=0.01, settle_s=0.4)
    t = _feed_nudge(n, 0.0, 1.0)
    _feed_nudge(n, t, 2.0, nose=(NOSE[0], NOSE[1] + 40))
    assert np.linalg.norm(n.offset) <= 200.0 + 1e-6
    n.reset()
    assert np.allclose(n.offset, 0.0)


def test_correction_field_is_local_and_fades() -> None:
    f = CorrectionField(1920, radius_frac=0.25, half_life_s=300.0, prior=0.5)
    f.add(0.0, np.array([300.0, 900.0]), np.array([0.0, 90.0]))
    near = f.at(np.array([320.0, 880.0]))
    far = f.at(np.array([1700.0, 100.0]))
    assert 50 < near[1] < 90  # one correction moves it part of the way
    assert abs(far[1]) < 5
    f.add(600.0, np.array([1700.0, 100.0]), np.array([0.0, 0.0]))  # ten minutes later
    assert f.at(np.array([320.0, 880.0]))[1] < 0.5 * near[1]
    d = DriftCorrector(f)
    d.reset()
    assert len(f) == 0


def _engine(settings, geometry, registry=None):
    settings.snap.enabled = False
    settings.mode = ControlMode.HEAD_FINE_TUNE
    backend = RecordingBackend()
    return ControlEngine(settings, geometry, backend, registry=registry), backend


def _feed(eng, t0, seconds, gaze, nose=NOSE):
    t, out = t0, None
    for _ in range(int(round(seconds * 30))):
        t += 1 / 30
        obs = GazeObservation(np.zeros(2), np.array([0.0, 100.0, 600.0]), np.asarray(gaze, float))
        out = eng.update(EngineInput(t, obs, OK, head_point=(nose[0], nose[1], EYE)))
    return out, t


def test_engine_nudges_the_cursor_onto_the_key_and_learns(settings, geometry) -> None:
    """The gaze sits a key above the key looked at; a small nod moves the cursor down onto it,
    the key is pressed there, and the next look at that area lands lower by itself."""
    reg = TargetRegistry()
    pressed: list[str] = []
    reg.register(GazeTarget("keyboard:up", Rect(600, 600, 160, 160), lambda: pressed.append("up")))
    reg.register(GazeTarget("keyboard:down", Rect(600, 760, 160, 160), lambda: pressed.append("down")))
    settings.dwell.dwell_ms = 800
    eng, _ = _engine(settings, geometry, reg)
    gaze = (680.0, 680.0)  # on "up"; the user looks at "down"
    out, t = _feed(eng, 0.0, 0.5, gaze)
    assert out.cursor is not None and out.cursor[1] < 760
    # Nod: the nose moves down 7 px (0.09 eye distances) over 0.3 s, then holds.
    for i in range(1, 10):
        _, t = _feed(eng, t, 1 / 30, gaze, nose=(NOSE[0], NOSE[1] + 7 * i / 9))
    out, t = _feed(eng, t, 1.5, gaze, nose=(NOSE[0], NOSE[1] + 7))
    assert pressed == ["down"]
    assert eng.drift.field is not None and len(eng.drift.field) == 1
    # Look away (a saccade resets the nudge), come back: the cursor now lands lower unaided.
    _, t = _feed(eng, t, 0.6, (1600.0, 200.0))
    out, t = _feed(eng, t, 0.3, gaze)
    assert out.cursor is not None and out.cursor[1] > 680 + 60


def test_moving_the_head_restarts_the_dwell(settings, geometry) -> None:
    """A dwell 0.4 s along when the user starts to nudge does not finish while the head moves,
    and needs a full dwell after it stops."""
    reg = TargetRegistry()
    pressed: list[str] = []
    reg.register(GazeTarget("keyboard:big", Rect(200, 200, 1000, 700), lambda: pressed.append("big")))
    settings.dwell.dwell_ms = 800
    eng, _ = _engine(settings, geometry, reg)
    out, t = _feed(eng, 0.0, 0.4, (700.0, 500.0))
    # Nod slowly for 1.5 s (well past the dwell time) within the big key.
    for i in range(45):
        _, t = _feed(eng, t, 1 / 30, (700.0, 500.0), nose=(NOSE[0], NOSE[1] + 0.2 * i))
    assert pressed == []
    _feed(eng, t, 0.6, (700.0, 500.0), nose=(NOSE[0], NOSE[1] + 0.2 * 44))
    assert pressed == []  # under a full dwell since the head stopped
    _feed(eng, t + 0.6, 1.0, (700.0, 500.0), nose=(NOSE[0], NOSE[1] + 0.2 * 44))
    assert pressed == ["big"]


def test_head_turning_with_a_gliding_gaze_is_not_a_nudge(settings, geometry) -> None:
    """20 Nov run: the eyes glided 300 px (no saccade) and the head turned with them; that turn
    was read as a nudge and carried the cursor 190 px further, onto Settings."""
    eng, _ = _engine(settings, geometry)
    out, t = _feed(eng, 0.0, 1.0, (900.0, 500.0))
    for i in range(1, 31):  # one second: gaze glides 300 px left, the nose turns with it
        out, t = _feed(eng, t, 1 / 30, (900.0 - 10 * i, 500.0), nose=(NOSE[0] + 0.2 * i, NOSE[1]))
    out, t = _feed(eng, t, 0.5, (600.0, 500.0), nose=(NOSE[0] + 6, NOSE[1]))
    assert out.cursor is not None and abs(out.cursor[0] - 600) < 40


def test_eyes_only_mode_ignores_the_head(settings, geometry) -> None:
    eng, _ = _engine(settings, geometry)
    settings.mode = ControlMode.EYES_ONLY
    eng.apply_settings(settings)
    out, t = _feed(eng, 0.0, 1.0, (700.0, 500.0))
    out, t = _feed(eng, t, 1.0, (700.0, 500.0), nose=(NOSE[0], NOSE[1] + 10))
    assert out.cursor is not None and abs(out.cursor[1] - 500) < 5
