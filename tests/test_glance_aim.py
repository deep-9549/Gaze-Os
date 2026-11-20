"""Glance aim after each word, and learning from corrected near misses (20 Nov 2026)."""

from __future__ import annotations

import numpy as np

from gazeos.calibration.glance_aim import GlanceAim
from gazeos.calibration.implicit import KeyPress, MistakeLearner


def _glance(points, t0=0.0, **kw):
    g = GlanceAim((900.0, 600.0), t0, **kw)
    for i, p in enumerate(points):
        g.add(t0 + i / 30, np.asarray(p, float))
    return g


def test_glance_moves_the_cursor_onto_the_dot(rng) -> None:
    # The eye travels for the first 0.35 s (ignored), then rests 60 px left of and 40 px above the dot.
    pts = [(300.0, 1000.0)] * 10 + [(840 + rng.normal(0, 20), 560 + rng.normal(0, 20)) for _ in range(20)]
    shift = _glance(pts).result()
    assert shift is not None and abs(shift[0] - 60) < 15 and abs(shift[1] - 40) < 15


def test_glance_that_did_not_happen_changes_nothing(rng) -> None:
    scattered = [(rng.uniform(0, 1900), rng.uniform(0, 1000)) for _ in range(30)]
    g = _glance(scattered)
    assert g.result() is None and "not on the dot" in g.reason
    far = _glance([(500.0, 600.0)] * 30)  # 400 px away: looking at another key, not drift
    assert far.result() is None and "too large" in far.reason
    few = _glance([(880.0, 600.0)] * 14)  # tracking dropped: only 4 frames after 0.35 s
    assert few.result() is None and "too few" in few.reason


def _press(t, name, x, y, cursor=None):
    c = np.array([x, y], float)
    return KeyPress(t, name, c, 176.0, c.copy(), c.copy() if cursor is None else np.asarray(cursor, float))


def test_near_miss_corrected_with_del_is_learned() -> None:
    m = MistakeLearner()
    assert m.on_press(_press(0.0, "key_r", 600, 400, cursor=(610, 420))) is None
    assert m.on_press(_press(1.5, "key_backspace", 800, 900)) is None
    out = m.on_press(_press(3.0, "key_t", 776, 400))
    assert out is not None
    where, corr = out
    assert np.allclose(corr, [166, -20]) and np.allclose(where, [600, 400])


def test_editing_is_not_learned() -> None:
    m = MistakeLearner()
    for seq in (
        [("key_r", 600), ("key_backspace", 800), ("key_backspace", 800), ("key_t", 776)],  # two Dels
        [("key_r", 600), ("key_backspace", 800), ("key_p", 1500)],  # far away
        [("key_r", 600), ("key_backspace", 800), ("key_r", 600)],  # the same key again
    ):
        m = MistakeLearner()
        outs = [m.on_press(_press(i * 1.0, n, x, 400)) for i, (n, x) in enumerate(seq)]
        assert all(o is None for o in outs), seq
    slow = MistakeLearner()
    slow.on_press(_press(0.0, "key_r", 600, 400))
    slow.on_press(_press(10.0, "key_backspace", 800, 900))  # deleted much later: not a near miss
    assert slow.on_press(_press(11.0, "key_t", 776, 400)) is None


def test_engine_learns_a_corrected_near_miss(settings, geometry) -> None:
    from gazeos.calibration.protocol import GazeObservation
    from gazeos.control.engine import ControlEngine, EngineInput
    from gazeos.control.targets import GazeTarget, TargetRegistry
    from gazeos.os_backends.base import RecordingBackend
    from gazeos.types import Rect, TrackingStatus
    from gazeos.vision.quality import QualityReport

    ok = QualityReport(ok=True, status=TrackingStatus.TRACKING)
    reg = TargetRegistry()
    for name, x, y in (("key_r", 500, 400), ("key_t", 676, 400), ("key_backspace", 900, 800)):
        reg.register(GazeTarget(f"keyboard:{name}:1", Rect(x, y, 176, 176), lambda: None, "keyboard"))
    settings.snap.enabled = False
    eng = ControlEngine(settings, geometry, RecordingBackend(), registry=reg)
    t = 0.0

    def look(px, seconds):
        nonlocal t
        for _ in range(int(seconds * 30)):
            t += 1 / 30
            eng.update(EngineInput(t, GazeObservation(np.zeros(2), np.array([0.0, 100, 600]), np.asarray(px, float)), ok))

    look((640, 480), 1.6)  # the user looks at t (left part); the cursor sits on r: r is typed
    look((990, 890), 1.6)  # Del
    look((760, 480), 1.6)  # t
    assert eng.drift.field is not None and len(eng.drift.field) == 1
    assert eng.drift.field.at(np.array([640.0, 480.0]))[0] > 50  # the cursor now goes right there


def test_posture_compensation_follows_the_head_after_an_aim(settings, geometry) -> None:
    """After an aim the cursor is corrected for how far the head has moved since (20 Nov: the
    error followed the head, +3.7 px per app mm sideways and -7.6 vertically)."""
    from gazeos.calibration.protocol import GazeObservation
    from gazeos.control.engine import ControlEngine, EngineInput
    from gazeos.os_backends.base import RecordingBackend
    from gazeos.types import TrackingStatus
    from gazeos.vision.quality import QualityReport

    ok = QualityReport(ok=True, status=TrackingStatus.TRACKING)
    settings.snap.enabled = False
    settings.dwell.enabled = False
    settings.fixation.hold_deadband_deg = 0.0  # the cursor's hold-still band would hide part of the move
    eng = ControlEngine(settings, geometry, RecordingBackend())
    t = 0.0

    def run(origin, seconds=2.0):
        nonlocal t
        out = None
        for _ in range(int(seconds * 30)):
            t += 1 / 30
            obs = GazeObservation(np.zeros(2), np.asarray(origin, float), np.array([900.0, 500.0]))
            out = eng.update(EngineInput(t, obs, ok))
        return out

    run((0.0, 0.0, 600.0))
    out = run((10.0, 10.0, 600.0))  # no aim yet: nothing to measure from
    assert abs(out.cursor[0] - 900) < 5 and abs(out.cursor[1] - 500) < 5
    eng.set_posture_reference(np.array([[0.0, 0.0, 600.0]] * 5))
    out = run((10.0, 10.0, 600.0), 3.0)  # head 10 mm to the camera's x and down
    sx, sy = geometry.px_per_mm
    assert abs(out.cursor[0] - (900 - 0.66 * 10 * sx)) < 8
    assert abs(out.cursor[1] - (500 + 1.36 * 10 * sy)) < 8
    settings.calibration.posture_compensation = False
    out = run((10.0, 10.0, 600.0), 3.0)
    assert abs(out.cursor[0] - 900) < 5
