from __future__ import annotations

import numpy as np
import pytest

from gazeos.calibration.protocol import GazeObservation
from gazeos.control.dwell import DwellClicker
from gazeos.control.engine import Action, ControlEngine, EngineInput
from gazeos.control.filters import KalmanFilter2D, MovingAverage, NoFilter, OneEuroFilter, make_filter
from gazeos.control.fixation import EyeEvent, FixationDetector
from gazeos.control.mapping import magnetic_edges
from gazeos.control.snapping import TargetSnapper
from gazeos.control.targets import GazeTarget, TargetRegistry
from gazeos.control.zoom import make_zoom_region
from gazeos.gestures.blink import BlinkEvent
from gazeos.os_backends.base import RecordingBackend
from gazeos.settings import FilterSettings
from gazeos.types import ControlState, Rect, TrackingStatus, UIElement
from gazeos.vision.quality import QualityReport

OK = QualityReport(ok=True, status=TrackingStatus.TRACKING)


def _jitter(rng, n=120, sigma=15.0, center=(500.0, 400.0)):
    return [np.array(center) + rng.normal(0, sigma, 2) for _ in range(n)]


def _rms_jitter(points):
    p = np.stack(points)
    return float(np.sqrt(np.mean(np.sum(np.diff(p, axis=0) ** 2, axis=1))))


@pytest.mark.parametrize("kind", ["moving_average", "one_euro", "kalman", "uncertainty_kalman"])
def test_filters_reduce_jitter(kind, rng) -> None:
    f = make_filter(FilterSettings(kind=kind))
    raw = _jitter(rng)
    out = [f(i / 30, p, 15.0) for i, p in enumerate(raw)]
    assert _rms_jitter(out[10:]) < 0.6 * _rms_jitter(raw[10:])


def test_one_euro_passes_fast_motion() -> None:
    f = OneEuroFilter(1.0, 0.05, 1.0)
    out = None
    for i in range(40):
        out = f(i / 30, np.array([i * 40.0, 0.0]))
    assert out is not None and out[0] > 39 * 40.0 - 200  # little lag at speed


def test_no_filter_and_reset() -> None:
    assert np.allclose(NoFilter()(0, np.array([1.0, 2.0])), [1, 2])
    m = MovingAverage(3)
    m(0, np.array([0.0, 0.0]))
    m.reset()
    assert np.allclose(m(0, np.array([9.0, 9.0])), [9, 9])


def test_uncertainty_kalman_moves_less_when_unsure() -> None:
    sure, unsure = KalmanFilter2D(uncertainty_aware=True), KalmanFilter2D(uncertainty_aware=True)
    for f in (sure, unsure):
        for i in range(30):
            f(i / 30, np.array([100.0, 100.0]), 20.0)
    a = sure(1.0 + 1 / 30, np.array([200.0, 100.0]), 5.0)
    b = unsure(1.0 + 1 / 30, np.array([200.0, 100.0]), 200.0)
    assert a[0] > b[0]


def test_fixation_detector(rng) -> None:
    det = FixationDetector(px_per_deg=58.0)
    events = []
    for i, p in enumerate(_jitter(rng, 30, sigma=5.0)):
        events.append(det.update(i / 30, p, p).event)
    assert EyeEvent.FIXATION in events
    out = det.update(31 / 30, np.array([1500.0, 400.0]), np.array([1500.0, 400.0]))
    assert out.event == EyeEvent.SACCADE and out.saccade_started


def test_dwell_state_machine() -> None:
    d = DwellClicker(dwell_ms=800, radius_px=50, cooldown_ms=500)
    p = np.array([300.0, 300.0])
    fired = []
    for i in range(40):
        upd = d.update(i / 30, p)
        fired.append(upd.fired)
        if i == 10:
            assert upd.state == ControlState.DWELLING and 0 < upd.progress < 1
    assert sum(fired) == 1  # a long stare clicks once
    assert d.update(2.0, np.array([800.0, 300.0])).state == ControlState.IDLE  # leaving rearms
    # Looking away cancels the ring.
    d2 = DwellClicker(800, 50, 500)
    for i in range(10):
        d2.update(i / 30, p)
    assert d2.update(11 / 30, p + 200).progress == 0.0


def test_magnetic_edges() -> None:
    assert np.allclose(magnetic_edges(np.array([10.0, 500.0]), 1920, 1080, 30), [0, 500])
    assert np.allclose(magnetic_edges(np.array([5000.0, -40.0]), 1920, 1080, 30), [1919, 0])
    assert np.allclose(magnetic_edges(np.array([900.0, 500.0]), 1920, 1080, 30), [900, 500])


def test_snapping_prefers_near_and_large() -> None:
    s = TargetSnapper(threshold=0.7)
    near = UIElement(Rect(100, 100, 80, 40))
    far = UIElement(Rect(400, 100, 80, 40))
    d = s.decide(np.array([150.0, 125.0]), 40.0, [near, far])
    assert d.element == near and d.point == near.rect.center
    # Two equally close candidates: not confident, so no snap.
    a, b = UIElement(Rect(100, 100, 40, 40)), UIElement(Rect(150, 100, 40, 40))
    d2 = s.decide(np.array([145.0, 120.0]), 40.0, [a, b])
    assert d2.point is None


def test_zoom_region_roundtrip() -> None:
    z = make_zoom_region((1900.0, 20.0), 1920, 1080, 240, 3.0)
    assert 0 <= z.src_left <= 1920 - 240 and z.view_left + z.view_size <= 1920
    x, y = z.screen_to_view(1850.0, 60.0)
    assert np.allclose(z.view_to_screen(x, y), (1850, 60))


def test_registry_innermost_hit() -> None:
    r = TargetRegistry()
    calls = []
    r.register(GazeTarget("outer", Rect(0, 0, 500, 500), lambda: calls.append("outer")))
    r.register(GazeTarget("inner", Rect(100, 100, 50, 50), lambda: calls.append("inner")))
    hit = r.hit(120, 120)
    assert hit is not None and hit.key == "inner"
    r.clear_group("")
    assert r.hit(120, 120) is None


# -- engine -----------------------------------------------------------------------


def _obs(px, sigma=None) -> GazeObservation:
    return GazeObservation(np.zeros(2), np.array([0.0, 100.0, 600.0]), np.asarray(px, float), sigma=sigma)


def _engine(settings, geometry, backend=None, registry=None):
    backend = backend or RecordingBackend()
    return ControlEngine(settings, geometry, backend, registry=registry), backend


def _run(engine, point, seconds, t0=0.0, **kw):
    t = t0
    out = None
    for _ in range(int(seconds * 30)):
        t += 1 / 30
        out = engine.update(EngineInput(t, _obs(point), kw.get("quality", OK), **{k: v for k, v in kw.items() if k != "quality"}))
    return out, t


def test_engine_dwell_click(settings, geometry) -> None:
    settings.snap.enabled = False
    eng, be = _engine(settings, geometry)
    out, _ = _run(eng, (700, 500), 1.5)
    assert ("click", "left", 1) in be.events
    assert be.of_kind("move")


def test_engine_never_sends_input_while_paused(settings, geometry) -> None:
    eng, be = _engine(settings, geometry)
    eng.toggle_pause()
    _run(eng, (700, 500), 2.0, blink_event=None)
    assert not be.events
    eng.type_text("secret")
    assert not be.of_kind("type")
    assert not eng.run_shortcut("copy")


def test_engine_auto_pause_on_face_loss(settings, geometry) -> None:
    eng, be = _engine(settings, geometry)
    _run(eng, (700, 500), 0.3)
    lost = QualityReport(ok=False, status=TrackingStatus.NO_FACE, hold_cursor=True, auto_pause=True)
    out = eng.update(EngineInput(1.0, None, lost))
    assert out.paused and "auto_pause" in out.events
    back = eng.update(EngineInput(1.1, _obs((700, 500)), OK))
    assert not back.paused and "auto_resume" in back.events


def test_engine_emergency_stop_ends_drag(settings, geometry) -> None:
    eng, be = _engine(settings, geometry)
    eng.set_action(Action.DRAG)
    settings.snap.enabled = False
    _run(eng, (700, 500), 1.2)
    assert eng.dragging and be.of_kind("press")
    eng.emergency_stop()
    assert not eng.dragging and be.of_kind("release") and eng.paused


def test_engine_long_blink_clicks_at_frozen_point(settings, geometry) -> None:
    settings.dwell.enabled = False
    eng, be = _engine(settings, geometry)
    _, t = _run(eng, (600, 400), 0.5)
    # Eyes close: the cursor is frozen even if the estimate drifts.
    eng.update(EngineInput(t + 0.03, _obs((900, 700)), OK, eyes_closed=True))
    eng.update(EngineInput(t + 0.5, None, OK, blink_event=BlinkEvent.LONG_BLINK))
    clicks = be.of_kind("click")
    assert clicks == [("click", "left", 1)]
    last_move = be.of_kind("move")[-1]
    assert abs(last_move[1] - 600) < 40 and abs(last_move[2] - 400) < 40


def test_engine_pause_toggle_by_eyes_closed(settings, geometry) -> None:
    eng, _ = _engine(settings, geometry)
    out = eng.update(EngineInput(0.1, None, OK, blink_event=BlinkEvent.PAUSE_TOGGLE))
    assert out.paused
    out = eng.update(EngineInput(0.2, None, OK, blink_event=BlinkEvent.PAUSE_TOGGLE))
    assert not out.paused


def test_engine_invokes_own_targets_instead_of_clicking(settings, geometry) -> None:
    reg = TargetRegistry()
    hits = []
    reg.register(GazeTarget("key_a", Rect(650, 450, 100, 100), lambda: hits.append("a")))
    eng, be = _engine(settings, geometry, registry=reg)
    _run(eng, (700, 500), 1.5)
    assert hits == ["a"] and not be.of_kind("click")


def test_engine_zoom_click(settings, geometry) -> None:
    be = RecordingBackend(elements=[UIElement(Rect(700, 500, 10, 10)), UIElement(Rect(712, 500, 10, 10))])
    eng, _ = _engine(settings, geometry, backend=be)
    out, t = _run(eng, (710, 505), 1.2)
    assert eng.zoom is not None and not be.of_kind("click")
    # The target, 20 px away on screen, is 60 px away in the 3x magnified view.
    view = eng.zoom.screen_to_view(690.0, 505.0)
    _run(eng, view, 2.0, t0=t + 0.5)
    clicks = be.of_kind("click")
    assert clicks, be.events[-5:]
    i = be.events.index(clicks[0])
    move = [e for e in be.events[:i] if e[0] == "move"][-1]
    assert abs(move[1] - 690) <= 4 and abs(move[2] - 505) <= 4


def test_engine_scroll_zone(settings, geometry) -> None:
    eng, be = _engine(settings, geometry)
    eng.set_action(Action.SCROLL)
    _run(eng, (900, geometry.height_px - 10), 2.0)
    scrolls = be.of_kind("scroll")
    assert scrolls and all(s[2] > 0 for s in scrolls)
    assert not be.of_kind("click")


def test_engine_corner_menu_and_pause(settings, geometry) -> None:
    eng, _ = _engine(settings, geometry)
    events = []
    t = 0.0
    for _ in range(60):
        t += 1 / 30
        events += eng.update(EngineInput(t, _obs((geometry.width_px - 5, 5)), OK)).events
    assert "open_menu" in events
    for _ in range(60):
        t += 1 / 30
        events += eng.update(EngineInput(t, _obs((5, 5)), OK)).events
    assert eng.paused


def test_engine_destructive_shortcut_needs_confirmation(settings, geometry) -> None:
    be = RecordingBackend()
    eng = ControlEngine(settings, geometry, be, confirm=lambda name: False)
    assert not eng.run_shortcut("close_window")
    assert eng.run_shortcut("copy")
    assert be.of_kind("hotkey")[-1][1:] in (("ctrl", "c"), ("cmd", "c"))


def test_engine_rest_reminder(settings, geometry) -> None:
    settings.system.rest_reminder_min = 0.01  # 0.6 s
    eng, _ = _engine(settings, geometry)
    events = []
    t = 0.0
    for _ in range(30):
        t += 1 / 30
        events += eng.update(EngineInput(t, None, OK)).events
    assert "rest_reminder" in events
