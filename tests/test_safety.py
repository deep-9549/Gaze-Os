"""No action the user did not choose: lost tracking, a stopped camera, calibration and exit
(fable_opinion.md section 5)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from gazeos.calibration.protocol import GazeObservation
from gazeos.capture.camera import CaptureThread
from gazeos.control.engine import Action, ControlEngine, EngineInput
from gazeos.gestures.blink import BlinkEvent
from gazeos.inference.gaze import GazeEstimate
from gazeos.os_backends.base import RecordingBackend
from gazeos.pipeline import FrameProcessor, ProcessingThread
from gazeos.types import Rect, TrackingStatus, UIElement
from gazeos.vision.quality import QualityReport

OK = QualityReport(ok=True, status=TrackingStatus.TRACKING)
HELD = QualityReport(ok=False, status=TrackingStatus.TRACKING, hold_cursor=True)
POINT = (700.0, 500.0)


def _obs(px=POINT) -> GazeObservation:
    return GazeObservation(np.zeros(2), np.array([0.0, 100.0, 600.0]), np.asarray(px, float))


def _engine(settings, geometry, backend=None):
    settings.snap.enabled = False
    backend = backend or RecordingBackend()
    return ControlEngine(settings, geometry, backend), backend


def _feed(engine, t0, seconds, valid=True, quality=None, **kw):
    """30 frames a second from t0; returns the time of the last frame."""
    t = t0
    for _ in range(int(round(seconds * 30))):
        t += 1 / 30
        engine.update(EngineInput(t, _obs() if valid else None, quality or (OK if valid else HELD), **kw))
    return t


def test_dwell_does_not_finish_on_the_first_frame_after_lost_tracking(settings, geometry) -> None:
    eng, be = _engine(settings, geometry)
    t = _feed(eng, 0.0, 0.3)
    t = _feed(eng, t, 1.0, valid=False)  # longer than the 800 ms dwell
    t = _feed(eng, t, 0.2)
    assert not be.of_kind("click")
    _feed(eng, t, 0.9)  # a full dwell of valid samples does click
    assert be.of_kind("click") == [("click", "left", 1)]


def test_a_blink_inside_a_dwell_is_not_counted_toward_it(settings, geometry) -> None:
    eng, be = _engine(settings, geometry)
    t = _feed(eng, 0.0, 0.4)
    t = _feed(eng, t, 0.3, valid=False)  # an ordinary blink
    t = _feed(eng, t, 0.2)  # 0.9 s on the clock, 0.6 s of gaze
    assert not be.of_kind("click")
    _feed(eng, t, 0.3)
    assert be.of_kind("click") == [("click", "left", 1)]


@pytest.mark.parametrize("status", [TrackingStatus.MULTIPLE_FACES, TrackingStatus.NO_FACE, TrackingStatus.MOVE_CLOSER])
def test_blink_click_needs_one_tracked_face(settings, geometry, status) -> None:
    settings.dwell.enabled = False
    eng, be = _engine(settings, geometry)
    t = _feed(eng, 0.0, 0.5)
    eng.update(EngineInput(t + 0.5, None, QualityReport(ok=False, status=status), blink_event=BlinkEvent.LONG_BLINK))
    assert not be.of_kind("click")


def test_blink_click_needs_a_recent_gaze_sample(settings, geometry) -> None:
    settings.dwell.enabled = False
    eng, be = _engine(settings, geometry)
    t = _feed(eng, 0.0, 0.5)
    eng.update(EngineInput(t + 4.0, None, HELD, blink_event=BlinkEvent.LONG_BLINK))  # the cursor is 4 s old
    assert not be.of_kind("click")
    t = _feed(eng, t + 4.0, 0.2)
    eng.update(EngineInput(t + 0.5, None, HELD, blink_event=BlinkEvent.LONG_BLINK))
    assert be.of_kind("click") == [("click", "left", 1)]


def test_small_target_chosen_with_confidence_opens_the_magnifier(settings, geometry) -> None:
    be = RecordingBackend(elements=[UIElement(Rect(752, 492, 16, 16))])  # one 16 px button, 60 px from the gaze
    eng = ControlEngine(settings, geometry, be)
    _feed(eng, 0.0, 1.2)
    assert eng.zoom is not None and not be.of_kind("click")


def test_release_inputs_lets_go_of_a_drag_without_pausing(settings, geometry) -> None:
    eng, be = _engine(settings, geometry)
    eng.set_action(Action.DRAG)
    _feed(eng, 0.0, 1.2)
    assert eng.dragging
    eng.release_inputs()
    assert not eng.dragging and be.of_kind("release") and not eng.paused


def test_saved_settings_reach_the_running_dwell(settings, geometry) -> None:
    eng, be = _engine(settings, geometry)
    changed = settings.model_copy(deep=True)
    changed.dwell.dwell_ms = 1600.0
    eng.apply_settings(changed)
    t = _feed(eng, 0.0, 1.2)
    assert not be.of_kind("click")  # the old 800 ms would have clicked
    _feed(eng, t, 0.7)
    assert be.of_kind("click")


class _DeadCamera:
    def latest(self, timeout: float = 0.5):
        time.sleep(0.02)
        return None


def test_a_stopped_camera_releases_the_drag_and_pauses(settings, geometry, intrinsics) -> None:
    eng, be = _engine(settings, geometry)
    eng.set_action(Action.DRAG)
    _feed(eng, 0.0, 1.2)
    assert eng.dragging
    worker = ProcessingThread(_DeadCamera(), FrameProcessor(settings, geometry, intrinsics, None), eng)
    worker.STALL_S = 0.1
    worker.start()
    deadline = time.monotonic() + 3
    while not worker.stalled and time.monotonic() < deadline:
        time.sleep(0.02)
    worker.stop()
    worker.join(2.0)
    assert worker.stalled and eng.paused and eng.auto_paused and not eng.dragging and be.of_kind("release")


class _BlankSource:
    def read(self):
        time.sleep(0.005)
        return np.zeros((4, 4, 3), np.uint8)

    def release(self) -> None:
        pass


def test_capture_thread_can_be_joined() -> None:
    cap = CaptureThread(_BlankSource())
    cap.start()
    assert cap.latest(1.0) is not None
    cap.stop()
    cap.join(2.0)  # raised TypeError while the stop flag was named like Thread._stop
    assert not cap.is_alive()


class _CountingModel:
    def __init__(self) -> None:
        self.seen: list[object] = []

    def estimate(self, norm):
        self.seen.append(norm)
        return GazeEstimate(np.zeros(2), np.zeros(2), np.radians(np.array([3.0, 3.0])), np.zeros(256))


def test_saved_crops_replay_through_the_gaze_network(settings, geometry, intrinsics, user) -> None:
    """E6 on recordings: the crops stored with a research recording reach the network."""
    model = _CountingModel()
    proc = FrameProcessor(settings, geometry, intrinsics, None, model)
    face = np.full((224, 224, 3), 7, np.uint8)
    eye = np.zeros((36, 60), np.uint8)
    pf = proc.process_landmarks(user.look_at(960, 540), 0.1, crops=(face, eye, eye))
    assert len(model.seen) == 1 and model.seen[0].face is face
    assert np.allclose(model.seen[0].R_n @ model.seen[0].R_n.T, np.eye(3), atol=1e-9)
    assert pf.observation is not None and pf.observation.features is not None
    # Without crops or an image the geometric baseline answers instead.
    proc.process_landmarks(user.look_at(960, 540), 0.2)
    assert len(model.seen) == 1


def test_calibration_runs_and_accepts_without_any_button(qtbot, settings, geometry, intrinsics, user) -> None:
    pytest.importorskip("PySide6")
    from gazeos.calibration.personalization import make_personalizer
    from gazeos.settings import CalibrationSettings
    from gazeos.ui.calibration import CalibrationWindow

    outcomes: list[object] = []
    cal = CalibrationSettings(target_duration_s=0.3, keep_from_s=0.05, min_samples_per_point=3,
                              auto_start_s=0.2, auto_decide_s=1.0)
    win = CalibrationWindow(geometry, make_personalizer("polynomial", geometry), cal, outcomes.append)
    qtbot.addWidget(win)
    win.resize(800, 600)
    win.show()
    proc = FrameProcessor(settings, geometry, intrinsics, None)
    deadline = time.monotonic() + 20
    while not outcomes and time.monotonic() < deadline:
        pts = win._points()
        if win.phase in ("calibrate", "validate") and 0 <= win.index < len(pts):
            pf = proc.process_landmarks(user.look_at(*pts[win.index]), time.monotonic())
            win.add_frame(time.monotonic(), pf.observation, False)
        qtbot.wait(10)
    assert outcomes and outcomes[0] is not None and outcomes[0].validation.score >= 70


def test_calibration_without_eye_data_retries_then_closes_by_itself(qtbot, geometry) -> None:
    pytest.importorskip("PySide6")
    from gazeos.calibration.personalization import make_personalizer
    from gazeos.settings import CalibrationSettings
    from gazeos.ui.calibration import CalibrationWindow

    outcomes: list[object] = []
    cal = CalibrationSettings(target_duration_s=0.05, keep_from_s=0.01, auto_start_s=0.1, auto_decide_s=1.0,
                              auto_redo_limit=1)
    win = CalibrationWindow(geometry, make_personalizer("polynomial", geometry), cal, outcomes.append)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitUntil(lambda: bool(outcomes), timeout=30000)
    assert outcomes == [None] and win._auto_redos == 1


def test_opening_the_keyboard_never_blocks_the_screen(qtbot, tmp_path) -> None:
    """The word predictor (about 3.5 s with the trained 3-gram) loads in the background from
    startup; a Keyboard press before it is ready returns at once and opens the keyboard when
    it is (18 Nov 2026: the screen froze on every first open)."""
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from gazeos.ui.app import GazeOSApp

    app = GazeOSApp(SimpleNamespace(home=str(tmp_path), profile="t", simulate=True, dry_run=True,
                                    gaze_model=None, camera=None))
    try:
        from concurrent.futures import ThreadPoolExecutor

        slow = ThreadPoolExecutor(max_workers=1)  # still loading when Keyboard is pressed
        app._predictor = slow.submit(lambda: (time.sleep(1.0), app._build_predictor())[1])
        slow.shutdown(wait=False)
        t0 = time.perf_counter()
        app.toggle_keyboard()
        app.toggle_keyboard()  # a second press while loading changes nothing
        assert time.perf_counter() - t0 < 0.5
        qtbot.waitUntil(lambda: app.keyboard is not None and app.keyboard.isVisible(), timeout=30000)
    finally:
        app.worker.stop()
        app.worker.join(2.0)
        if app.keyboard is not None:
            app.keyboard.close()
        for w in (app.overlay, app.action_bar, app.menu, app.shortcuts, app.debug):
            w.close()


LID = QualityReport(ok=True, status=TrackingStatus.TRACKING)  # face tracked, sample dropped by the lid gate


def test_frames_dropped_for_lowered_lids_still_count_toward_the_dwell(settings, geometry) -> None:
    """Looking at the lower keyboard rows lowers the lids; the lid gate drops many of those
    frames while the user keeps looking at the key. That time counts (18 Nov 2026 traces:
    13-78% of the frames dropped on the lower rows froze the dwell)."""
    eng, be = _engine(settings, geometry)
    t = _feed(eng, 0.0, 0.3)
    for _ in range(4):  # 0.4 s with every other frame dropped by the lid gate
        t += 1 / 30
        eng.update(EngineInput(t, None, LID))
        t = _feed(eng, t, 1 / 30)
        t += 1 / 30
        eng.update(EngineInput(t, None, LID))
    _feed(eng, t, 0.2)  # 0.9 s on the clock, a third of it without a sample
    assert be.of_kind("click") == [("click", "left", 1)]


def test_a_long_lid_gap_starts_the_dwell_again(settings, geometry) -> None:
    eng, be = _engine(settings, geometry)
    t = _feed(eng, 0.0, 0.2)
    for _ in range(60):  # 2 s of dropped frames: the gaze could be anywhere by now
        t += 1 / 30
        eng.update(EngineInput(t, None, LID))
    t = _feed(eng, t, 0.3)
    assert not be.of_kind("click")  # no click on the frames right after the gap
    _feed(eng, t, 0.6)  # a fresh full dwell does click
    assert be.of_kind("click") == [("click", "left", 1)]


def test_while_paused_only_the_pause_button_responds(settings, geometry) -> None:
    """A user without hands resumes by looking at Pause; nothing else is pressed or clicked."""
    from gazeos.control.targets import GazeTarget, TargetRegistry
    from gazeos.types import Rect

    pressed: list[str] = []
    reg = TargetRegistry()
    reg.register(GazeTarget("keyboard:key_a:1", Rect(650, 450, 100, 100), lambda: pressed.append("a"), "keyboard"))
    reg.register(GazeTarget("action_bar:pause:2", Rect(1756, 787, 155, 90), lambda: pressed.append("pause"), "action_bar"))
    eng, be = _engine(settings, geometry)
    eng.registry = reg
    eng.set_paused(True)
    t = _feed(eng, 0.0, 2.0)  # 2 s on key A (POINT is inside it)
    assert pressed == [] and not be.of_kind("click")
    for _ in range(36):  # 1.2 s on Pause
        t += 1 / 30
        eng.update(EngineInput(t, _obs((1830.0, 830.0)), OK))
    assert pressed == ["pause"] and not be.of_kind("click") and not be.of_kind("move")


def test_keep_looking_at_a_typing_key_to_type_it_again(settings, geometry) -> None:
    """The second l of "hello" without looking away; other GazeOS buttons press once."""
    from gazeos.control.targets import GazeTarget, TargetRegistry
    from gazeos.types import Rect

    pressed: list[str] = []
    reg = TargetRegistry()
    reg.register(GazeTarget("keyboard:key_l:1", Rect(650, 450, 100, 100), lambda: pressed.append("l"),
                            "keyboard", repeat=True))
    reg.register(GazeTarget("action_bar:pause:2", Rect(1756, 787, 155, 90), lambda: pressed.append("pause"),
                            "action_bar"))
    eng, _ = _engine(settings, geometry)
    eng.registry = reg
    t = _feed(eng, 0.0, 1.2)
    assert pressed == ["l"]  # once at 0.8 s; not again yet
    t = _feed(eng, t, 0.9)
    assert pressed == ["l"]  # still once at 2.1 s
    t = _feed(eng, t, 0.3)
    assert pressed == ["l", "l"]  # again at 0.8 + 0.7 + 0.8 = 2.3 s
    for _ in range(90):  # 3 s on Pause: pressed once only
        t += 1 / 30
        eng.update(EngineInput(t, _obs((1830.0, 830.0)), OK))
    assert pressed == ["l", "l", "pause"]


def test_hands_free_typing_through_the_whole_app(qtbot, tmp_path) -> None:
    """Gaze on the real keyboard's registered keys types text: H, I; held L twice; 123, 7, ABC;
    and a paused app resumes only through Pause. The whole chain: TargetSync registration,
    engine dwell and repeat, the main-thread bridge, the keyboard and the typing session."""
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from gazeos.ui.app import GazeOSApp

    app = GazeOSApp(SimpleNamespace(home=str(tmp_path), profile="t", simulate=True, dry_run=True,
                                    gaze_model=None, camera=None))
    try:
        app.worker.stop()  # the test drives the engine itself
        app.worker.join(2.0)
        app.toggle_keyboard()
        qtbot.waitUntil(lambda: app.keyboard is not None and app.keyboard.isVisible(), timeout=30000)
        clock = [0.0]

        def look(prefix: str, seconds: float) -> None:
            app.sync.sync()
            target = next(t for t in app.registry.all() if t.key.startswith(prefix))
            cx, cy = target.rect.center
            for _ in range(int(round(seconds * 30))):
                clock[0] += 1 / 30
                app.engine.update(EngineInput(clock[0], _obs((cx, cy)), OK))
            qtbot.wait(50)  # let the queued button presses run on the GUI thread

        look("keyboard:key_h:", 1.0)
        look("keyboard:key_i:", 1.0)
        look("keyboard:key_l:", 2.4)  # held: typed twice
        assert app.keyboard is not None
        assert app.keyboard.session.text == "hill"
        look("keyboard:key_numbers:", 1.0)
        qtbot.wait(100)
        look("keyboard:key_sym_digit_seven:", 1.0)
        look("keyboard:key_letters:", 1.0)
        qtbot.wait(100)
        assert app.keyboard.session.text == "hill7"
        assert any(t.key.startswith("keyboard:key_a:") for t in (app.sync.sync() or app.registry.all()))
        app.engine.set_paused(True)
        look("keyboard:key_a:", 1.5)  # paused: keys do nothing
        assert app.keyboard.session.text == "hill7" and app.engine.paused
        look("action_bar:pause:", 1.2)  # Pause resumes
        assert not app.engine.paused
    finally:
        if app.keyboard is not None:
            app.keyboard.close()
        for w in (app.overlay, app.action_bar, app.menu, app.shortcuts, app.debug):
            w.close()
