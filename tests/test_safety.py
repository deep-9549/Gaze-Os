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
# -- stray glances: every GazeOS button is pressed by looking at it for the dwell time -----------


@pytest.fixture
def gaze_app(qtbot, tmp_path):
    """The whole app in --simulate mode with a throwaway profile: no camera, no real input."""
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
    yield app
    app.worker.stop()
    app.worker.join(2.0)
    if hasattr(app, "_hotkeys"):
        app._hotkeys.stop()
    for w in (app.overlay, app.action_bar, app.menu, app.shortcuts, app.debug, app.keyboard, app._confirm_panel,
              getattr(app, "_privacy", None), getattr(app, "_settings_dialog", None)):
        if w is not None:
            w.close()


@pytest.fixture
def quits(gaze_app, monkeypatch) -> list[int]:
    """Records quitting instead of stopping the app (and the test's Qt application)."""
    calls: list[int] = []
    monkeypatch.setattr(gaze_app, "quit", lambda: calls.append(1))
    return calls


def test_quit_from_the_gaze_menu_needs_a_separate_yes(qtbot, gaze_app, quits) -> None:
    """Quit asks in its own panel, whose Yes is well away from the Quit button; No is safe."""
    from gazeos.ui.panels import MIN_YES_CLEARANCE_PX, global_rect

    app = gaze_app
    app._show_menu()
    qtbot.waitUntil(app.menu.isVisible)
    quit_btn = app.menu.buttons["menu_quit"]
    asked_at = global_rect(quit_btn)
    quit_btn.click()
    panel = app._confirm_panel
    assert panel is not None and panel.isVisible() and not app.menu.isVisible() and not quits
    assert panel.yes_btn.text() == "Yes, quit"
    assert panel.yes_clearance() >= MIN_YES_CLEARANCE_PX and not global_rect(panel.yes_btn).intersects(asked_at)
    groups = {t.group for t in app.registry.all()}
    assert "confirm" in groups and "menu" not in groups  # the hidden menu's buttons cannot be pressed
    panel.no_btn.click()
    assert not panel.isVisible() and not quits
    app._show_menu()
    quit_btn.click()
    assert app._confirm_panel is not panel and not quits
    app._confirm_panel.yes_btn.click()
    assert quits == [1]


def test_an_unanswered_question_is_no(qtbot, gaze_app, quits) -> None:
    from gazeos.ui.panels import CONFIRM_TIMEOUT_S

    app = gaze_app
    app._show_menu()
    qtbot.waitUntil(app.menu.isVisible)
    app.menu.buttons["menu_quit"].click()
    panel = app._confirm_panel
    assert panel is not None and panel.idle_timer.isActive()
    assert panel.idle_timer.interval() == CONFIRM_TIMEOUT_S * 1000 == 10000
    panel.idle_timer.start(50)  # instead of waiting 10 s
    qtbot.waitUntil(lambda: not panel.isVisible(), timeout=5000)
    assert panel.answer is False and not quits
    panel.yes_btn.click()  # a press that arrives after the timeout does nothing
    assert not quits


def test_one_long_stare_at_quit_cannot_quit(qtbot, gaze_app, quits) -> None:
    """The stare that pressed Quit goes on for seconds, wandering over the button: it cannot
    press Yes too (No sits under the eye and Yes is far off). Looking at Yes does quit."""
    from PySide6.QtWidgets import QApplication

    from gazeos.ui.panels import global_rect

    app = gaze_app
    app.worker.stop()  # the engine is fed from here, frame by frame
    app.worker.join(2.0)
    app.engine.apply_settings(app.settings)  # fresh dwell, filter and fixation state
    app._show_menu()
    qtbot.waitUntil(app.menu.isVisible)
    app.sync.sync()
    quit_target = next(t for t in app.registry.all() if t.key.startswith("menu:menu_quit:"))
    clock = [time.monotonic()]  # the simulated frames so far carried monotonic times

    def look(x: float, y: float, seconds: float) -> None:
        for i in range(int(seconds * 30)):
            clock[0] += 1 / 30
            px = (x + 70 * np.sin(0.7 * i), y + 25 * np.cos(1.3 * i))  # restless, but on the button
            app.engine.update(EngineInput(clock[0], _obs(px), OK))
            QApplication.processEvents()  # a gaze press reaches its button through the bridge
            app.sync.sync()  # as the 300 ms timer would

    look(*quit_target.rect.center, 6.0)
    assert not quits
    panel = app._confirm_panel
    assert panel is not None and not app.menu.isVisible()  # the stare did press Quit...
    assert panel.isVisible() and panel.answer is None  # ...and nothing after it
    yes = global_rect(panel.yes_btn).center()
    look(yes.x(), yes.y(), 1.5)
    assert quits == [1]


def test_close_window_asks_with_yes_away_from_the_button(qtbot, gaze_app) -> None:
    from gazeos.ui.panels import MIN_YES_CLEARANCE_PX, global_rect

    app = gaze_app
    app._show_shortcuts()
    qtbot.waitUntil(app.shortcuts.isVisible)
    button = app.shortcuts.buttons["sc_close_window"]
    asked_at = global_rect(button)
    button.click()
    qtbot.waitUntil(lambda: app._confirm_panel is not None and app._confirm_panel.isVisible())
    panel = app._confirm_panel
    assert not app.backend.of_kind("hotkey")
    assert panel.yes_clearance() >= MIN_YES_CLEARANCE_PX and not global_rect(panel.yes_btn).intersects(asked_at)
    panel.yes_btn.click()
    assert app.backend.of_kind("hotkey") == [("hotkey", "alt", "f4")]


def test_delete_all_my_data_needs_the_separate_yes(qtbot, tmp_path) -> None:
    """Pressing 'Delete all my data' twice no longer deletes: a separate panel asks, with Yes
    away from the button; No (or no answer) brings the privacy panel back and keeps the data."""
    pytest.importorskip("PySide6")
    from gazeos.profiles.store import ProfileStore
    from gazeos.ui.panels import MIN_YES_CLEARANCE_PX, ConfirmPanel, PrivacyPanel, global_rect

    p = ProfileStore(tmp_path).create("A")
    deleted: list[int] = []
    asked: list[ConfirmPanel] = []

    def present(panel: ConfirmPanel) -> None:
        qtbot.addWidget(panel)
        asked.append(panel)
        panel.show()

    privacy = PrivacyPanel(p, lambda: deleted.append(1), present=present)
    qtbot.addWidget(privacy)
    privacy.show()
    asked_at = global_rect(privacy.delete_btn)
    privacy.delete_btn.click()
    privacy.delete_btn.click()  # the same spot again: the old scheme deleted here
    assert p.path.exists() and len(asked) == 1 and not privacy.isVisible()
    confirm = asked[0]
    assert confirm.isVisible() and confirm.yes_btn.text() == "Yes, delete everything"
    assert confirm.yes_clearance() >= MIN_YES_CLEARANCE_PX and not global_rect(confirm.yes_btn).intersects(asked_at)
    confirm.no_btn.click()
    assert privacy.isVisible() and p.path.exists() and not deleted
    privacy.delete_btn.click()
    asked[1].idle_timer.start(50)  # no answer
    qtbot.waitUntil(lambda: not asked[1].isVisible(), timeout=5000)
    assert privacy.isVisible() and p.path.exists() and not deleted
    privacy.delete_btn.click()
    asked[2].yes_btn.click()
    assert not p.path.exists() and deleted == [1] and not privacy.isVisible()


def test_idle_menus_and_panels_close_themselves(qtbot, gaze_app, settings) -> None:
    """A panel opened by accident does not stay open: nothing pressed for a while closes it."""
    from gazeos.ui.panels import ButtonPanel, PrivacyPanel
    from gazeos.ui.settings_dialog import SettingsDialog

    app = gaze_app
    assert app.menu.idle_timer.interval() == 15000 and app.shortcuts.idle_timer.interval() == 15000
    app._show_menu()
    qtbot.waitUntil(app.menu.isVisible)
    assert app.menu.idle_timer.isActive()
    app.menu.idle_timer.start(50)
    qtbot.waitUntil(lambda: not app.menu.isVisible(), timeout=5000)
    assert PrivacyPanel(app.profile, lambda: None).idle_timer.interval() == 30000  # text to read
    assert SettingsDialog(settings, lambda s: None).idle_timer.interval() == 60000  # unsaved edits

    presses: list[int] = []
    panel = ButtonPanel("T", [("A", "a", lambda: presses.append(1))], close_after=False, idle_s=0.6)
    qtbot.addWidget(panel)
    panel.show()
    qtbot.wait(400)
    panel.buttons["a"].click()  # a press starts the count again
    assert presses == [1] and panel.isVisible() and panel.idle_timer.remainingTime() > 450
    qtbot.waitUntil(lambda: not panel.isVisible(), timeout=5000)
    panel.buttons["a"].click()  # left over from before it closed: ignored
    assert presses == [1]


def test_menu_items_can_be_undone_by_gaze(qtbot, gaze_app) -> None:
    """Hide gaze cursor, Debug view and Pause each leave a way back that gaze alone can use."""
    from PySide6.QtWidgets import QApplication

    app = gaze_app
    item = app.menu.buttons["menu_overlay"]
    app._show_menu()
    qtbot.waitUntil(app.menu.isVisible)
    item.click()
    assert not app.overlay.show_cursor and item.text() == "Show gaze cursor"
    app._show_menu()
    item.click()
    assert app.overlay.show_cursor and item.text() == "Hide gaze cursor"

    app._toggle_debug()
    assert app.debug.isVisible()
    app.sync.sync()
    close = [t for t in app.registry.all() if t.key.startswith("debug:debug_close:")]
    assert len(close) == 1
    close[0].callback()  # what a dwell on it does
    QApplication.processEvents()
    assert not app.debug.isVisible()

    app.toggle_pause()
    assert app.engine.paused and app.overlay.toast is not None and "top-left corner" in app.overlay.toast[0]
    app.toggle_pause()
    assert not app.engine.paused


def test_hidden_cursor_keeps_the_dwell_ring(qtbot) -> None:
    pytest.importorskip("PySide6")
    from gazeos.control.engine import EngineOutput
    from gazeos.types import ControlState
    from gazeos.ui.overlay import Overlay

    ov = Overlay()
    qtbot.addWidget(ov)
    ov.show()
    ov.resize(400, 300)
    ov.show_cursor = False

    def alpha(progress: float, x: int, y: int) -> int:
        ov.set_output(EngineOutput(cursor=(200, 150), state=ControlState.DWELLING, status=TrackingStatus.TRACKING,
                                   dwell_progress=progress, sigma_px=30.0))
        return int(ov.grab().toImage().pixelColor(x, y).alpha())

    assert alpha(0.5, 226, 150) > 0  # the ring, half full: its 3 o'clock point is drawn
    assert alpha(0.5, 200, 150) == 0  # no cursor dot
    assert alpha(0.0, 226, 150) == 0  # no dwell, no ring


def _tab(tabs, title: str):
    """The page of the settings tab called `title`."""
    return tabs.widget(next(i for i in range(tabs.count()) if tabs.tabText(i) == title))


def test_buttons_out_of_view_are_not_gaze_targets(qtbot, settings) -> None:
    """Settings rows scrolled out of view were still gaze targets, some under Save: a dwell there
    pressed a button nobody could see. Only what is visible is a target, and only while shown."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QPushButton, QScrollArea, QTabWidget

    from gazeos.control.targets import TargetRegistry
    from gazeos.ui.common import MainThreadBridge, TargetSync
    from gazeos.ui.panels import global_rect
    from gazeos.ui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(settings, lambda s: None)
    qtbot.addWidget(dlg)
    dlg.show()
    tabs = dlg.findChild(QTabWidget)
    scroll = _tab(tabs, "Cursor")  # 19 rows: more than fit
    tabs.setCurrentWidget(scroll)
    assert isinstance(scroll, QScrollArea)
    view = global_rect(scroll.viewport())
    hidden = {b.objectName() for b in scroll.findChildren(QPushButton)
              if b.isVisible() and not view.intersects(global_rect(b))}
    assert hidden  # some rows are out of view, so the test means something
    reg = TargetRegistry()
    sync = TargetSync(reg, MainThreadBridge())
    sync.track("settings", dlg)
    rows = [t for t in reg.all() if t.key.startswith("settings:set_")]
    assert rows and all(view.contains(QRect(t.rect.left, t.rect.top, t.rect.width, t.rect.height)) for t in rows)
    assert not {t.key.split(":")[1] for t in rows} & hidden
    dlg.hide()
    assert not reg.all()  # gone as soon as the window hides, not at the next sync


def test_risky_settings_ask_before_saving(qtbot, settings) -> None:
    """A stray dwell on Save keeps no change that can strand a person who uses only their eyes:
    those ask in a separate panel (Yes away from Save); No brings the dialog back, edits kept."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QLabel, QPushButton, QTabWidget

    from gazeos.ui.panels import ConfirmPanel
    from gazeos.ui.settings_dialog import SettingsDialog

    saved: list = []
    asked: list[ConfirmPanel] = []

    def present(panel: ConfirmPanel) -> None:
        qtbot.addWidget(panel)
        asked.append(panel)
        panel.show()

    dlg = SettingsDialog(settings, saved.append, present=present)
    qtbot.addWidget(dlg)
    dlg.show()
    dlg.findChild(QPushButton, "set_layout").click()  # a harmless change saves at once
    dlg.save_btn.click()
    assert len(saved) == 1 and not asked

    dlg = SettingsDialog(settings, saved.append, present=present)
    qtbot.addWidget(dlg)
    dlg.show()
    _tab(dlg.findChild(QTabWidget), "Dwell").findChild(QPushButton, "set_enabled").click()  # dwell off
    dlg.save_btn.click()
    assert len(saved) == 1 and len(asked) == 1 and not dlg.isVisible()
    assert any("Dwell clicking off" in label.text() for label in asked[0].findChildren(QLabel))
    asked[0].no_btn.click()
    assert dlg.isVisible() and len(saved) == 1 and not dlg.settings.dwell.enabled  # back, edit kept
    dlg.save_btn.click()
    asked[1].yes_btn.click()
    assert len(saved) == 2 and not saved[1].dwell.enabled and not dlg.isVisible()


def test_which_settings_changes_are_risky(settings) -> None:
    from gazeos.settings import ControlMode
    from gazeos.ui.settings_dialog import risky_changes

    def changed(**edits) -> list[str]:
        """`section__field=value`, or `field=value` for a top-level field."""
        new = settings.model_copy(deep=True)
        for path, value in edits.items():
            section, _, name = path.partition("__")
            if name:
                setattr(getattr(new, section), name, value)
            else:
                setattr(new, section, value)
        return risky_changes(settings, new)

    assert not changed(theme="high_contrast")
    assert not changed(dwell__dwell_ms=1000.0) and not changed(typing__layout="alphabetical")
    for edit in ({"dwell__enabled": False}, {"mode": ControlMode.HYBRID}, {"dwell__dwell_ms": 2000.0},
                 {"dwell__dwell_ms": 300.0}, {"camera__device_index": 1}, {"system__show_overlay": False},
                 {"system__corner_size_px": 12}, {"system__confirm_destructive_shortcuts": False},
                 {"system__auto_resume_after_face_lost": False}, {"calibration__auto_start_s": 0.0},
                 {"gestures__enabled": True}):
        assert len(changed(**edit)) == 1, edit
    on = settings.model_copy(deep=True)
    on.system.start_with_os = True
    off = on.model_copy(deep=True)
    off.system.start_with_os = False
    assert risky_changes(on, off) and not risky_changes(off, on)
