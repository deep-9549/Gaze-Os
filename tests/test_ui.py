from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.qt

from gazeos.calibration.personalization import make_personalizer  # noqa: E402
from gazeos.control.engine import Action, EngineOutput  # noqa: E402
from gazeos.control.targets import TargetRegistry  # noqa: E402
from gazeos.control.zoom import make_zoom_region  # noqa: E402
from gazeos.pipeline import FrameProcessor  # noqa: E402
from gazeos.settings import CalibrationSettings, Settings  # noqa: E402
from gazeos.text_entry.layouts import get_layout  # noqa: E402
from gazeos.text_entry.phrases import PhraseBank  # noqa: E402
from gazeos.text_entry.predictor import Predictor  # noqa: E402
from gazeos.text_entry.session import TypingSession  # noqa: E402
from gazeos.text_entry.tts import SilentSpeaker  # noqa: E402
from gazeos.types import ControlState, TrackingStatus  # noqa: E402


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor.default(get_layout("qwerty"))


def test_calibration_window_end_to_end(qtbot, settings, geometry, intrinsics, user) -> None:
    """The calibration screen collects frames from the processing hook, fits, validates and scores."""
    from gazeos.ui.calibration import CalibrationWindow

    outcomes = []
    cal = CalibrationSettings(target_duration_s=0.3, keep_from_s=0.05, min_samples_per_point=3)
    win = CalibrationWindow(geometry, make_personalizer("polynomial", geometry), cal, outcomes.append)
    qtbot.addWidget(win)
    win.resize(800, 600)
    win.show()
    proc = FrameProcessor(settings, geometry, intrinsics, None)
    win.start()
    deadline = time.monotonic() + 15
    while win.phase in ("calibrate", "validate") and time.monotonic() < deadline:
        pts = win._points()
        if 0 <= win.index < len(pts):
            pf = proc.process_landmarks(user.look_at(*pts[win.index]), time.monotonic())
            win.add_frame(time.monotonic(), pf.observation, False)
        qtbot.wait(10)
    assert win.phase == "result" and win.result is not None
    assert win.result.score >= 70, win.result
    win._accept()
    assert outcomes and outcomes[0].validation.score == win.result.score


def test_calibration_repeats_a_dot_without_data_then_names_the_area(qtbot, settings, geometry, intrinsics,
                                                                   user) -> None:
    """A dot whose frames are all dropped (e.g. lids covering the eyes at the bottom of the
    screen) is shown once more; if it fails again calibration stops instead of guessing."""
    from gazeos.ui.calibration import CalibrationWindow

    cal = CalibrationSettings(target_duration_s=0.3, keep_from_s=0.05, min_samples_per_point=3)
    win = CalibrationWindow(geometry, make_personalizer("polynomial", geometry), cal, lambda o: None)
    qtbot.addWidget(win)
    win.resize(800, 600)
    win.show()
    proc = FrameProcessor(settings, geometry, intrinsics, None)
    blind = (0.5 * geometry.width_px, 0.9 * geometry.height_px)  # bottom centre
    win.start()
    deadline = time.monotonic() + 20
    while win.phase in ("calibrate", "validate") and time.monotonic() < deadline:
        pts = win._points()
        if 0 <= win.index < len(pts):
            pf = proc.process_landmarks(user.look_at(*pts[win.index]), time.monotonic())
            win.add_frame(time.monotonic(), None if pts[win.index] == blind else pf.observation, False)
        qtbot.wait(10)
    assert len(win.targets) == 10 and win.targets[-1] == blind  # shown once more at the end
    assert win.phase == "failed" and win.result is None
    assert "bottom" in win.fail_message


def test_keyboard_types_through_gaze_targets(qtbot, predictor) -> None:
    from gazeos.ui.common import MainThreadBridge, TargetSync
    from gazeos.ui.keyboard import KeyboardWindow

    typed: list[str] = []
    session = TypingSession(get_layout("qwerty"), predictor, emit=typed.append)
    kb = KeyboardWindow(session, PhraseBank(), SilentSpeaker(), 40.0, lambda: None)
    qtbot.addWidget(kb)
    kb.show()
    reg = TargetRegistry()
    bridge = MainThreadBridge()
    sync = TargetSync(reg, bridge)
    sync.track("keyboard", kb)
    keys = {t.key.split(":")[1]: t for t in reg.all()}
    assert {"key_h", "key_i", "key_space", "pred0"} <= set(keys)
    for k in ("key_h", "key_i"):
        keys[k].callback()  # what the engine calls when a dwell lands on the key
    qtbot.waitUntil(lambda: session.current_word == "hi", timeout=2000)
    assert "".join(typed) == "hi"
    assert kb.pred_buttons[0].text() != ""


def test_settings_dialog_edits_and_validates(qtbot, settings) -> None:
    from PySide6.QtWidgets import QPushButton

    from gazeos.ui.settings_dialog import SettingsDialog

    saved: list[Settings] = []
    dlg = SettingsDialog(settings, saved.append)
    qtbot.addWidget(dlg)
    dlg.show()
    inc = dlg.findChild(QPushButton, "set_dwell_ms_inc")
    toggle = dlg.findChild(QPushButton, "set_winks_enabled")
    cycle = dlg.findChild(QPushButton, "set_layout")
    assert inc is not None and toggle is not None and cycle is not None
    inc.click()
    toggle.click()
    cycle.click()
    for _ in range(40):  # clamps at the schema maximum instead of crashing
        inc.click()
    dlg.findChild(QPushButton, "settings_save").click()
    assert not saved and dlg.confirm is not None  # a 2 s dwell asks first (tests/test_safety.py)
    qtbot.addWidget(dlg.confirm)
    dlg.confirm.yes_btn.click()
    s = saved[0]
    assert 800 < s.dwell.dwell_ms <= 2000 and s.blink.winks_enabled and s.typing.layout == "alphabetical"


def test_privacy_panel_deletes(qtbot, tmp_path) -> None:
    from gazeos.profiles.store import ProfileStore
    from gazeos.ui.panels import PrivacyPanel

    p = ProfileStore(tmp_path).create("A")
    deleted = []
    panel = PrivacyPanel(p, lambda: deleted.append(1))
    qtbot.addWidget(panel)
    panel.show()
    assert "Settings" in panel.items.text()
    panel.delete_btn.click()
    assert p.path.exists() and panel.confirm is not None  # asks in a separate panel (tests/test_safety.py)
    qtbot.addWidget(panel.confirm)
    panel.confirm.yes_btn.click()
    assert not p.path.exists() and deleted


def test_overlay_and_action_bar_render(qtbot, geometry) -> None:
    from gazeos.ui.action_bar import ActionBar
    from gazeos.ui.overlay import Overlay

    ov = Overlay()
    qtbot.addWidget(ov)
    ov.show()
    out = EngineOutput(cursor=(400, 300), state=ControlState.DWELLING, status=TrackingStatus.TRACKING,
                       dwell_progress=0.5, sigma_px=50.0, action=Action.LEFT,
                       zoom=make_zoom_region((400, 300), geometry.width_px, geometry.height_px))
    ov.set_output(out)
    ov.show_toast("hello")
    ov.banner = "Research recording is ON"
    img = ov.grab()
    assert not img.isNull()
    actions = []
    bar = ActionBar(actions.append, lambda: None, lambda: None, lambda: None, lambda: None)
    qtbot.addWidget(bar)
    bar.buttons["drag"].click()
    assert actions == [Action.DRAG]


def test_onboarding_blink_tuning(qtbot) -> None:
    from gazeos.pipeline import ProcessedFrame
    from gazeos.types import FrameResult
    from gazeos.ui.onboarding import Onboarding

    done = []
    onb = Onboarding(lambda: None, lambda ms, closed: done.append((ms, closed)))
    qtbot.addWidget(onb)
    onb.show()
    onb.stack.setCurrentIndex(1)
    pf = ProcessedFrame(FrameResult(0.0), None)  # type: ignore[arg-type]
    img = np.zeros((480, 640, 3), np.uint8)
    peaks = (0.55, 0.6, 0.58, 0.62, 0.57, 0.74, 0.75, 0.73, 0.76, 0.74)  # long blinks close further
    for ms, peak in zip((150, 180, 200, 170, 160, 520, 600, 560, 650, 580), peaks, strict=True):
        onb.update_frame(pf, img, float(ms), peak)
    assert onb.threshold is not None and 200 < onb.threshold < 520
    # 90% of the long blinks' typical peak: above a keyboard glance (about 0.6) for this user.
    assert onb.closed_score is not None and abs(onb.closed_score - 0.9 * 0.74) < 1e-6
    onb._finish()
    assert done == [(onb.threshold, onb.closed_score)]


def test_overlay_painting_does_not_leak_none(qtbot, geometry) -> None:
    """Guards the PySide6 6.12.0 bug that over-released None in QPainter setters and
    crashed the 60 Hz overlay after a few minutes on Python 3.11."""
    import sys

    from gazeos.ui.overlay import Overlay

    ov = Overlay()
    qtbot.addWidget(ov)
    ov.resize(400, 300)
    ov.set_output(EngineOutput(cursor=(100, 100), state=ControlState.DWELLING, status=TrackingStatus.TRACKING,
                               dwell_progress=0.4, sigma_px=30.0))
    before = sys.getrefcount(None)
    for _ in range(300):
        ov.grab()
    assert sys.getrefcount(None) - before > -100


def test_personal_closed_score_ignores_a_keyboard_glance(settings, geometry, intrinsics, user) -> None:
    """With the tuned level (0.67), lids at 0.6 (looking down at the keyboard) are not a
    closure, so no long-blink click; a real long blink (0.75) still is (ADR 0007)."""
    from gazeos.gestures.blink import BlinkEvent

    s = settings.model_copy(deep=True)
    s.blink.closed_score = 0.67
    proc = FrameProcessor(s, geometry, intrinsics, None)
    events = []
    t = 0.0
    for score, frames in ((0.05, 10), (0.6, 18), (0.05, 10), (0.75, 18), (0.05, 10)):  # 600 ms each
        for _ in range(frames):
            lm = user.look_at(960, 540)
            lm.blendshapes = {**(lm.blendshapes or {}), "eyeBlinkLeft": score, "eyeBlinkRight": score}
            events.append(proc.process_landmarks(lm, t).engine_input.blink_event)
            t += 1 / 30
    assert [e for e in events if e is not None] == [BlinkEvent.LONG_BLINK]


def test_small_target_test_records_hits_misses_and_wrong_clicks(qtbot) -> None:
    from tools.small_targets import SmallTargetTest, make_trials

    trials = make_trials(per_size=1, seed=0)
    w = SmallTargetTest(trials)
    qtbot.addWidget(w)
    w.resize(1280, 800)
    w.start()
    target = trials[0]["target"]
    w.buttons[(target + 1) % 7].click()  # a wrong button first
    w.buttons[target].click()
    w.buttons[trials[1]["target"]].click()
    w.t_start -= 25  # trial 3 times out
    w._check_timeout()
    r = w.results
    assert [x["hit"] for x in r] == [True, True, False]
    assert (r[0]["first_click_correct"], r[0]["wrong_clicks"]) == (False, 1)
    assert r[1]["first_click_correct"] and r[1]["wrong_clicks"] == 0
    assert w.buttons[0].width() == trials[3]["size"]  # now on trial 4
