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
    cal = CalibrationSettings(target_duration_s=0.3, keep_from_s=0.05)
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
    s = saved[0]
    assert 800 < s.dwell.dwell_ms <= 2000 and s.blink.winks_enabled and s.typing.layout == "alphabetical"


def test_privacy_panel_deletes(qtbot, tmp_path) -> None:
    from PySide6.QtWidgets import QPushButton

    from gazeos.profiles.store import ProfileStore
    from gazeos.ui.panels import PrivacyPanel

    p = ProfileStore(tmp_path).create("A")
    deleted = []
    panel = PrivacyPanel(p, lambda: deleted.append(1))
    qtbot.addWidget(panel)
    assert "Settings" in panel.items.text()
    btn = panel.findChild(QPushButton, "privacy_delete")
    btn.click()
    assert p.path.exists()  # first click only arms the confirmation
    btn.click()
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
    onb = Onboarding(lambda: None, done.append)
    qtbot.addWidget(onb)
    onb.show()
    onb.stack.setCurrentIndex(1)
    pf = ProcessedFrame(FrameResult(0.0), None)  # type: ignore[arg-type]
    img = np.zeros((480, 640, 3), np.uint8)
    for ms in (150, 180, 200, 170, 160, 520, 600, 560, 650, 580):
        onb.update_frame(pf, img, float(ms))
    assert onb.threshold is not None and 200 < onb.threshold < 520
    onb._finish()
    assert done == [onb.threshold]


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
