from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gazeos.os_backends.base import InputBackend, RecordingBackend, get_backend
from gazeos.os_backends.keys import DESTRUCTIVE_SHORTCUTS, shortcut_keys
from gazeos.types import Rect, UIElement


def test_recording_backend_protocol() -> None:
    b = RecordingBackend(elements=[UIElement(Rect(10, 10, 50, 20))])
    assert isinstance(b, InputBackend)
    b.move(1, 2)
    b.click("right", 2)
    b.hotkey("ctrl", "c")
    assert b.events == [("move", 1, 2), ("click", "right", 2), ("hotkey", "ctrl", "c")]
    assert b.element_at(20, 15) is not None and b.element_at(200, 200) is None
    assert len(b.elements_near(80, 20, 30)) == 1
    assert isinstance(get_backend(dry_run=True), RecordingBackend)


def test_shortcut_tables() -> None:
    assert shortcut_keys("copy", "win32") == ("ctrl", "c")
    assert shortcut_keys("copy", "darwin") == ("cmd", "c")
    assert "close_window" in DESTRUCTIVE_SHORTCUTS


@pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("Xvfb") is None, reason="needs Xvfb")
def test_x11_backend_events_arrive_in_order() -> None:
    """PRD 23 'OS backend': every event arrives, in order, at the right coordinates."""
    pytest.importorskip("Xlib")
    display = ":97"
    xvfb = subprocess.Popen(["Xvfb", display, "-screen", "0", "800x600x24", "-nolisten", "tcp"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        env = {**os.environ, "DISPLAY": display, "QT_QPA_PLATFORM": "xcb"}
        probe = Path(__file__).with_name("x11_probe.py")
        out = subprocess.run([sys.executable, str(probe)], env=env, capture_output=True, text=True, timeout=60)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("{")), None)
        assert line is not None, out.stderr[-2000:]
        data = json.loads(line)
        ev = data["events"]
        assert data["size"] == [800, 600]
        presses = [e for e in ev if e[0] == "press"]
        assert presses[0][1] == "LeftButton" and abs(presses[0][2] - 100) <= 2 and abs(presses[0][3] - 120) <= 2
        assert presses[1][1] == "RightButton" and abs(presses[1][2] - 300) <= 2
        wheels = [e for e in ev if e[0] == "wheel"]
        assert wheels and all(w[1] < 0 for w in wheels)  # positive dy scrolls content down
        keys = [e for e in ev if e[0] == "key"]
        assert [k[1] for k in keys[:2]] == ["h", "i"]
        assert any(k[2] & 0x04000000 for k in keys)  # Ctrl held for the hotkey
    finally:
        xvfb.terminate()
        xvfb.wait(timeout=5)


@pytest.mark.skipif(sys.platform == "win32", reason="uses the Linux and macOS file locations")
def test_autostart_files(tmp_path, monkeypatch) -> None:
    from gazeos import autostart

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(autostart.Path, "home", lambda: tmp_path)
    for platform in ("linux", "darwin"):
        autostart.set_enabled(True, platform)
        assert autostart.is_enabled(platform)
        autostart.set_enabled(False, platform)
        assert not autostart.is_enabled(platform)
    autostart.set_enabled(True, "linux")
    assert "gazeos.main" in (tmp_path / "autostart" / "gazeos.desktop").read_text()


class _Box:
    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self.left, self.top, self.right, self.bottom = left, top, right, bottom


class _Ctrl:
    """A UI Automation control, as far as nearby_elements uses one."""

    def __init__(self, name: str, role: str, rect: tuple[int, int, int, int], children=()) -> None:
        self.Name, self.ControlTypeName, self.BoundingRectangle = name, role, _Box(*rect)
        self.children, self.parent = list(children), None
        for c in self.children:
            c.parent = self

    def GetChildren(self):  # noqa: N802
        return self.children

    def GetParentControl(self):  # noqa: N802
        return self.parent

    def at(self, x: int, y: int):
        for c in self.children:
            r = c.BoundingRectangle
            if r.left <= x < r.right and r.top <= y < r.bottom:
                return c.at(x, y)
        return self


def test_nearby_elements_finds_small_icons_between_probe_points() -> None:
    """Probing every radius/2 found only buttons of 48 px and up (28 Oct, real UIA on Qt);
    searching the containers finds 16 px icons too, and nothing outside the radius."""
    from gazeos.os_backends.nearby import nearby_elements

    icons = [_Ctrl(f"icon{i}", "ButtonControl", (300 + 24 * i, 100, 316 + 24 * i, 116)) for i in range(20)]
    toolbar = _Ctrl("toolbar", "ToolBarControl", (290, 90, 800, 126), icons)
    window = _Ctrl("app", "WindowControl", (0, 0, 1920, 1080), [toolbar, _Ctrl("text", "EditControl", (0, 200, 1920, 1080))])
    desktop = _Ctrl("desktop", "PaneControl", (0, 0, 1920, 1080), [window])

    def to_element(c):
        r = c.BoundingRectangle
        return UIElement(Rect(r.left, r.top, r.right - r.left, r.bottom - r.top), role=c.ControlTypeName, name=c.Name,
                         invokable=c.ControlTypeName == "ButtonControl")

    names = {e.name for e in nearby_elements(desktop.at, to_element, 450, 108, 60)}
    # icons whose rectangle is within 60 px of x=450: left edges 300+24i in [382, 510]
    assert names == {f"icon{i}" for i in range(20) if 300 + 24 * i + 16 >= 390 and 300 + 24 * i <= 510}
