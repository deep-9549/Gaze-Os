"""Input backend interface and platform selection (PRD 20.1)."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from gazeos.types import UIElement

log = logging.getLogger(__name__)


@runtime_checkable
class InputBackend(Protocol):
    def move(self, x: int, y: int) -> None: ...
    def click(self, button: str = "left", count: int = 1) -> None: ...
    def press(self, button: str) -> None: ...  # drag start
    def release(self, button: str) -> None: ...  # drag end
    def scroll(self, dx: int, dy: int) -> None: ...
    def type_text(self, text: str) -> None: ...
    def hotkey(self, *keys: str) -> None: ...
    def element_at(self, x: int, y: int) -> UIElement | None: ...  # for snapping
    def elements_near(self, x: int, y: int, radius: int) -> list[UIElement]: ...
    def screen_size(self) -> tuple[int, int]: ...


@dataclass
class RecordingBackend:
    """Records every event; used by tests and the --dry-run mode."""

    size: tuple[int, int] = (1920, 1080)
    elements: list[UIElement] = field(default_factory=list)
    events: list[tuple[object, ...]] = field(default_factory=list)

    def move(self, x: int, y: int) -> None:
        self.events.append(("move", x, y))

    def click(self, button: str = "left", count: int = 1) -> None:
        self.events.append(("click", button, count))

    def press(self, button: str) -> None:
        self.events.append(("press", button))

    def release(self, button: str) -> None:
        self.events.append(("release", button))

    def scroll(self, dx: int, dy: int) -> None:
        self.events.append(("scroll", dx, dy))

    def type_text(self, text: str) -> None:
        self.events.append(("type", text))

    def hotkey(self, *keys: str) -> None:
        self.events.append(("hotkey", *keys))

    def element_at(self, x: int, y: int) -> UIElement | None:
        for e in self.elements:
            if e.rect.contains(x, y):
                return e
        return None

    def elements_near(self, x: int, y: int, radius: int) -> list[UIElement]:
        out = []
        for e in self.elements:
            cx, cy = e.rect.center
            if abs(cx - x) <= radius + e.rect.width / 2 and abs(cy - y) <= radius + e.rect.height / 2:
                out.append(e)
        return out

    def screen_size(self) -> tuple[int, int]:
        return self.size

    def of_kind(self, kind: str) -> list[tuple[object, ...]]:
        return [e for e in self.events if e[0] == kind]


def get_backend(dry_run: bool = False) -> InputBackend:
    """Pick the best backend for this platform, falling back to pynput."""
    if dry_run:
        return RecordingBackend()
    try:
        if sys.platform == "win32":
            from gazeos.os_backends.windows import WindowsBackend

            return WindowsBackend()
        if sys.platform == "darwin":
            from gazeos.os_backends.macos import MacBackend

            return MacBackend()
        if sys.platform.startswith("linux"):
            import os

            if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY"):
                from gazeos.os_backends.linux_uinput import UInputBackend

                return UInputBackend()
            from gazeos.os_backends.linux_x11 import X11Backend

            return X11Backend()
    except Exception as exc:  # pragma: no cover - platform specific
        log.warning("native input backend unavailable (%s); using pynput", exc)
    from gazeos.os_backends.pynput_backend import PynputBackend

    return PynputBackend()
