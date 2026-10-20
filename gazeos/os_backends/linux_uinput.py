"""Linux Wayland input backend through a virtual uinput device (PRD 20.4, stretch).

Wayland blocks synthetic input from ordinary apps, so GazeOS creates a virtual
absolute pointer and keyboard. The user must be in the `input` group (or have
a udev rule for /dev/uinput).
"""

from __future__ import annotations

import time
from typing import Any

from evdev import AbsInfo, UInput, ecodes  # type: ignore[import-not-found]

from gazeos.types import UIElement

_BUTTONS = {"left": ecodes.BTN_LEFT, "right": ecodes.BTN_RIGHT, "middle": ecodes.BTN_MIDDLE}
_KEYS = {
    "ctrl": ecodes.KEY_LEFTCTRL, "alt": ecodes.KEY_LEFTALT, "shift": ecodes.KEY_LEFTSHIFT,
    "win": ecodes.KEY_LEFTMETA, "cmd": ecodes.KEY_LEFTMETA, "enter": ecodes.KEY_ENTER,
    "esc": ecodes.KEY_ESC, "tab": ecodes.KEY_TAB, "backspace": ecodes.KEY_BACKSPACE,
    "delete": ecodes.KEY_DELETE, "space": ecodes.KEY_SPACE, "left": ecodes.KEY_LEFT,
    "right": ecodes.KEY_RIGHT, "up": ecodes.KEY_UP, "down": ecodes.KEY_DOWN,
    **{f"f{i}": getattr(ecodes, f"KEY_F{i}") for i in range(1, 13)},
}
_PUNCT = {
    ".": ("KEY_DOT", False), ",": ("KEY_COMMA", False), "?": ("KEY_SLASH", True),
    "!": ("KEY_1", True), "'": ("KEY_APOSTROPHE", False), "-": ("KEY_MINUS", False),
    "\n": ("KEY_ENTER", False), " ": ("KEY_SPACE", False),
}


class UInputBackend:
    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self.width, self.height = width, height
        caps: dict[int, Any] = {
            ecodes.EV_KEY: list(_BUTTONS.values()) + list(_KEYS.values())
            + [getattr(ecodes, f"KEY_{c}") for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"]
            + [getattr(ecodes, n) for n, _ in _PUNCT.values()],
            ecodes.EV_ABS: [
                (ecodes.ABS_X, AbsInfo(0, 0, width - 1, 0, 0, 0)),
                (ecodes.ABS_Y, AbsInfo(0, 0, height - 1, 0, 0, 0)),
            ],
            ecodes.EV_REL: [ecodes.REL_WHEEL, ecodes.REL_HWHEEL],
        }
        self.ui = UInput(caps, name="gazeos-virtual-input")
        time.sleep(0.2)  # let the compositor pick up the device

    def _syn(self) -> None:
        self.ui.syn()

    def move(self, x: int, y: int) -> None:
        self.ui.write(ecodes.EV_ABS, ecodes.ABS_X, int(x))
        self.ui.write(ecodes.EV_ABS, ecodes.ABS_Y, int(y))
        self._syn()

    def click(self, button: str = "left", count: int = 1) -> None:
        for _ in range(count):
            self.press(button)
            self.release(button)

    def press(self, button: str) -> None:
        self.ui.write(ecodes.EV_KEY, _BUTTONS[button], 1)
        self._syn()

    def release(self, button: str) -> None:
        self.ui.write(ecodes.EV_KEY, _BUTTONS[button], 0)
        self._syn()

    def scroll(self, dx: int, dy: int) -> None:
        if dy:
            self.ui.write(ecodes.EV_REL, ecodes.REL_WHEEL, -int(round(dy / 120)) or (-1 if dy > 0 else 1))
        if dx:
            self.ui.write(ecodes.EV_REL, ecodes.REL_HWHEEL, int(round(dx / 120)) or (1 if dx > 0 else -1))
        self._syn()

    def _tap(self, code: int, shift: bool = False) -> None:
        if shift:
            self.ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 1)
        self.ui.write(ecodes.EV_KEY, code, 1)
        self.ui.write(ecodes.EV_KEY, code, 0)
        if shift:
            self.ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 0)
        self._syn()

    def type_text(self, text: str) -> None:
        """US keymap only: uinput sends key codes, not characters."""
        for ch in text:
            if ch.isalnum() and ch.isascii():
                self._tap(getattr(ecodes, f"KEY_{ch.upper()}"), shift=ch.isupper())
            elif ch in _PUNCT:
                name, shift = _PUNCT[ch]
                self._tap(getattr(ecodes, name), shift=shift)

    def hotkey(self, *keys: str) -> None:
        codes = [_KEYS.get(k.lower()) or getattr(ecodes, f"KEY_{k.upper()}") for k in keys]
        for c in codes:
            self.ui.write(ecodes.EV_KEY, c, 1)
        for c in reversed(codes):
            self.ui.write(ecodes.EV_KEY, c, 0)
        self._syn()

    def element_at(self, x: int, y: int) -> UIElement | None:
        from gazeos.os_backends.atspi import element_at

        return element_at(x, y)

    def elements_near(self, x: int, y: int, radius: int) -> list[UIElement]:
        e = self.element_at(x, y)
        return [e] if e is not None else []

    def screen_size(self) -> tuple[int, int]:
        return self.width, self.height
