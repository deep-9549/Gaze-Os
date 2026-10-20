"""Linux X11 input backend through the XTest extension (PRD 20.4, stretch)."""

from __future__ import annotations

from Xlib import X, XK, display  # type: ignore[import-not-found]
from Xlib.ext import xtest  # type: ignore[import-not-found]

from gazeos.types import UIElement

_BUTTONS = {"left": 1, "middle": 2, "right": 3}
_KEYSYM_NAMES = {
    "ctrl": "Control_L", "alt": "Alt_L", "shift": "Shift_L", "win": "Super_L", "cmd": "Super_L",
    "enter": "Return", "esc": "Escape", "tab": "Tab", "backspace": "BackSpace",
    "delete": "Delete", "space": "space", "left": "Left", "right": "Right", "up": "Up",
    "down": "Down", "home": "Home", "end": "End", "pageup": "Prior", "pagedown": "Next",
    **{f"f{i}": f"F{i}" for i in range(1, 13)},
}


class X11Backend:
    def __init__(self, display_name: str | None = None) -> None:
        self.d = display.Display(display_name)
        if not self.d.has_extension("XTEST"):
            raise RuntimeError("X server has no XTEST extension")
        self.root = self.d.screen().root

    def _flush(self) -> None:
        self.d.sync()

    def move(self, x: int, y: int) -> None:
        xtest.fake_input(self.d, X.MotionNotify, x=int(x), y=int(y))
        self._flush()

    def click(self, button: str = "left", count: int = 1) -> None:
        b = _BUTTONS[button]
        for _ in range(count):
            xtest.fake_input(self.d, X.ButtonPress, b)
            xtest.fake_input(self.d, X.ButtonRelease, b)
        self._flush()

    def press(self, button: str) -> None:
        xtest.fake_input(self.d, X.ButtonPress, _BUTTONS[button])
        self._flush()

    def release(self, button: str) -> None:
        xtest.fake_input(self.d, X.ButtonRelease, _BUTTONS[button])
        self._flush()

    def scroll(self, dx: int, dy: int) -> None:
        # X11 wheel: buttons 4/5 vertical, 6/7 horizontal; one notch per 120 units.
        for amount, (neg, pos) in ((dy, (4, 5)), (dx, (6, 7))):
            notches = int(round(abs(amount) / 120)) or (1 if amount else 0)
            b = pos if amount > 0 else neg
            for _ in range(notches):
                xtest.fake_input(self.d, X.ButtonPress, b)
                xtest.fake_input(self.d, X.ButtonRelease, b)
        self._flush()

    def _keycode(self, key: str) -> int:
        name = _KEYSYM_NAMES.get(key.lower(), key)
        sym = XK.string_to_keysym(name)
        if sym == 0 and len(key) == 1:
            sym = ord(key)
        return int(self.d.keysym_to_keycode(sym))

    def _tap(self, keycode: int, shift: bool = False) -> None:
        shift_code = self._keycode("shift")
        if shift:
            xtest.fake_input(self.d, X.KeyPress, shift_code)
        xtest.fake_input(self.d, X.KeyPress, keycode)
        xtest.fake_input(self.d, X.KeyRelease, keycode)
        if shift:
            xtest.fake_input(self.d, X.KeyRelease, shift_code)

    def type_text(self, text: str) -> None:
        for ch in text:
            name = {" ": "space", "\n": "Return"}.get(ch, ch)
            sym = XK.string_to_keysym(name) or ord(ch)
            code = self.d.keysym_to_keycode(sym)
            if code == 0:
                continue  # character not on the current keymap
            shifted = self.d.keycode_to_keysym(code, 0) != sym
            self._tap(int(code), shift=shifted)
        self._flush()

    def hotkey(self, *keys: str) -> None:
        codes = [self._keycode(k) for k in keys]
        for c in codes:
            xtest.fake_input(self.d, X.KeyPress, c)
        for c in reversed(codes):
            xtest.fake_input(self.d, X.KeyRelease, c)
        self._flush()

    def element_at(self, x: int, y: int) -> UIElement | None:
        from gazeos.os_backends.atspi import element_at

        return element_at(x, y)

    def elements_near(self, x: int, y: int, radius: int) -> list[UIElement]:
        e = self.element_at(x, y)
        return [e] if e is not None else []

    def screen_size(self) -> tuple[int, int]:
        s = self.d.screen()
        return int(s.width_in_pixels), int(s.height_in_pixels)
