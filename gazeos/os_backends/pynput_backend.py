"""Cross-platform fallback backend using pynput (PRD 19)."""

from __future__ import annotations

from gazeos.types import UIElement


class PynputBackend:
    def __init__(self) -> None:
        from pynput import keyboard, mouse

        self._mouse = mouse.Controller()
        self._kb = keyboard.Controller()
        self._Button = mouse.Button
        self._Key = keyboard.Key

    def _button(self, name: str) -> object:
        return {"left": self._Button.left, "right": self._Button.right, "middle": self._Button.middle}[name]

    def move(self, x: int, y: int) -> None:
        self._mouse.position = (int(x), int(y))

    def click(self, button: str = "left", count: int = 1) -> None:
        self._mouse.click(self._button(button), count)

    def press(self, button: str) -> None:
        self._mouse.press(self._button(button))

    def release(self, button: str) -> None:
        self._mouse.release(self._button(button))

    def scroll(self, dx: int, dy: int) -> None:
        self._mouse.scroll(dx / 120.0, -dy / 120.0)

    def type_text(self, text: str) -> None:
        self._kb.type(text)

    def _key(self, name: str) -> object:
        alias = {"ctrl": "ctrl", "alt": "alt", "shift": "shift", "win": "cmd", "cmd": "cmd",
                 "enter": "enter", "esc": "esc", "backspace": "backspace", "pageup": "page_up",
                 "pagedown": "page_down"}
        n = alias.get(name.lower(), name.lower())
        return getattr(self._Key, n) if hasattr(self._Key, n) else n

    def hotkey(self, *keys: str) -> None:
        resolved = [self._key(k) for k in keys]
        for k in resolved:
            self._kb.press(k)
        for k in reversed(resolved):
            self._kb.release(k)

    def element_at(self, x: int, y: int) -> UIElement | None:
        return None

    def elements_near(self, x: int, y: int, radius: int) -> list[UIElement]:
        return []

    def screen_size(self) -> tuple[int, int]:
        try:
            from PySide6.QtGui import QGuiApplication

            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                g = screen.geometry()
                return g.width(), g.height()
        except Exception:
            pass
        return 1920, 1080
