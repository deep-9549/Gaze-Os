"""macOS input backend: Quartz CGEventPost and the AX API (PRD 20.3, stretch).

Needs Accessibility and Camera permission in System Settings > Privacy & Security.
"""

from __future__ import annotations

import time

import Quartz  # type: ignore[import-not-found]

from gazeos.types import Rect, UIElement

_MAC_KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "o": 31, "u": 32,
    "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "enter": 36, "tab": 48, "space": 49, "backspace": 51, "esc": 53, "delete": 117,
    "left": 123, "right": 124, "down": 125, "up": 126,
}
_MODIFIERS = {
    "cmd": Quartz.kCGEventFlagMaskCommand,
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "alt": Quartz.kCGEventFlagMaskAlternate,
    "shift": Quartz.kCGEventFlagMaskShift,
}
_BUTTONS = {
    "left": (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp, Quartz.kCGMouseButtonLeft),
    "right": (Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp, Quartz.kCGMouseButtonRight),
}


class MacBackend:
    def __init__(self) -> None:
        self._pos = (0, 0)
        self._dragging: str | None = None

    def _post(self, event: object) -> None:
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def move(self, x: int, y: int) -> None:
        self._pos = (x, y)
        kind = Quartz.kCGEventMouseMoved
        button = Quartz.kCGMouseButtonLeft
        if self._dragging:
            kind = Quartz.kCGEventLeftMouseDragged if self._dragging == "left" else Quartz.kCGEventRightMouseDragged
            button = _BUTTONS[self._dragging][2]
        self._post(Quartz.CGEventCreateMouseEvent(None, kind, (x, y), button))

    def click(self, button: str = "left", count: int = 1) -> None:
        down, up, btn = _BUTTONS[button]
        for i in range(1, count + 1):
            for kind in (down, up):
                ev = Quartz.CGEventCreateMouseEvent(None, kind, self._pos, btn)
                Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventClickState, i)
                self._post(ev)
            time.sleep(0.02)

    def press(self, button: str) -> None:
        down, _, btn = _BUTTONS[button]
        self._dragging = button
        self._post(Quartz.CGEventCreateMouseEvent(None, down, self._pos, btn))

    def release(self, button: str) -> None:
        _, up, btn = _BUTTONS[button]
        self._dragging = None
        self._post(Quartz.CGEventCreateMouseEvent(None, up, self._pos, btn))

    def scroll(self, dx: int, dy: int) -> None:
        ev = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitPixel, 2, int(-dy / 3), int(dx / 3)
        )
        self._post(ev)

    def type_text(self, text: str) -> None:
        for ch in text:
            for down in (True, False):
                ev = Quartz.CGEventCreateKeyboardEvent(None, 0, down)
                Quartz.CGEventKeyboardSetUnicodeString(ev, len(ch), ch)
                self._post(ev)

    def hotkey(self, *keys: str) -> None:
        flags = 0
        main = None
        for k in keys:
            if k.lower() in _MODIFIERS:
                flags |= _MODIFIERS[k.lower()]
            else:
                main = _MAC_KEYCODES[k.lower()]
        if main is None:
            return
        for down in (True, False):
            ev = Quartz.CGEventCreateKeyboardEvent(None, main, down)
            Quartz.CGEventSetFlags(ev, flags)
            self._post(ev)

    def element_at(self, x: int, y: int) -> UIElement | None:
        try:
            from ApplicationServices import (  # type: ignore[import-not-found]
                AXUIElementCopyAttributeValue,
                AXUIElementCopyElementAtPosition,
                AXUIElementCreateSystemWide,
                AXValueGetValue,
                kAXPositionAttribute,
                kAXRoleAttribute,
                kAXSizeAttribute,
                kAXValueCGPointType,
                kAXValueCGSizeType,
            )
        except ImportError:
            return None
        system = AXUIElementCreateSystemWide()
        err, el = AXUIElementCopyElementAtPosition(system, float(x), float(y), None)
        if err or el is None:
            return None
        _, pos = AXUIElementCopyAttributeValue(el, kAXPositionAttribute, None)
        _, size = AXUIElementCopyAttributeValue(el, kAXSizeAttribute, None)
        _, role = AXUIElementCopyAttributeValue(el, kAXRoleAttribute, None)
        if pos is None or size is None:
            return None
        _, p = AXValueGetValue(pos, kAXValueCGPointType, None)
        _, s = AXValueGetValue(size, kAXValueCGSizeType, None)
        role = str(role or "")
        return UIElement(
            rect=Rect(int(p.x), int(p.y), int(s.width), int(s.height)),
            role=role,
            invokable=role in {"AXButton", "AXLink", "AXMenuItem", "AXTextField", "AXCheckBox"},
            is_password=role == "AXSecureTextField",
        )

    def elements_near(self, x: int, y: int, radius: int) -> list[UIElement]:
        e = self.element_at(x, y)
        return [e] if e is not None else []

    def screen_size(self) -> tuple[int, int]:
        b = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return int(b.size.width), int(b.size.height)
