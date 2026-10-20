"""Windows input backend: SendInput through ctypes, UI Automation for snapping (PRD 20.2).

Known limitation: Windows blocks injected input into apps running as
administrator (UIPI). A signed build with uiAccess="true" installed under
Program Files lifts it; until then GazeOS can optionally run elevated.
"""

from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes

from gazeos.os_backends.keys import WINDOWS_VK
from gazeos.types import Rect, UIElement

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
SM_CXSCREEN, SM_CYSCREEN = 0, 1

ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT

_BUTTONS = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}

_PASSWORD_HINTS = ("password", "passcode", "pin")


def _send(*inputs: INPUT) -> None:
    arr = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        log.debug("SendInput sent %d of %d (error %d)", sent, len(inputs), ctypes.get_last_error())


def _mouse(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> INPUT:
    inp = INPUT(type=INPUT_MOUSE)
    inp.mi = MOUSEINPUT(dx, dy, data & 0xFFFFFFFF, flags, 0, 0)
    return inp


def _key(vk: int = 0, scan: int = 0, flags: int = 0) -> INPUT:
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.ki = KEYBDINPUT(vk, scan, flags, 0, 0)
    return inp


class WindowsBackend:
    def __init__(self) -> None:
        from gazeos.control.mapping import make_dpi_aware

        make_dpi_aware()
        self._uia = None
        try:
            import uiautomation  # type: ignore[import-not-found]

            self._uia = uiautomation
        except ImportError:
            log.info("uiautomation not installed; target snapping disabled")

    # -- mouse ---------------------------------------------------------------
    def move(self, x: int, y: int) -> None:
        vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        vw = max(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN), 1)
        vh = max(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN), 1)
        nx = int(round((x - vx) * 65535 / (vw - 1)))
        ny = int(round((y - vy) * 65535 / (vh - 1)))
        _send(_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, nx, ny))

    def click(self, button: str = "left", count: int = 1) -> None:
        down, up = _BUTTONS[button]
        for _ in range(count):
            _send(_mouse(down), _mouse(up))
            if count > 1:
                time.sleep(0.05)

    def press(self, button: str) -> None:
        _send(_mouse(_BUTTONS[button][0]))

    def release(self, button: str) -> None:
        _send(_mouse(_BUTTONS[button][1]))

    def scroll(self, dx: int, dy: int) -> None:
        # Positive dy scrolls content down (wheel toward the user), like other backends.
        if dy:
            _send(_mouse(MOUSEEVENTF_WHEEL, data=-dy))
        if dx:
            _send(_mouse(MOUSEEVENTF_HWHEEL, data=dx))

    # -- keyboard ------------------------------------------------------------
    def type_text(self, text: str) -> None:
        """KEYEVENTF_UNICODE handles any character, including Hindi and Gujarati."""
        inputs = []
        for unit in self._utf16_units(text):
            inputs.append(_key(scan=unit, flags=KEYEVENTF_UNICODE))
            inputs.append(_key(scan=unit, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        if inputs:
            _send(*inputs)

    @staticmethod
    def _utf16_units(text: str) -> list[int]:
        data = text.replace("\n", "\r").encode("utf-16-le")
        return [int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2)]

    def _vk(self, key: str) -> int:
        k = key.lower()
        if k in WINDOWS_VK:
            return WINDOWS_VK[k]
        if len(k) == 1:
            res = user32.VkKeyScanW(ord(k))
            return int(res & 0xFF)
        raise ValueError(f"unknown key {key!r}")

    def hotkey(self, *keys: str) -> None:
        vks = [self._vk(k) for k in keys]
        downs = [_key(vk=v) for v in vks]
        ups = [_key(vk=v, flags=KEYEVENTF_KEYUP) for v in reversed(vks)]
        _send(*downs, *ups)

    # -- accessibility -------------------------------------------------------
    def _to_element(self, ctrl: object) -> UIElement | None:
        try:
            r = ctrl.BoundingRectangle  # type: ignore[attr-defined]
            rect = Rect(int(r.left), int(r.top), int(r.right - r.left), int(r.bottom - r.top))
            if rect.width <= 0 or rect.height <= 0:
                return None
            role = str(ctrl.ControlTypeName)  # type: ignore[attr-defined]
            name = str(ctrl.Name or "")  # type: ignore[attr-defined]
            invokable = role in {
                "ButtonControl", "HyperlinkControl", "MenuItemControl", "TabItemControl",
                "ListItemControl", "CheckBoxControl", "RadioButtonControl", "EditControl",
                "ComboBoxControl", "TreeItemControl", "SplitButtonControl",
            }
            is_password = bool(getattr(ctrl, "IsPassword", False)) or any(
                h in name.lower() for h in _PASSWORD_HINTS
            )
            return UIElement(rect=rect, role=role, name=name, invokable=invokable, is_password=is_password)
        except Exception:  # UIA calls can fail for transient elements
            return None

    def element_at(self, x: int, y: int) -> UIElement | None:
        if self._uia is None:
            return None
        try:
            ctrl = self._uia.ControlFromPoint(x, y)
        except Exception:
            return None
        return None if ctrl is None else self._to_element(ctrl)

    def elements_near(self, x: int, y: int, radius: int) -> list[UIElement]:
        """Sample a small grid of points; UIA has no cheap region query."""
        if self._uia is None:
            return []
        seen: dict[tuple[int, int, int, int], UIElement] = {}
        step = max(radius // 2, 8)
        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                e = self.element_at(x + dx, y + dy)
                if e is not None and e.invokable:
                    seen[(e.rect.left, e.rect.top, e.rect.width, e.rect.height)] = e
        return list(seen.values())

    def focused_is_password(self) -> bool:
        if self._uia is None:
            return False
        try:
            ctrl = self._uia.GetFocusedControl()
            return bool(ctrl is not None and getattr(ctrl, "IsPassword", False))
        except Exception:
            return False

    def screen_size(self) -> tuple[int, int]:
        return int(user32.GetSystemMetrics(SM_CXSCREEN)), int(user32.GetSystemMetrics(SM_CYSCREEN))


def screen_physical_size_mm() -> tuple[float, float] | None:
    """Physical size of the primary monitor from GDI, if the driver reports it."""
    gdi32 = ctypes.WinDLL("gdi32")  # type: ignore[attr-defined]
    hdc = user32.GetDC(0)
    try:
        HORZSIZE, VERTSIZE = 4, 6
        w, h = gdi32.GetDeviceCaps(hdc, HORZSIZE), gdi32.GetDeviceCaps(hdc, VERTSIZE)
        return (float(w), float(h)) if w > 0 and h > 0 else None
    finally:
        user32.ReleaseDC(0, hdc)
