"""Shared UI plumbing: themes, window flags, thread bridge and gaze-target sync."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import QAbstractButton, QWidget

from gazeos.control.targets import GazeTarget, TargetRegistry, covered_fraction
from gazeos.types import Rect, TrackingStatus

# Colours chosen for WCAG 2.1 AA contrast (>= 4.5:1 for text) on their backgrounds.
THEMES: dict[str, dict[str, str]] = {
    "default": {
        "bg": "#1E1F24", "panel": "#2A2C33", "fg": "#F2F3F5", "muted": "#B8BCC6",
        "accent": "#4DA3FF", "accent_fg": "#0B1220", "ok": "#3FCF8E", "warn": "#FFC24D",
        "bad": "#FF6B6B", "ring": "#4DA3FF", "cursor": "#4DA3FF",
        # Calibration runs on the same dark surface as the keyboard: the screen lights the face,
        # and a fit made under a light screen was 6 mm worse on the dark keyboard than one made
        # under a dark screen (18 Nov 2026 captures, 12 against 18 mm). Overrides ADR 0006's
        # light surface, chosen before the keyboard was in daily use.
        "calib_bg": "#2A2C33", "calib_fg": "#F2F3F5", "calib_dot": "#4DA3FF", "calib_check": "#3FCF8E",
    },
    "high_contrast": {
        "bg": "#000000", "panel": "#000000", "fg": "#FFFFFF", "muted": "#FFFFFF",
        "accent": "#FFFF00", "accent_fg": "#000000", "ok": "#00FF00", "warn": "#FFFF00",
        "bad": "#FF4040", "ring": "#FFFF00", "cursor": "#00FFFF",
        "calib_bg": "#FFFFFF", "calib_fg": "#000000", "calib_dot": "#000000", "calib_check": "#0000C0",
    },
}
THEMES["large_text"] = THEMES["default"]

STATUS_COLORS = {
    TrackingStatus.TRACKING: "ok",
    TrackingStatus.PAUSED: "muted",
    TrackingStatus.NO_FACE: "bad",
    TrackingStatus.LOW_LIGHT: "warn",
    TrackingStatus.MOVE_CLOSER: "warn",
    TrackingStatus.MULTIPLE_FACES: "warn",
}


def theme(name: str) -> dict[str, str]:
    return THEMES.get(name, THEMES["default"])


def font_scale(name: str) -> float:
    return 1.4 if name == "large_text" else 1.0


def stylesheet(name: str) -> str:
    t = theme(name)
    pt = int(16 * font_scale(name))
    return f"""
    QWidget {{ background: {t['bg']}; color: {t['fg']}; font-size: {pt}pt; }}
    QPushButton {{ background: {t['panel']}; color: {t['fg']}; border: 2px solid {t['muted']};
                   border-radius: 10px; padding: 10px 16px; min-height: 48px; }}
    QPushButton:checked, QPushButton[active="true"] {{ background: {t['accent']}; color: {t['accent_fg']};
                   border-color: {t['accent']}; }}
    QPushButton:hover {{ border-color: {t['accent']}; }}
    QLabel[role="title"] {{ font-size: {int(pt * 1.6)}pt; font-weight: 600; }}
    QLabel[role="muted"] {{ color: {t['muted']}; }}
    QSlider::groove:horizontal {{ height: 14px; background: {t['panel']}; border-radius: 7px; }}
    QSlider::handle:horizontal {{ width: 40px; margin: -14px 0; background: {t['accent']}; border-radius: 20px; }}
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {t['panel']}; min-height: 44px;
                   border: 2px solid {t['muted']}; border-radius: 8px; padding: 4px 10px; }}
    QCheckBox::indicator {{ width: 32px; height: 32px; }}
    """


def color(name: str, key: str, alpha: int = 255) -> QColor:
    c = QColor(theme(name)[key])
    c.setAlpha(alpha)
    return c


def overlay_flags() -> Qt.WindowType:
    """Stay on top, frameless, tool window; transparent to the mouse (PRD 20.2)."""
    return (
        Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.Tool
        | Qt.WindowType.WindowTransparentForInput
        | Qt.WindowType.WindowDoesNotAcceptFocus
    )


def panel_flags() -> Qt.WindowType:
    """Windows that show buttons but must never take focus (keyboard, action bar)."""
    return (
        Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.Tool
        | Qt.WindowType.WindowDoesNotAcceptFocus
    )


def apply_no_activate(widget: QWidget, click_through: bool = False) -> None:
    """Windows extended styles so the window never steals focus (and optionally clicks)."""
    widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
    if sys.platform != "win32":
        return
    import ctypes

    GWL_EXSTYLE = -20
    WS_EX_LAYERED, WS_EX_TRANSPARENT = 0x00080000, 0x00000020
    WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST = 0x08000000, 0x00000080, 0x00000008
    hwnd = int(widget.winId())
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST
    if click_through:
        style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)


def bring_to_top(widget: QWidget) -> None:
    """Put an always-on-top window above the other always-on-top windows without
    activating it (a shown or clicked topmost window otherwise covers it)."""
    if sys.platform != "win32":
        widget.raise_()
        return
    import ctypes
    from ctypes import wintypes

    set_pos = ctypes.windll.user32.SetWindowPos  # type: ignore[attr-defined]
    # Without argtypes ctypes passes HWND_TOPMOST (-1) as a 32-bit int, which 64-bit
    # Windows does not read as -1, and the call silently does nothing.
    set_pos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        wintypes.UINT]
    HWND_TOPMOST = wintypes.HWND(-1)
    SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
    set_pos(int(widget.winId()), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)


def device_ratio(widget: QWidget | None = None) -> float:
    screen = widget.screen() if widget is not None else QGuiApplication.primaryScreen()
    return float(screen.devicePixelRatio()) if screen is not None else 1.0


def to_physical(rect: QRect, ratio: float) -> Rect:
    """Qt logical pixels -> physical screen pixels used by the engine."""
    return Rect(int(rect.x() * ratio), int(rect.y() * ratio), int(rect.width() * ratio), int(rect.height() * ratio))


def to_logical(x: float, y: float, ratio: float) -> QPoint:
    return QPoint(int(x / ratio), int(y / ratio))


class MainThreadBridge(QObject):
    """Runs callables on the GUI thread; the engine calls targets from its own thread."""

    call = Signal(object)
    # One packet object per frame: PySide6 can corrupt None's refcount when a bare
    # None is passed through a multi-argument Signal(object, ...).
    frame = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def _run(self, fn: Callable[[], Any]) -> None:
        fn()

    def later(self, fn: Callable[[], Any]) -> Callable[[], None]:
        return lambda: self.call.emit(fn)


# Which GazeOS window is in front where two overlap (higher = in front): the side bar is behind
# everything, the keyboard in front of it, panels in front of the keyboard, questions on top.
TARGET_LAYERS = {"action_bar": 0, "keyboard": 10, "debug": 20, "menu": 30, "shortcuts": 30, "settings": 40,
                 "privacy": 40, "onboarding": 40, "calibration": 50, "confirm": 60}
DEFAULT_LAYER = 20
# A button this much under a window in front of it is hidden: not a target.
HIDDEN_FRACTION = 0.25


class TargetSync(QObject):
    """Keeps the engine's target registry in sync with the visible GazeOS buttons,
    so every GazeOS screen is operable by gaze alone (PRD 7.7).

    Only what can be seen can be pressed: a button scrolled out of view (Settings) is no
    target, nor one under another GazeOS window in front of it (the side bar's buttons under
    the keyboard took presses meant for its keys, 18 Nov 2026), and a window's targets go the
    moment it hides, not at the next sync."""

    def __init__(self, registry: TargetRegistry, bridge: MainThreadBridge, interval_ms: int = 300) -> None:
        super().__init__()
        self.registry = registry
        self.bridge = bridge
        self._windows: dict[str, QWidget] = {}
        # Windows at a screen edge whose buttons respond this many px further toward the screen's
        # middle (physical px, by group). Webcam gaze is least accurate at the edges and the eye
        # cannot look past the edge to pull the cursor over: on 20 Nov the cursor sat 100-150 px
        # left of the gaze at the right-hand bar, and Menu could not be reached.
        self.reach: dict[str, float] = {}
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.sync)
        self._timer.start(interval_ms)

    def track(self, group: str, window: QWidget) -> None:
        old = self._windows.get(group)
        if old is not None and old is not window:
            old.removeEventFilter(self)
        self._windows[group] = window
        window.installEventFilter(self)
        self.sync()

    def untrack(self, group: str) -> None:
        window = self._windows.pop(group, None)
        if window is not None:
            window.removeEventFilter(self)
        self.registry.clear_group(group)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Hide:
            for group, win in self._windows.items():
                if win is watched:
                    self.registry.clear_group(group)
        return False

    @Slot()
    def sync(self) -> None:
        frames = {g: (TARGET_LAYERS.get(g, DEFAULT_LAYER), to_physical(w.frameGeometry(), device_ratio(w)))
                  for g, w in self._windows.items() if w.isVisible()}
        for group, win in list(self._windows.items()):
            self.registry.clear_group(group)
            if not win.isVisible():
                continue
            layer = TARGET_LAYERS.get(group, DEFAULT_LAYER)
            in_front = [r for g, (lay, r) in frames.items() if g != group and lay > layer]
            ratio = device_ratio(win)
            for i, btn in enumerate(win.findChildren(QAbstractButton)):
                if not btn.isVisible() or not btn.isEnabled():
                    continue
                seen = btn.visibleRegion().boundingRect()  # the part not clipped by a scroll area
                if seen.isEmpty():
                    continue
                top_left = btn.mapToGlobal(seen.topLeft())
                rect = to_physical(QRect(top_left, seen.size()), ratio)
                if any(covered_fraction(rect, f) > HIDDEN_FRACTION for f in in_front):
                    continue
                if group in self.reach:
                    rect = _reach_inward(rect, frames[group][1], win, self.reach[group])
                key = f"{group}:{btn.objectName() or i}:{id(btn)}"
                self.registry.register(GazeTarget(key, rect, self.bridge.later(btn.click), group,
                                                  repeat=bool(btn.property("gaze_repeat")),
                                                  dwell_factor=float(btn.property("gaze_dwell_factor") or 1.0),
                                                  layer=layer))


def _reach_inward(rect: Rect, frame: Rect, window: QWidget, reach: float) -> Rect:
    """`rect` grown by `reach` px on the side facing the screen's middle, when `frame` (the
    window, physical px) sits at the left, right or bottom edge of its screen."""
    screen = window.screen() or QGuiApplication.primaryScreen()
    if screen is None or reach <= 0:
        return rect
    s = to_physical(screen.geometry(), device_ratio(window))
    r = int(round(reach))
    tol = 24
    if frame.left + frame.width >= s.left + s.width - tol:  # docked right: grow to the left
        return Rect(rect.left - r, rect.top, rect.width + r, rect.height)
    if frame.left <= s.left + tol:  # docked left: grow to the right
        return Rect(rect.left, rect.top, rect.width + r, rect.height)
    if frame.top + frame.height >= s.top + s.height - tol:  # docked at the bottom: grow upward
        return Rect(rect.left, rect.top - r, rect.width, rect.height + r)
    return rect
