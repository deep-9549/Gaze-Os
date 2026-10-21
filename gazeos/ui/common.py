"""Shared UI plumbing: themes, window flags, thread bridge and gaze-target sync."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QPoint, QRect, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import QAbstractButton, QWidget

from gazeos.control.targets import GazeTarget, TargetRegistry
from gazeos.types import Rect, TrackingStatus

# Colours chosen for WCAG 2.1 AA contrast (>= 4.5:1 for text) on their backgrounds.
THEMES: dict[str, dict[str, str]] = {
    "default": {
        "bg": "#1E1F24", "panel": "#2A2C33", "fg": "#F2F3F5", "muted": "#B8BCC6",
        "accent": "#4DA3FF", "accent_fg": "#0B1220", "ok": "#3FCF8E", "warn": "#FFC24D",
        "bad": "#FF6B6B", "ring": "#4DA3FF", "cursor": "#4DA3FF",
    },
    "high_contrast": {
        "bg": "#000000", "panel": "#000000", "fg": "#FFFFFF", "muted": "#FFFFFF",
        "accent": "#FFFF00", "accent_fg": "#000000", "ok": "#00FF00", "warn": "#FFFF00",
        "bad": "#FF4040", "ring": "#FFFF00", "cursor": "#00FFFF",
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


class TargetSync(QObject):
    """Keeps the engine's target registry in sync with the visible GazeOS buttons,
    so every GazeOS screen is operable by gaze alone (PRD 7.7)."""

    def __init__(self, registry: TargetRegistry, bridge: MainThreadBridge, interval_ms: int = 300) -> None:
        super().__init__()
        self.registry = registry
        self.bridge = bridge
        self._windows: dict[str, QWidget] = {}
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.sync)
        self._timer.start(interval_ms)

    def track(self, group: str, window: QWidget) -> None:
        self._windows[group] = window
        self.sync()

    def untrack(self, group: str) -> None:
        self._windows.pop(group, None)
        self.registry.clear_group(group)

    @Slot()
    def sync(self) -> None:
        for group, win in list(self._windows.items()):
            self.registry.clear_group(group)
            if not win.isVisible():
                continue
            ratio = device_ratio(win)
            for i, btn in enumerate(win.findChildren(QAbstractButton)):
                if not btn.isVisible() or not btn.isEnabled():
                    continue
                top_left = btn.mapToGlobal(QPoint(0, 0))
                rect = to_physical(QRect(top_left, btn.size()), ratio)
                key = f"{group}:{btn.objectName() or i}:{id(btn)}"
                self.registry.register(GazeTarget(key, rect, self.bridge.later(btn.click), group))
