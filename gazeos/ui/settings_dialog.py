"""Settings screen generated from the pydantic schema (FR-36), operable by gaze alone.

Every value gets big buttons: on/off toggles, choice cyclers and -/+ steppers,
because spin boxes and sliders are hard to use with gaze. Changes apply only on Save,
and Save asks again in a separate panel before keeping a change that could leave a
person who uses only their eyes stuck (`risky_changes`).
"""

from __future__ import annotations

import enum
import typing
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gazeos.settings import ControlMode, Settings
from gazeos.ui.panels import ConfirmPanel, GazePanel, global_rect

SETTINGS_IDLE_S = 60.0  # a form with unsaved changes: closes (without saving) only after a long pause
SAFE_DWELL_MS = (500.0, 1500.0)  # shorter: every glance clicks; longer: hard to hold still that long
MIN_CORNER_PX = 40  # smaller screen corners are hard to look into (menu; resume from pause)


def risky_changes(old: Settings, new: Settings) -> list[str]:
    """The changes from `old` to `new` that can leave a person who uses only their eyes unable to
    click, to see where they look, to resume, or to start GazeOS: Save asks before keeping them."""
    out: list[str] = []
    if old.dwell.enabled and not new.dwell.enabled:
        out.append("Dwell clicking off: looking at a button no longer presses it")
    if new.mode != old.mode:
        need = ": clicks need a physical switch" if new.mode == ControlMode.HYBRID else ""
        out.append(f"Control mode {new.mode.value}{need}")
    lo, hi = SAFE_DWELL_MS
    if new.dwell.dwell_ms != old.dwell.dwell_ms and not lo <= new.dwell.dwell_ms <= hi:
        why = "every glance clicks" if new.dwell.dwell_ms < lo else "hard to hold the eyes still that long"
        out.append(f"Dwell time {new.dwell.dwell_ms:g} ms: {why}")
    camera = ("device_index", "width", "height", "fps")
    if any(getattr(new.camera, f) != getattr(old.camera, f) for f in camera):
        out.append("Camera changed: if it does not work, tracking will not start next time")
    if old.system.start_with_os and not new.system.start_with_os:
        out.append("GazeOS will no longer start with the computer")
    if old.system.show_overlay and not new.system.show_overlay:
        out.append("No gaze cursor or status from the next start")
    if new.system.corner_size_px < min(MIN_CORNER_PX, old.system.corner_size_px):
        out.append("Screen corners too small to open the menu or resume from pause by gaze")
    if old.system.auto_resume_after_face_lost and not new.system.auto_resume_after_face_lost:
        out.append("Tracking stays paused after your face is lost")
    if old.system.confirm_destructive_shortcuts and not new.system.confirm_destructive_shortcuts:
        out.append("Close window and Close tab no longer ask first")
    if old.calibration.auto_start_s > 0 >= new.calibration.auto_start_s:
        out.append("Calibration waits for a Start button that gaze cannot press")
    if not old.gestures.enabled and new.gestures.enabled:
        out.append("Face gestures click: opening the mouth starts a drag")
    return out

# Sections and fields shown on the screen; others are edited in settings.json.
SECTIONS = {
    "General": ("mode", "theme"),
    "Dwell": ("dwell",),
    "Blinks": ("blink",),
    "Gestures": ("gestures",),
    "Cursor": ("filter", "fixation", "snap"),
    "Scrolling": ("scroll",),
    "Typing": ("typing",),
    "Calibration": ("calibration",),
    "System": ("system",),
    "Camera": ("camera",),
    "Privacy": ("privacy",),
}
SKIP = {"mapping", "intrinsics_file", "piper_voice", "llm_model", "emergency_hotkey", "pause_hotkey", "camera_offset_mm"}


def _choices(annotation: Any) -> list[Any] | None:
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return list(typing.get_args(annotation))
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return list(annotation)
    return None


def _step(value: float, meta: list[Any]) -> float:
    lo = hi = None
    for m in meta:
        lo = getattr(m, "ge", None) or getattr(m, "gt", None) or lo
        hi = getattr(m, "le", None) or getattr(m, "lt", None) or hi
    if lo is not None and hi is not None:
        return (float(hi) - float(lo)) / 20.0
    return abs(value) * 0.1 if value else 0.1


class SettingsDialog(GazePanel):
    """Edits a copy of `settings`; Save hands it to `on_save`. `present` shows the confirmation
    that Save asks for risky changes (the app also makes it gaze-operable)."""

    def __init__(self, settings: Settings, on_save: Callable[[Settings], None], theme_name: str = "default",
                 present: Callable[[ConfirmPanel], None] | None = None, idle_s: float = SETTINGS_IDLE_S) -> None:
        super().__init__(Qt.WindowType.WindowStaysOnTopHint, theme_name, idle_s)
        self.setWindowTitle("GazeOS settings")
        self.original = settings
        self.settings = settings.model_copy(deep=True)
        self.on_save = on_save
        self.present: Callable[[ConfirmPanel], None] = present or ConfirmPanel.show
        self.confirm: ConfirmPanel | None = None
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.currentChanged.connect(lambda _index: self.touch())
        root.addWidget(tabs)
        for title, names in SECTIONS.items():
            page = QWidget()
            grid = QGridLayout(page)
            row = 0
            for name in names:
                value = getattr(self.settings, name)
                if isinstance(value, BaseModel):
                    for fname, finfo in type(value).model_fields.items():
                        if fname in SKIP:
                            continue
                        self._add_row(grid, row, value, fname, finfo.annotation, list(finfo.metadata))
                        row += 1
                else:
                    finfo = Settings.model_fields[name]
                    self._add_row(grid, row, self.settings, name, finfo.annotation, list(finfo.metadata))
                    row += 1
            grid.setRowStretch(row, 1)
            scroll = QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            tabs.addTab(scroll, title)
        buttons = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("settings_save")
        self.save_btn.clicked.connect(self._save)
        cancel = QPushButton("Close")
        cancel.setObjectName("settings_close")
        cancel.clicked.connect(self.close)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(cancel)
        root.addLayout(buttons)
        self.resize(1000, 720)
        self.watch_presses()

    def _label(self, name: str) -> str:
        return name.replace("_", " ").replace(" ms", " (ms)").capitalize()

    def _add_row(self, grid: QGridLayout, row: int, model: BaseModel, name: str, ann: Any, meta: list[Any]) -> None:
        value = getattr(model, name)
        grid.addWidget(QLabel(self._label(name)), row, 0)
        value_label = QLabel(self._fmt(value))
        value_label.setMinimumWidth(160)
        grid.addWidget(value_label, row, 1)
        choices = _choices(ann)

        def setter(v: Any) -> None:
            try:
                setattr(model, name, v)
                value_label.setText(self._fmt(getattr(model, name)))
            except Exception:
                value_label.setText(self._fmt(getattr(model, name)) + "  (out of range)")

        if isinstance(value, bool):
            b = QPushButton("Toggle")
            b.setObjectName(f"set_{name}")
            b.clicked.connect(lambda: setter(not getattr(model, name)))
            grid.addWidget(b, row, 2)
        elif choices:
            b = QPushButton("Next option")
            b.setObjectName(f"set_{name}")
            b.clicked.connect(lambda: setter(choices[(choices.index(getattr(model, name)) + 1) % len(choices)]))
            grid.addWidget(b, row, 2)
        elif isinstance(value, (int, float)):
            step = _step(float(value), meta)
            is_int = isinstance(value, int)
            for col, sign, text in ((2, -1, "−"), (3, 1, "+")):
                b = QPushButton(text)
                b.setObjectName(f"set_{name}_{'dec' if sign < 0 else 'inc'}")
                b.setMinimumWidth(80)

                def bump(_: bool = False, s: int = sign) -> None:
                    cur = getattr(model, name)
                    nv = cur + s * (max(1, round(step)) if is_int else step)
                    setter(int(nv) if is_int else round(float(nv), 4))

                b.clicked.connect(bump)
                grid.addWidget(b, row, col)

    @staticmethod
    def _fmt(v: Any) -> str:
        if isinstance(v, enum.Enum):
            return str(v.value)
        if isinstance(v, bool):
            return "On" if v else "Off"
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    def _save(self) -> None:
        if not self.isVisible():
            return  # a press that arrives after the dialog closed (a stale gaze target)
        new = Settings.model_validate(self.settings.model_dump())
        risky = risky_changes(self.original, new)
        if not risky:
            self._commit(new)
            return
        # Ask in a separate panel with Yes away from Save. The dialog hides meanwhile, edits
        # kept, and comes back on No (or no answer).
        avoid = global_rect(self.save_btn)
        self.hide()
        self.confirm = ConfirmPanel(
            "Save changes that can make GazeOS hard to use by gaze?", lambda: self._commit(new), self.theme_name,
            yes_label="Yes, save them", detail="\n".join(f"• {r}" for r in risky), on_no=self.show, avoid=avoid)
        self.present(self.confirm)

    def _commit(self, new: Settings) -> None:
        self.on_save(new)
        self.close()
