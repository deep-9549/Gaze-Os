"""Settings screen generated from the pydantic schema (FR-36), operable by gaze alone.

Every value gets big buttons: on/off toggles, choice cyclers and -/+ steppers,
because spin boxes and sliders are hard to use with gaze.
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

from gazeos.settings import Settings
from gazeos.ui.common import stylesheet

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


class SettingsDialog(QWidget):
    def __init__(self, settings: Settings, on_save: Callable[[Settings], None], theme_name: str = "default") -> None:
        super().__init__(None, Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("GazeOS settings")
        self.setStyleSheet(stylesheet(theme_name))
        self.settings = settings.model_copy(deep=True)
        self.on_save = on_save
        root = QVBoxLayout(self)
        tabs = QTabWidget()
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
        save = QPushButton("Save")
        save.setObjectName("settings_save")
        save.clicked.connect(self._save)
        cancel = QPushButton("Close")
        cancel.setObjectName("settings_close")
        cancel.clicked.connect(self.close)
        buttons.addWidget(save)
        buttons.addWidget(cancel)
        root.addLayout(buttons)
        self.resize(1000, 720)

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
        self.on_save(Settings.model_validate(self.settings.model_dump()))
        self.close()
