"""Calibration screen: 9 or 16 shrinking targets, then validation and a score (PRD 11.2, FR-09-FR-11).

Observations arrive from the processing thread through `add_frame`; the window
only decides which target is showing and when frames count.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.calibration.personalization import Personalizer
from gazeos.calibration.protocol import (
    CalibrationSample,
    GazeObservation,
    ValidationResult,
    filter_point_samples,
    grid_targets,
    in_sample_window,
    quick_targets,
    shuffled,
    target_radius,
    validate,
    validation_targets,
)
from gazeos.settings import CalibrationSettings
from gazeos.ui.common import color, device_ratio, stylesheet


@dataclass
class CalibrationOutcome:
    samples: list[CalibrationSample]
    personalizer: Personalizer
    validation: ValidationResult
    quick_offset: np.ndarray | None = None


class CalibrationWindow(QWidget):
    def __init__(
        self,
        geometry: ScreenGeometry,
        personalizer: Personalizer,
        settings: CalibrationSettings,
        on_done: Callable[[CalibrationOutcome | None], None],
        theme_name: str = "default",
        quick: bool = False,
        current_predict: Callable[[GazeObservation], np.ndarray] | None = None,
    ) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.geometry_ = geometry
        self.personalizer = personalizer
        self.s = settings
        self.on_done = on_done
        self.theme_name = theme_name
        self.quick = quick
        self.current_predict = current_predict
        self.setStyleSheet(stylesheet(theme_name))
        self._lock = threading.Lock()
        if quick:
            self.targets = quick_targets(geometry)
            self.validation = []
        else:
            self.targets = shuffled(grid_targets(geometry, settings.num_points))
            self.validation = validation_targets(geometry)
        self.phase = "intro"
        self.index = -1
        self.shown_at = 0.0
        self.point_samples: list[CalibrationSample] = []
        self.samples: list[CalibrationSample] = []
        self.val_obs: list[list[GazeObservation]] = []
        self.result: ValidationResult | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._build_buttons()

    def _build_buttons(self) -> None:
        self.buttons = QWidget(self)
        lay = QHBoxLayout(self.buttons)
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("calib_start")
        self.accept_btn = QPushButton("Accept")
        self.accept_btn.setObjectName("calib_accept")
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setObjectName("calib_redo")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("calib_cancel")
        for b in (self.start_btn, self.accept_btn, self.redo_btn, self.cancel_btn):
            b.setMinimumSize(200, 90)
            lay.addWidget(b)
        self.start_btn.clicked.connect(self.start)
        self.accept_btn.clicked.connect(self._accept)
        self.redo_btn.clicked.connect(self.restart)
        self.cancel_btn.clicked.connect(lambda: self._finish(None))
        self.accept_btn.hide()
        self.redo_btn.hide()
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        outer.addWidget(self.buttons, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)

    # -- flow ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            self.phase = "calibrate"
            self.index = -1
            self.samples.clear()
            self.val_obs = [[] for _ in self.validation]
        self.buttons.hide()
        self._next()
        self._timer.start(16)

    def restart(self) -> None:
        self.result = None
        self.accept_btn.hide()
        self.redo_btn.hide()
        if not self.quick:
            self.targets = shuffled(self.targets)
        self.start()

    def _points(self) -> list[tuple[float, float]]:
        return self.targets if self.phase == "calibrate" else self.validation

    def _next(self) -> None:
        with self._lock:
            if self.phase == "calibrate" and self.index >= 0:
                self.samples.extend(filter_point_samples(self.point_samples, self.s.mad_threshold))
            self.point_samples = []
            self.index += 1
            if self.index >= len(self._points()):
                if self.phase == "calibrate" and self.validation:
                    self.phase, self.index = "validate", 0
                else:
                    self.phase = "result"
            self.shown_at = time.monotonic()
        if self.phase == "result":
            self._timer.stop()
            self._compute_result()

    def _tick(self) -> None:
        if time.monotonic() - self.shown_at >= self.s.target_duration_s:
            self._next()
        self.update()

    def add_frame(self, t: float, obs: GazeObservation | None, eyes_closed: bool) -> None:
        """Called from the processing thread for every frame."""
        if obs is None or eyes_closed:
            return  # blink frames are dropped (PRD 11.2 step 4)
        with self._lock:
            if self.phase not in ("calibrate", "validate") or self.index < 0:
                return
            since = t - self.shown_at
            if not in_sample_window(since, self.s.keep_from_s, self.s.target_duration_s):
                return
            target = self._points()[self.index]
            if self.phase == "calibrate":
                self.point_samples.append(CalibrationSample(obs, np.asarray(target, dtype=np.float64), t=t))
            else:
                self.val_obs[self.index].append(obs)

    def _compute_result(self) -> None:
        if self.quick:
            quick = [self.current_predict(s.obs) for s in self.samples] if self.current_predict else []
            if not quick:
                self._finish(None)
                return
            med = np.median(np.stack(quick), axis=0)
            offset = np.asarray(self.targets[0], dtype=np.float64) - med
            # Report the drift that was corrected.
            res = validate([(float(med[0]), float(med[1]))], [self.targets[0]], self.geometry_)
            self._finish(CalibrationOutcome(self.samples, self.personalizer, res, quick_offset=offset))
            return
        if len(self.samples) < 9:
            self.result = None
            self.phase = "failed"
            self.redo_btn.show()
            self.buttons.show()
            self.start_btn.hide()
            self.update()
            return
        self.personalizer.fit(self.samples)
        preds: list[tuple[float, float]] = []
        targets: list[tuple[float, float]] = []
        for obs_list, target in zip(self.val_obs, self.validation, strict=True):
            if obs_list:
                p = np.median(np.stack([self.personalizer.predict_px(o) for o in obs_list]), axis=0)
                preds.append((float(p[0]), float(p[1])))
                targets.append(target)
        self.result = validate(preds, targets, self.geometry_) if preds else None
        self.start_btn.hide()
        self.accept_btn.show()
        self.redo_btn.show()
        self.buttons.show()
        self.update()

    def _accept(self) -> None:
        if self.result is not None:
            self._finish(CalibrationOutcome(self.samples, self.personalizer, self.result))

    def _finish(self, outcome: CalibrationOutcome | None) -> None:
        self._timer.stop()
        self.phase = "done"
        self.close()
        self.on_done(outcome)

    # -- drawing ---------------------------------------------------------------
    def paintEvent(self, event: object) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), color(self.theme_name, "bg"))
        ratio = device_ratio(self)
        font = QFont()
        font.setPointSize(20)
        p.setFont(font)
        p.setPen(color(self.theme_name, "fg"))
        top = QRectF(0, 40, self.width(), 120)
        if self.phase == "intro":
            msg = ("Look at the centre dot." if self.quick else
                   f"Follow each dot with your eyes until it disappears ({len(self.targets)} dots, about "
                   f"{int(len(self.targets) * self.s.target_duration_s + 6)} s). Keep your head still.")
            p.drawText(top, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, msg)
        elif self.phase in ("calibrate", "validate"):
            pts = self._points()
            if 0 <= self.index < len(pts):
                x, y = pts[self.index]
                since = time.monotonic() - self.shown_at
                radius = target_radius(since, self.s.target_duration_s)
                c = QPointF(x / ratio, y / ratio)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(color(self.theme_name, "accent" if self.phase == "calibrate" else "ok")))
                p.drawEllipse(c, radius, radius)
                p.setBrush(QBrush(color(self.theme_name, "bg")))
                p.drawEllipse(c, 3, 3)
            label = "Calibrating" if self.phase == "calibrate" else "Checking accuracy"
            p.setPen(color(self.theme_name, "muted"))
            p.drawText(QRectF(0, self.height() - 60, self.width(), 40), Qt.AlignmentFlag.AlignCenter,
                       f"{label} · {self.index + 1} / {len(pts)}")
        elif self.phase == "result" and self.result is not None:
            r = self.result
            p.drawText(top, Qt.AlignmentFlag.AlignCenter,
                       f"Score {r.score} / 100   ·   mean error {r.mean_deg:.1f}° ({r.mean_cm:.1f} cm)")
            for (tx, ty), err in zip(self.validation, r.per_point_deg, strict=False):
                c = QPointF(tx / ratio, ty / ratio)
                key = "ok" if err <= 2.0 else "warn" if err <= 3.5 else "bad"
                p.setPen(QPen(color(self.theme_name, key), 3))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(c, max(err * 12, 8), max(err * 12, 8))
                p.drawText(QRectF(c.x() - 60, c.y() + 30, 120, 30), Qt.AlignmentFlag.AlignCenter, f"{err:.1f}°")
        elif self.phase == "failed":
            p.drawText(top, Qt.AlignmentFlag.AlignCenter,
                       "Not enough eye data was captured. Check the lighting and that your face is visible, then redo.")
        p.end()
