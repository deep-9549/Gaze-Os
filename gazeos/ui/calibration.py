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
    region_name,
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
        keep_previous: bool = False,
    ) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.geometry_ = geometry
        self.personalizer = personalizer
        self.s = settings
        self.on_done = on_done
        self.theme_name = theme_name
        self.quick = quick
        self.current_predict = current_predict
        # A calibration is already saved: a poor new one is not kept by itself (Accept still can).
        self.keep_previous = keep_previous
        self.setStyleSheet(stylesheet(theme_name))
        self._lock = threading.Lock()
        if quick:
            self.targets = quick_targets(geometry)
            self.validation = []
        else:
            self.targets = shuffled(grid_targets(geometry, settings.num_points))
            self.validation = validation_targets(geometry)
        self._base_targets, self._base_validation = list(self.targets), list(self.validation)
        self.phase = "intro"
        self.index = -1
        self.shown_at = 0.0
        self.point_samples: list[CalibrationSample] = []
        self.samples: list[CalibrationSample] = []
        self.val_obs: list[list[GazeObservation]] = []
        self.result: ValidationResult | None = None
        # Dots short of usable samples are shown once more at the end; these failed twice.
        self._retried: set[tuple[str, tuple[float, float]]] = set()
        self.failed: list[tuple[str, tuple[float, float]]] = []
        self.fail_message = ""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        # The step that happens by itself when nobody presses a button (settings: auto_*).
        self._auto_at: float | None = None
        self._auto_step: Callable[[], None] | None = None
        self._auto_text = ""
        self._auto_redos = 0
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._auto_tick)
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
        self.redo_btn.clicked.connect(self._manual_redo)
        self.cancel_btn.clicked.connect(lambda: self._finish(None))
        self.accept_btn.hide()
        self.redo_btn.hide()
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        outer.addWidget(self.buttons, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)

    # -- steps without a button ------------------------------------------------
    def showEvent(self, event: object) -> None:  # noqa: N802
        if self.phase == "intro" and self.s.auto_start_s > 0:
            self._arm(self.s.auto_start_s, self.start, "Starting")

    def _arm(self, seconds: float, step: Callable[[], None], text: str) -> None:
        self._auto_at, self._auto_step, self._auto_text = time.monotonic() + seconds, step, text
        self._auto_timer.start(200)

    def _disarm(self) -> None:
        self._auto_at = self._auto_step = None
        self._auto_timer.stop()

    def _auto_tick(self) -> None:
        if self._auto_at is not None and time.monotonic() >= self._auto_at and self._auto_step is not None:
            step = self._auto_step
            self._disarm()
            step()
        self.update()

    def _auto_redo(self) -> None:
        self._auto_redos += 1
        self.restart()

    def _manual_redo(self) -> None:
        self._auto_redos = 0  # a person is here: count again from the start
        self.restart()

    def _arm_after_run(self, good: bool) -> None:
        """What happens by itself after a run: accept a good one; repeat a poor or failed one
        a few times; then keep what there is (a poor fit, when nothing better is saved) or close,
        so that a person who cannot press a button is never left on this screen and a stray
        glance at "Calibrate" cannot replace a good calibration with a poor one."""
        d = self.s.auto_decide_s
        if good:
            self._arm(d, self._accept, "Accepting")
        elif self._auto_redos < self.s.auto_redo_limit:
            self._arm(d + 2.0, self._auto_redo, "Trying again")
        elif self.result is not None and not self.keep_previous:
            self._arm(d + 2.0, self._accept, "Keeping this result")
        else:
            self._arm(d + 4.0, lambda: self._finish(None), "Closing (the earlier calibration is kept)")

    # -- flow ------------------------------------------------------------------
    def start(self) -> None:
        self._disarm()
        with self._lock:
            self.phase = "calibrate"
            self.index = -1
            self.samples.clear()
            self.targets, self.validation = list(self._base_targets), list(self._base_validation)
            self.val_obs = [[] for _ in self.validation]
            self._retried.clear()
            self.failed.clear()
        self.buttons.hide()
        self._next()
        self._timer.start(16)

    def restart(self) -> None:
        self.result = None
        self.accept_btn.hide()
        self.redo_btn.hide()
        if not self.quick:
            self._base_targets = shuffled(self._base_targets)
        self.start()

    def _points(self) -> list[tuple[float, float]]:
        return self.targets if self.phase == "calibrate" else self.validation

    def _next(self) -> None:
        with self._lock:
            if self.phase in ("calibrate", "validate") and self.index >= 0:
                self._close_point()
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

    def _close_point(self) -> None:
        """Keep the finished dot's samples, or queue it once more (caller holds the lock)."""
        target = self._points()[self.index]
        if self.phase == "calibrate":
            kept = filter_point_samples(self.point_samples, self.s.mad_threshold)
            if len(kept) >= self.s.min_samples_per_point:
                self.samples.extend(kept)
                return
        elif len(self.val_obs[self.index]) >= self.s.min_samples_per_point:
            return
        else:
            self.val_obs[self.index] = []  # too few to score; the retry (if any) gets its own list
        if (self.phase, target) in self._retried:
            self.failed.append((self.phase, target))
            return
        self._retried.add((self.phase, target))
        self._points().append(target)
        if self.phase == "validate":
            self.val_obs.append([])

    def _regions(self, phase: str) -> list[str]:
        return [region_name(t, self.geometry_) for ph, t in self.failed if ph == phase]

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
        missing = self._regions("calibrate")
        if missing or len(self.samples) < 9:
            self.result = None
            self.fail_message = (
                f"No usable eye data at the {', '.join(missing)} of the screen. If your eyes were open, "
                "the lids may cover them when you look there: raise the screen or sit a little higher, "
                "keep a light in front of you, then press Redo."
                if missing else
                "Not enough eye data was captured. Check the lighting and that your face is visible, then redo.")
            self.phase = "failed"
            self.redo_btn.show()
            self.buttons.show()
            self.start_btn.hide()
            self._arm_after_run(good=False)
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
        self._arm_after_run(good=self.result is not None and self.result.score >= self.s.auto_accept_score)
        self.update()

    def _accept(self) -> None:
        if self.result is not None:
            self._finish(CalibrationOutcome(self.samples, self.personalizer, self.result))

    def _finish(self, outcome: CalibrationOutcome | None) -> None:
        self._disarm()
        self._timer.stop()
        self.phase = "done"
        self.close()
        self.on_done(outcome)

    # -- drawing ---------------------------------------------------------------
    def paintEvent(self, event: object) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        light = self.phase in ("intro", "calibrate", "validate")  # the calibration surface (theme calib_*)
        bg_key = "calib_bg" if light else "bg"
        p.fillRect(self.rect(), color(self.theme_name, bg_key))
        ratio = device_ratio(self)
        font = QFont()
        font.setPointSize(20)
        p.setFont(font)
        p.setPen(color(self.theme_name, "calib_fg" if light else "fg"))
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
                p.setBrush(QBrush(color(self.theme_name, "calib_dot" if self.phase == "calibrate" else "calib_check")))
                p.drawEllipse(c, radius, radius)
                p.setBrush(QBrush(color(self.theme_name, bg_key)))
                p.drawEllipse(c, 3, 3)
            label = "Calibrating" if self.phase == "calibrate" else "Checking accuracy"
            p.setPen(color(self.theme_name, "calib_fg"))
            p.drawText(QRectF(0, self.height() - 60, self.width(), 40), Qt.AlignmentFlag.AlignCenter,
                       f"{label} · {self.index + 1} / {len(pts)}")
        elif self.phase == "result" and self.result is not None:
            r = self.result
            p.drawText(top, Qt.AlignmentFlag.AlignCenter,
                       f"Score {r.score} / 100   ·   mean error {r.mean_deg:.1f}° ({r.mean_cm:.1f} cm)")
            if self._regions("validate"):
                p.drawText(QRectF(0, 130, self.width(), 40), Qt.AlignmentFlag.AlignCenter,
                           f"Not checked (no eye data): {', '.join(self._regions('validate'))}")
            for (tx, ty), err in zip(self.validation, r.per_point_deg, strict=False):
                c = QPointF(tx / ratio, ty / ratio)
                key = "ok" if err <= 2.0 else "warn" if err <= 3.5 else "bad"
                p.setPen(QPen(color(self.theme_name, key), 3))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(c, max(err * 12, 8), max(err * 12, 8))
                p.drawText(QRectF(c.x() - 60, c.y() + 30, 120, 30), Qt.AlignmentFlag.AlignCenter, f"{err:.1f}°")
        elif self.phase == "failed":
            p.drawText(top.adjusted(80, 0, -80, 60), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       self.fail_message)
        if self._auto_at is not None:
            left = max(0, int(self._auto_at - time.monotonic()) + 1)
            p.setPen(color(self.theme_name, "calib_fg" if light else "fg"))
            p.drawText(QRectF(0, self.height() - 110, self.width(), 40), Qt.AlignmentFlag.AlignCenter,
                       f"{self._auto_text} in {left} s")
        p.end()
