"""Onboarding flow (PRD 7.6): camera preview and position guide, lighting check,
blink tuning, calibration, guided tutorial, done."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from gazeos.gestures.blink import tune_long_blink_threshold
from gazeos.pipeline import ProcessedFrame
from gazeos.types import TrackingStatus
from gazeos.ui.common import stylesheet
from gazeos.ui.debug_view import _pixmap
from gazeos.vision import features as feat

TUTORIAL_STEPS = [
    ("Open a website", "Look at your browser's address bar and dwell until the ring fills to click it."),
    ("Scroll it", "Choose Scroll on the action bar, then look near the bottom edge of the screen."),
    ("Click a link", "Look at any link and hold your gaze until the ring completes."),
    ("Type “hello”", "Open the keyboard from the action bar and type hello. Use the predictions."),
]


class Onboarding(QWidget):
    def __init__(
        self,
        start_calibration: Callable[[], None],
        on_finish: Callable[[float | None], None],
        theme_name: str = "default",
    ) -> None:
        super().__init__(None, Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("Welcome to GazeOS")
        self.setStyleSheet(stylesheet(theme_name))
        self.start_calibration = start_calibration
        self.on_finish = on_finish
        self.normal_ms: list[float] = []
        self.long_ms: list[float] = []
        self.blink_phase = "normal"
        self.threshold: float | None = None
        self._last_closure: float | None = None

        root = QVBoxLayout(self)
        self.stack = QStackedWidget()
        root.addWidget(self.stack)
        nav = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("onb_back")
        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("onb_next")
        self.back_btn.clicked.connect(lambda: self._go(-1))
        self.next_btn.clicked.connect(lambda: self._go(1))
        nav.addWidget(self.back_btn)
        nav.addWidget(self.next_btn)
        root.addLayout(nav)

        # 1-2. Camera preview, position and lighting.
        p1 = QWidget()
        l1 = QVBoxLayout(p1)
        self._title(l1, "Camera and position")
        self.preview = QLabel()
        self.preview.setMinimumSize(640, 480)
        l1.addWidget(self.preview)
        self.position_hint = QLabel("Looking for your face…")
        l1.addWidget(self.position_hint)
        self.stack.addWidget(p1)

        # 3. Blink tuning.
        p2 = QWidget()
        l2 = QVBoxLayout(p2)
        self._title(l2, "Blink tuning")
        self.blink_hint = QLabel()
        self.blink_hint.setWordWrap(True)
        l2.addWidget(self.blink_hint)
        self.stack.addWidget(p2)

        # 4. Calibration.
        p3 = QWidget()
        l3 = QVBoxLayout(p3)
        self._title(l3, "Calibration")
        l3.addWidget(QLabel("Nine dots will appear one at a time. Follow each with your eyes. It takes about 20 seconds."))
        cal = QPushButton("Start calibration")
        cal.setObjectName("onb_calibrate")
        cal.clicked.connect(self.start_calibration)
        l3.addWidget(cal)
        self.calib_result = QLabel("")
        l3.addWidget(self.calib_result)
        self.stack.addWidget(p3)

        # 5. Tutorial.
        p4 = QWidget()
        l4 = QVBoxLayout(p4)
        self._title(l4, "Try it")
        for i, (title, text) in enumerate(TUTORIAL_STEPS, 1):
            lab = QLabel(f"{i}. {title}: {text}")
            lab.setWordWrap(True)
            l4.addWidget(lab)
        l4.addWidget(QLabel("Close your eyes for 2 seconds at any time to pause or resume. Ctrl+Alt+G stops everything."))
        self.stack.addWidget(p4)

        # 6. Done.
        p5 = QWidget()
        l5 = QVBoxLayout(p5)
        self._title(l5, "All set")
        l5.addWidget(QLabel("Your profile is saved. Look at the top-right corner for 1.5 s to open the GazeOS menu."))
        done = QPushButton("Finish")
        done.setObjectName("onb_finish")
        done.clicked.connect(self._finish)
        l5.addWidget(done)
        self.stack.addWidget(p5)
        self._update_blink_hint()
        self.resize(900, 760)

    @staticmethod
    def _title(layout: QVBoxLayout, text: str) -> None:
        t = QLabel(text)
        t.setProperty("role", "title")
        layout.addWidget(t)

    def _go(self, d: int) -> None:
        i = max(0, min(self.stack.count() - 1, self.stack.currentIndex() + d))
        self.stack.setCurrentIndex(i)

    def _finish(self) -> None:
        self.close()
        self.on_finish(self.threshold)

    def calibration_done(self, score: int | None, mean_deg: float | None) -> None:
        if score is None or mean_deg is None:
            self.calib_result.setText("Calibration was cancelled. You can try again.")
        else:
            self.calib_result.setText(f"Score {score} / 100 (mean error {mean_deg:.1f}°).")
            self._go(1)

    def _update_blink_hint(self) -> None:
        if self.blink_phase == "normal":
            n = len(self.normal_ms)
            self.blink_hint.setText(f"Blink normally 5 times. ({n}/5)")
        elif self.blink_phase == "long":
            n = len(self.long_ms)
            self.blink_hint.setText(f"Now do 5 long blinks: close both eyes for about half a second. ({n}/5)")
        else:
            self.blink_hint.setText(f"Done. Your long-blink threshold is {self.threshold:.0f} ms.")

    def update_frame(self, pf: ProcessedFrame, image: NDArray[np.uint8], last_closure_ms: float | None) -> None:
        if not self.isVisible():
            return
        page = self.stack.currentIndex()
        r = pf.result
        if page == 0:
            vis = image.copy()
            self.preview.setPixmap(_pixmap(_draw_guide(vis, r.landmarks), 640))
            self.position_hint.setText(_position_hint(r.status, r.landmarks, image.shape[1]))
        elif page == 1 and last_closure_ms is not None and last_closure_ms != self._last_closure:
            self._last_closure = last_closure_ms
            if self.blink_phase == "normal":
                self.normal_ms.append(last_closure_ms)
                if len(self.normal_ms) >= 5:
                    self.blink_phase = "long"
            elif self.blink_phase == "long":
                if last_closure_ms > max(self.normal_ms) * 1.2:
                    self.long_ms.append(last_closure_ms)
                if len(self.long_ms) >= 5:
                    self.threshold = tune_long_blink_threshold(self.normal_ms, self.long_ms)
                    self.blink_phase = "done"
            self._update_blink_hint()


def _draw_guide(vis: NDArray[np.uint8], landmarks: NDArray[np.float64] | None) -> NDArray[np.uint8]:
    import cv2

    h, w = vis.shape[:2]
    cv2.ellipse(vis, (w // 2, int(h * 0.45)), (int(w * 0.17), int(h * 0.3)), 0, 0, 360, (77, 163, 255), 2)
    if landmarks is not None:
        x0, y0, x1, y1 = feat.face_bbox(landmarks)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (63, 207, 142), 2)
    return vis


def _position_hint(status: str, landmarks: NDArray[np.float64] | None, width: int) -> str:
    if status == TrackingStatus.NO_FACE.value or landmarks is None:
        return "No face found. Sit in front of the camera, with your face inside the outline."
    if status == TrackingStatus.MULTIPLE_FACES.value:
        return "More than one face is visible. Only the person using GazeOS should be in view."
    if status == TrackingStatus.LOW_LIGHT.value:
        return "It is too dark. Turn on a light in front of you, not behind you."
    eye_px = feat.inter_eye_distance_px(landmarks)
    if status == TrackingStatus.MOVE_CLOSER.value or eye_px < 60:
        return "Move a little closer to the screen."
    if eye_px > width * 0.35:
        return "Move back a little."
    cx = (landmarks[:, 0].min() + landmarks[:, 0].max()) / 2
    if abs(cx - width / 2) > width * 0.15:
        return "Move to the centre of the camera view."
    return "Good position. Press Next (look at it until the ring fills)."

