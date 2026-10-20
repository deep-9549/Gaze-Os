"""GazeOS-Webcam data-collection app (PRD 15.2).

    python -m tools.collect --out data/recordings

Per session: 16-point static targets, a smooth-pursuit target, and free reading
(gesture negatives). Stores landmarks, normalized face and eye crops and the
labels; full camera frames are never written (PRD 15.2, 21.1). Requires the
consent box to be ticked, i.e. a signed consent form on file (docs/study/).
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QGuiApplication, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.calibration.protocol import grid_targets
from gazeos.recording import SessionMeta, SessionRecorder
from gazeos.settings import Settings
from gazeos.vision.geometry import vector_to_pitchyaw
from gazeos.vision.head_pose import CameraIntrinsics, HeadPoseEstimator
from gazeos.vision.normalization import normalize

READING_TEXT = (
    "Please read this passage at your normal pace. Eye trackers that work with an ordinary webcam could help "
    "many people who cannot use their hands. This recording helps us measure how the system behaves while "
    "you read, talk, or simply look around, so that it never clicks by accident. Thank you for taking part."
)
STATIC_S, KEEP_FROM_S, PURSUIT_S, READING_S = 2.0, 0.5, 30.0, 60.0


class SetupForm(QWidget):
    start = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GazeOS-Webcam collection")
        form = QFormLayout(self)
        self.pid = QLineEdit("P01")
        self.session = QComboBox()
        self.session.addItems(["1", "2", "3"])
        self.distance = QComboBox()
        self.distance.addItems(["50", "65", "80"])
        self.lighting = QComboBox()
        self.lighting.addItems(["bright", "dim"])
        self.glasses = QComboBox()
        self.glasses.addItems(["no", "yes"])
        self.skin = QComboBox()
        self.skin.addItems(["not given", "I-II", "III-IV", "V-VI"])
        self.age = QComboBox()
        self.age.addItems(["not given", "18-25", "26-40", "41-60", "60+"])
        self.consent = QCheckBox("Signed consent form is on file for this participant")
        go = QPushButton("Start session")
        go.clicked.connect(self._go)
        for label, w in (("Participant ID", self.pid), ("Session", self.session), ("Distance (cm)", self.distance),
                         ("Lighting", self.lighting), ("Glasses", self.glasses),
                         ("Skin tone (Fitzpatrick, self-reported)", self.skin), ("Age group", self.age)):
            form.addRow(label, w)
        form.addRow(self.consent)
        form.addRow(go)

    def _go(self) -> None:
        if not self.consent.isChecked():
            self.consent.setStyleSheet("color: #e34948; font-weight: bold;")
            return
        self.start.emit({"participant": self.pid.text().strip(), "session": self.session.currentText(),
                         "distance_cm": self.distance.currentText(), "lighting": self.lighting.currentText(),
                         "glasses": self.glasses.currentText(), "skin_tone": self.skin.currentText(),
                         "age_group": self.age.currentText()})
        self.close()


class Collector(QWidget):
    def __init__(self, info: dict[str, str], out_dir: Path, camera: int) -> None:
        super().__init__()
        self.info = info
        screen = QGuiApplication.primaryScreen()
        ratio = screen.devicePixelRatio()
        g, mm = screen.geometry(), screen.physicalSize()
        s = Settings()
        self.geometry_ = ScreenGeometry(int(g.width() * ratio), int(g.height() * ratio),
                                        mm.width() or s.screen.width_mm, mm.height() or s.screen.height_mm)
        self.ratio = ratio
        self.K = CameraIntrinsics.default(s.camera.width, s.camera.height)
        meta = SessionMeta(
            screen={"width_px": self.geometry_.width_px, "height_px": self.geometry_.height_px,
                    "width_mm": self.geometry_.width_mm, "height_mm": self.geometry_.height_mm},
            camera={"width": s.camera.width, "height": s.camera.height, "index": camera},
            participant=info["participant"], condition={k: v for k, v in info.items() if k != "participant"},
        )
        path = out_dir / info["participant"] / f"session{info['session']}_{time.strftime('%Y%m%d-%H%M%S')}.npz"
        self.recorder = SessionRecorder(path, meta, keep_crops=True)
        from gazeos.capture.camera import CaptureThread, OpenCVSource
        from gazeos.vision.landmarks import FaceLandmarker

        self.cap = CaptureThread(OpenCVSource(camera, s.camera.width, s.camera.height, s.camera.fps))
        self.cap.start()
        self.landmarker = FaceLandmarker()
        self.head = HeadPoseEstimator(self.K)
        self.static = grid_targets(self.geometry_, 16)
        np.random.default_rng().shuffle(self.static)  # type: ignore[arg-type]
        self.phase, self.index, self.t0 = "static", 0, time.monotonic()
        self.target: tuple[float, float] | None = None
        self.lock = threading.Lock()
        self._stop = False
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

    def _loop(self) -> None:
        while not self._stop:
            fr = self.cap.latest()
            if fr is None:
                continue
            lm = self.landmarker.process(fr.image_rgb, fr.t_capture)
            with self.lock:
                target, phase, since = self.target, self.phase, fr.t_capture - self.t0
            label = target if (phase == "pursuit" or (phase == "static" and since >= KEEP_FROM_S)) else None
            norm = gaze_n = None
            if lm.landmarks_px is not None:
                pose = self.head.estimate(lm.landmarks_px)
                if pose is not None:
                    norm = normalize(fr.image_rgb, lm.landmarks_px, pose, self.K)
                    if label is not None:
                        tgt = self.geometry_.screen_mm_to_camera(*self.geometry_.px_to_mm(*label))
                        gaze_n = vector_to_pitchyaw(norm.R_n @ (tgt - pose.gaze_origin))
            self.recorder.add(fr.t_capture, lm, label, norm, gaze_n)

    def _tick(self) -> None:
        now = time.monotonic()
        with self.lock:
            since = now - self.t0
            if self.phase == "static":
                if since >= STATIC_S:
                    self.index += 1
                    self.t0 = now
                    if self.index >= len(self.static):
                        self.phase, self.t0 = "pursuit", now
                if self.phase == "static":
                    self.target = self.static[self.index]
            if self.phase == "pursuit":
                if since >= PURSUIT_S:
                    self.phase, self.t0, self.target = "reading", now, None
                else:
                    w, h = self.geometry_.width_px, self.geometry_.height_px
                    self.target = (w / 2 + 0.4 * w * math.sin(2 * math.pi * since / 9.0),
                                   h / 2 + 0.38 * h * math.sin(2 * math.pi * since / 6.0))
            if self.phase == "reading" and since >= READING_S:
                self.phase = "done"
        if self.phase == "done":
            self._finish()
        self.update()

    def _finish(self) -> None:
        self.timer.stop()
        self._stop = True
        self.worker.join(timeout=2)
        self.cap.stop()
        path = self.recorder.save()
        print(f"saved {len(self.recorder)} frames to {path}")
        QApplication.quit()

    def keyPressEvent(self, e: object) -> None:  # noqa: N802
        if getattr(e, "key", lambda: None)() == Qt.Key.Key_Escape:
            self._finish()

    def paintEvent(self, _: object) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#1e1f24"))
        p.setPen(QColor("#f2f3f5"))
        f = QFont()
        f.setPointSize(20)
        p.setFont(f)
        if self.phase in ("static", "pursuit") and self.target is not None:
            since = time.monotonic() - self.t0
            r = max(6.0, 30.0 - 24.0 * min(since / 1.5, 1.0)) if self.phase == "static" else 10.0
            c = QPointF(self.target[0] / self.ratio, self.target[1] / self.ratio)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor("#4da3ff")))
            p.drawEllipse(c, r, r)
            p.setBrush(QBrush(QColor("#1e1f24")))
            p.drawEllipse(c, 2.5, 2.5)
        elif self.phase == "reading":
            p.drawText(QRectF(self.width() * 0.15, self.height() * 0.3, self.width() * 0.7, self.height() * 0.4),
                       Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft, READING_TEXT)
        p.setPen(QColor("#b8bcc6"))
        p.drawText(QRectF(0, self.height() - 50, self.width(), 40), Qt.AlignmentFlag.AlignCenter,
                   f"{self.info['participant']} · session {self.info['session']} · {self.phase} · Esc to stop")
        p.end()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/recordings")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    app = QApplication(sys.argv)
    form = SetupForm()
    holder: dict[str, Collector] = {}

    def begin(info: dict[str, str]) -> None:
        holder["c"] = Collector(info, Path(a.out), a.camera)
        holder["c"].showFullScreen()

    form.start.connect(begin)
    form.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
