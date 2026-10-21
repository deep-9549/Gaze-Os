"""Gesture recording and labelling tool (PRD 13.3).

Prompts the participant through 20 repetitions of each gesture in random
order, plus natural-behaviour blocks (reading, talking, watching) as
negatives. The operator holds SPACE while the gesture is performed, which sets
the per-frame label; frames outside a held key are labelled "none" (or the
negative block's class). Saves features (N, 17) and labels (N,) per session.

    python -m tools.label_gestures --participant P01 --out data/gestures
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from gazeos.gestures.events import GESTURE_CLASSES
from gazeos.settings import Settings
from gazeos.vision.features import gesture_features
from gazeos.vision.geometry import rotation_to_euler
from gazeos.vision.head_pose import CameraIntrinsics, HeadPoseEstimator

PROMPTS = {
    "normal_blink": "Blink normally once",
    "long_blink": "Close both eyes for about half a second",
    "left_wink": "Wink with your LEFT eye",
    "right_wink": "Wink with your RIGHT eye",
    "eyebrow_raise": "Raise your eyebrows",
    "mouth_open": "Open your mouth",
    "smile": "Smile",
}
NEGATIVE_BLOCKS = [("talking_or_yawning", "Talk about your day (60 s)", 60.0),
                   ("none", "Read anything on screen or look around (60 s)", 60.0)]


class Labeller(QWidget):
    def __init__(self, participant: str, out: Path, camera: int, reps: int) -> None:
        super().__init__()
        self.out = out / f"{participant}_{time.strftime('%Y%m%d-%H%M%S')}.npz"
        s = Settings()
        from gazeos.capture.camera import CaptureThread, OpenCVSource
        from gazeos.vision.landmarks import FaceLandmarker

        self.cap = CaptureThread(OpenCVSource(camera, s.camera.width, s.camera.height, s.camera.fps))
        self.cap.start()
        self.lm = FaceLandmarker()
        self.head = HeadPoseEstimator(CameraIntrinsics.default(s.camera.width, s.camera.height))
        trials = [c for c in PROMPTS for _ in range(reps)]
        random.shuffle(trials)
        self.queue: list[tuple[str, str, float]] = [(c, PROMPTS[c], 0.0) for c in trials] + NEGATIVE_BLOCKS
        self.current = self.queue.pop(0)
        self.block_end = 0.0
        self.held = False
        self.feats: list[np.ndarray] = []
        self.labels: list[int] = []
        self.lock = threading.Lock()
        self._stop = False
        self._prev: tuple[float, float, float] | None = None
        threading.Thread(target=self._loop, daemon=True).start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def _loop(self) -> None:
        while not self._stop:
            fr = self.cap.latest()
            if fr is None:
                continue
            r = self.lm.process(fr.image_rgb, fr.t_capture)
            if r.landmarks_px is None:
                continue
            pose = self.head.estimate(r.landmarks_px)
            pyr = rotation_to_euler(pose.R) if pose is not None else (0.0, 0.0, 0.0)
            delta = (0.0, 0.0, 0.0) if self._prev is None else tuple(a - b for a, b in zip(pyr, self._prev, strict=True))
            self._prev = pyr
            with self.lock:
                cls, _, dur = self.current
                # A negative block labels every frame; prompts label only while SPACE is held.
                label = cls if dur > 0 or self.held else "none"
                self.feats.append(gesture_features(r.landmarks_px, r.blendshapes, delta))
                self.labels.append(GESTURE_CLASSES.index(label))

    def keyPressEvent(self, e) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self.held = True
        elif e.key() == Qt.Key.Key_Escape:
            self.finish()

    def keyReleaseEvent(self, e) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self.held = False
            if self.current[2] == 0.0:
                self.advance()

    def advance(self) -> None:
        with self.lock:
            if not self.queue:
                self.finish()
                return
            self.current = self.queue.pop(0)
            if self.current[2] > 0:
                self.block_end = time.monotonic() + self.current[2]

    def paintEvent(self, _) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        if self.current[2] > 0 and time.monotonic() >= self.block_end:
            self.advance()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1e1f24" if not self.held else "#16477f"))
        f = QFont()
        f.setPointSize(28)
        p.setFont(f)
        p.setPen(QColor("#f2f3f5"))
        p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, self.current[1])
        f.setPointSize(14)
        p.setFont(f)
        p.setPen(QColor("#b8bcc6"))
        hint = "Operator: hold SPACE while the gesture happens · Esc saves and quits" if self.current[2] == 0 else ""
        p.drawText(QRectF(0, self.height() - 60, self.width(), 40), Qt.AlignmentFlag.AlignCenter,
                   f"{len(self.queue)} prompts left. {hint}")
        p.end()

    def finish(self) -> None:
        self._stop = True
        self.cap.stop()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            np.savez_compressed(self.out, features=np.asarray(self.feats, np.float32),
                                labels=np.asarray(self.labels, np.int64), classes=np.asarray(GESTURE_CLASSES))
        print(f"saved {len(self.labels)} frames to {self.out}")
        QApplication.quit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--out", default="data/gestures")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    app = QApplication(sys.argv)
    w = Labeller(a.participant, Path(a.out), a.camera, a.reps)
    w.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
