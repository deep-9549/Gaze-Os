"""Debug and demo view (FR-39, PRD 28.1 step 2): landmarks, normalized eye crops,
the gaze ray, fusion attention, per-stage timings and live error."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from gazeos.pipeline import ProcessedFrame
from gazeos.vision import face_model as fm
from gazeos.vision.geometry import pitchyaw_to_vector
from gazeos.vision.head_pose import CameraIntrinsics, project_points


def _pixmap(img: NDArray[np.uint8], width: int | None = None) -> QPixmap:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = np.ascontiguousarray(img)
    if width is not None and img.shape[1] != width:
        h = int(img.shape[0] * width / img.shape[1])
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    q = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q.copy())


class DebugView(QWidget):
    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        super().__init__(None, Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("GazeOS debug view")
        self.intrinsics = intrinsics
        grid = QGridLayout(self)
        self.camera = QLabel()
        self.face = QLabel()
        self.right_eye = QLabel()
        self.left_eye = QLabel()
        self.attention = QLabel()
        self.stats = QLabel()
        self.stats.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.stats.setStyleSheet("font-family: monospace; font-size: 12pt;")
        grid.addWidget(self.camera, 0, 0, 3, 1)
        grid.addWidget(self.face, 0, 1)
        grid.addWidget(self.right_eye, 1, 1)
        grid.addWidget(self.left_eye, 2, 1)
        grid.addWidget(self.attention, 0, 2)
        grid.addWidget(self.stats, 1, 2, 2, 1)
        self.live_error_px: float | None = None
        self.fps = 0.0

    def update_frame(self, pf: ProcessedFrame, image: NDArray[np.uint8]) -> None:
        if not self.isVisible():
            return
        vis = image.copy()
        r = pf.result
        if r.landmarks is not None:
            for x, y in r.landmarks[:, :2].astype(int)[::3]:
                cv2.circle(vis, (int(x), int(y)), 1, (80, 200, 255), -1)
            for idx in (fm.RIGHT_IRIS_CENTER, fm.LEFT_IRIS_CENTER):
                x, y = r.landmarks[idx, :2].astype(int)
                cv2.circle(vis, (int(x), int(y)), 3, (255, 80, 80), -1)
        if pf.pose is not None and pf.estimate is not None:
            o = pf.pose.gaze_origin
            g = pitchyaw_to_vector(pf.estimate.pitchyaw)
            a, b = project_points(np.stack([o, o + 150.0 * g]), self.intrinsics).astype(int)
            cv2.arrowedLine(vis, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 255, 0), 2, tipLength=0.2)
            # Confidence circle grows when the network is unsure (e.g. eyes covered).
            sig = float(np.degrees(np.mean(pf.estimate.sigma)))
            cv2.circle(vis, tuple(int(v) for v in b), int(4 + sig * 4), (255, 255, 0), 1)
        self.camera.setPixmap(_pixmap(vis, 640))
        if pf.norm is not None:
            self.face.setPixmap(_pixmap(pf.norm.face, 224))
            self.right_eye.setPixmap(_pixmap(pf.norm.right_eye, 180))
            self.left_eye.setPixmap(_pixmap(pf.norm.left_eye, 180))
        if pf.estimate is not None and pf.estimate.attention is not None:
            att = pf.estimate.attention
            att = (255 * (att - att.min()) / max(att.max() - att.min(), 1e-9)).astype(np.uint8)
            heat = cv2.applyColorMap(cv2.resize(att, (160, 160), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_VIRIDIS)
            self.attention.setPixmap(_pixmap(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)))
            self.attention.setToolTip("Fusion attention: face, right eye, left eye, head pose")
        lines = [f"FPS        {self.fps:5.1f}", f"status     {r.status}", f"state      {r.state}"]
        if r.gaze_pitch_yaw is not None:
            p, y = np.degrees(r.gaze_pitch_yaw)
            lines.append(f"gaze       pitch {p:+6.1f}°  yaw {y:+6.1f}°")
        if r.gaze_sigma is not None:
            lines.append(f"sigma      {np.degrees(np.mean(r.gaze_sigma)):5.2f}°")
        if r.screen_xy is not None:
            lines.append(f"cursor     {r.screen_xy}")
        if self.live_error_px is not None:
            lines.append(f"live error {self.live_error_px:5.0f} px")
        lines.append("")
        for k, v in sorted(r.timings_ms.items()):
            lines.append(f"{k:<14}{v:6.1f} ms")
        self.stats.setText("\n".join(lines))
