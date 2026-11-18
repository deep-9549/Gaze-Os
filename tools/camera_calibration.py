"""Camera intrinsics from a checkerboard (PRD 9.2, ablation E4).

Without this, GazeOS assumes a focal length equal to the image width; on the development
laptop that put the same sitting position at 60 cm at 640x480 and at 80 cm at 1280x720
(REVIEW.md item 12). The focal length in pixels does not depend on the size of the
squares, so a checkerboard shown on a phone works as well as a printed one:

    python -m tools.camera_calibration --make-board board.png     # open it on a phone, full screen
    python -m tools.camera_calibration --out camera_intrinsics.json

Hold the board in front of the camera and move it: near and far, left and right, tilted.
A view is captured whenever the board is found in a new position; Q or Esc finishes
early. The camera opens the way the app opens it (same resolution and backend), and the
output path is written to the profile's `camera.intrinsics_file` with --profile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from gazeos.vision.head_pose import CameraIntrinsics


def make_board(path: str | Path, cols: int = 9, rows: int = 6, square_px: int = 120, margin_px: int = 80) -> Path:
    """A checkerboard image with `cols` x `rows` inner corners."""
    h, w = (rows + 1) * square_px + 2 * margin_px, (cols + 1) * square_px + 2 * margin_px
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y, x = margin_px + r * square_px, margin_px + c * square_px
                img[y:y + square_px, x:x + square_px] = 0
    path = Path(path)
    cv2.imwrite(str(path), img)
    return path


def find_corners(gray: np.ndarray, pattern: tuple[int, int], fast: bool = False) -> np.ndarray | None:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | (cv2.CALIB_CB_FAST_CHECK if fast else 0)
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))


def is_new_view(corners: np.ndarray, taken: list[np.ndarray], image_width: int, min_shift: float = 0.08) -> bool:
    """True when the board moved by more than `min_shift` of the image width (mean corner
    displacement) from every view taken so far."""
    pts = corners.reshape(-1, 2)
    return all(np.linalg.norm(pts - t.reshape(-1, 2), axis=1).mean() > min_shift * image_width for t in taken)


def calibrate(images: list[np.ndarray], pattern: tuple[int, int], square_mm: float,
              full_distortion: bool = False) -> tuple[CameraIntrinsics, float]:
    """Focal length, centre and one radial distortion term. The full five-term model fitted to
    a hand-held phone board (27 Oct 2026) broke head pose (distance 24 mm, reprojection
    111 px); a 640x480 webcam needs little more than k1. `full_distortion` for a printed,
    flat board."""
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm
    obj_pts, img_pts = [], []
    size = None
    for img in images:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners = find_corners(gray, pattern)
        if corners is not None:
            obj_pts.append(objp)
            img_pts.append(corners)
    if len(obj_pts) < 5 or size is None:
        raise ValueError(f"need at least 5 checkerboard views, found {len(obj_pts)}")
    flags = 0 if full_distortion else cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None, flags=flags)
    return CameraIntrinsics(np.asarray(K, float), np.asarray(dist, float).ravel()), float(rms)


def capture_views(width: int, height: int, index: int, pattern: tuple[int, int], target: int) -> list[np.ndarray]:
    """Live preview (Qt; this OpenCV build has no windows) that keeps a frame whenever the
    board is found in a new position."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QApplication, QLabel

    from gazeos.capture.camera import OpenCVSource

    app = QApplication.instance() or QApplication(sys.argv)
    src = OpenCVSource(index, width, height, 30)
    shots: list[np.ndarray] = []
    taken: list[np.ndarray] = []
    label = QLabel(None, Qt.WindowType.WindowStaysOnTopHint)
    label.setWindowTitle("GazeOS camera calibration")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def tick() -> None:
        rgb = src.read()
        if rgb is None:
            return
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray, pattern, fast=True)
        vis = rgb.copy()
        if corners is not None:
            cv2.drawChessboardCorners(vis, pattern, corners, True)
            if is_new_view(corners, taken, width):
                shots.append(bgr)
                taken.append(corners)
        msg = f"views {len(shots)}/{target}: move the board (near, far, sides, tilted). Q to finish."
        cv2.putText(vis, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 160, 0), 2)
        qimg = QImage(vis.data, vis.shape[1], vis.shape[0], vis.strides[0], QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg.copy()))
        if len(shots) >= target:
            label.close()

    def key(event) -> None:  # type: ignore[no-untyped-def]
        if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            label.close()

    label.keyPressEvent = key  # type: ignore[method-assign]
    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(30)
    label.resize(width + 40, height + 40)
    label.show()
    label.raise_()
    label.activateWindow()
    app.exec()
    timer.stop()
    src.release()
    return shots


def main() -> None:
    from gazeos.settings import Settings

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", type=int, default=None, help="camera index (default: from settings)")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square-mm", type=float, default=25.0, help="only scales the board's distance, not the result")
    ap.add_argument("--views", type=int, default=15)
    ap.add_argument("--out", default="camera_intrinsics.json")
    ap.add_argument("--make-board", metavar="PNG", help="write a checkerboard image and exit")
    ap.add_argument("--profile", default=None, help="also set camera.intrinsics_file in this GazeOS profile")
    ap.add_argument("--full-distortion", action="store_true", help="fit all five distortion terms (printed board)")
    a = ap.parse_args()
    if a.make_board:
        print(f"wrote {make_board(a.make_board, a.cols, a.rows)}")
        return
    from gazeos.profiles.store import ProfileStore

    profile = ProfileStore().get_or_create(a.profile) if a.profile else None
    settings = profile.load_settings() if profile is not None else Settings()
    cam = settings.camera
    shots = capture_views(cam.width, cam.height, cam.device_index if a.camera is None else a.camera,
                          (a.cols, a.rows), a.views)
    intr, rms = calibrate(shots, (a.cols, a.rows), a.square_mm, a.full_distortion)
    out = Path(a.out).resolve()
    intr.save(out)
    fx, fy = intr.matrix[0, 0], intr.matrix[1, 1]
    print(f"saved {out}: fx {fx:.1f} fy {fy:.1f} cx {intr.matrix[0, 2]:.1f} cy {intr.matrix[1, 2]:.1f} px "
          f"at {cam.width}x{cam.height} (default guess fx = {cam.width}); reprojection RMS {rms:.3f} px")
    if profile is not None:
        settings.camera.intrinsics_file = str(out)
        profile.save_settings(settings)
        print(f"profile {a.profile}: camera.intrinsics_file = {out}")


if __name__ == "__main__":
    main()
