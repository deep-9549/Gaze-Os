"""Optional accurate camera intrinsics from a printed checkerboard (PRD 9.2, ablation E4).

Print an A4 checkerboard (9 x 6 inner corners, 25 mm squares), then:

    python -m tools.camera_calibration --camera 0 --out camera_intrinsics.json

Press SPACE to capture a view (take about 15 from different angles), Q to finish.
Set `camera.intrinsics_file` in settings to the output file.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from gazeos.vision.head_pose import CameraIntrinsics


def calibrate(images: list[np.ndarray], pattern: tuple[int, int], square_mm: float) -> tuple[CameraIntrinsics, float]:
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm
    obj_pts, img_pts = [], []
    size = None
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            obj_pts.append(objp)
            img_pts.append(corners)
    if len(obj_pts) < 5 or size is None:
        raise ValueError(f"need at least 5 checkerboard views, found {len(obj_pts)}")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return CameraIntrinsics(np.asarray(K, float), np.asarray(dist, float).ravel()), float(rms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--out", default="camera_intrinsics.json")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.camera)
    shots: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()
        found, corners = cv2.findChessboardCorners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (a.cols, a.rows))
        if found:
            cv2.drawChessboardCorners(vis, (a.cols, a.rows), corners, found)
        cv2.putText(vis, f"views: {len(shots)}  SPACE capture  Q finish", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.imshow("GazeOS camera calibration", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            shots.append(frame)
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    intr, rms = calibrate(shots, (a.cols, a.rows), a.square_mm)
    intr.save(a.out)
    print(f"saved {a.out} (reprojection RMS {rms:.3f} px)")


if __name__ == "__main__":
    main()
