"""Stability and accuracy of gaze models on a guided webcam recording (9 dots, full frames).

Per model: the frame-to-frame change of the estimate while the user holds a dot, the
change that invisible perturbations of the same frame cause (sensor noise, JPEG
re-encoding, a 1 px shift), the per-frame spread after a polynomial calibration, and the
leave-one-dot-out error of that calibration (E6 protocol, docs/PLAN.md 27-28 Oct).

    python -m research.webcam_stability RECORDING.pkl models/gazenet_m3.onnx other.onnx --out r.json

The recording is a pickle of {"rec": [{"t", "part", "target", "jpg", ...}]} as written by
the guided recorder; it holds face images, so it stays local.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _perturb(face: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    f = face.astype(np.float32)
    if kind == "noise":
        f = f + rng.normal(0, 2.0, f.shape)
    elif kind == "jpeg":
        f = cv2.imdecode(cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], cv2.IMREAD_UNCHANGED)
    elif kind == "shift":
        f = cv2.warpAffine(f, np.float32([[1, 0, 1], [0, 1, 0]]), (f.shape[1], f.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(f, 0, 255).astype(np.uint8)


def evaluate(recording: str | Path, model_paths: list[str], settle_s: float = 0.8) -> dict[str, Any]:
    from gazeos.calibration.geometry import ScreenGeometry
    from gazeos.calibration.personalization import make_personalizer
    from gazeos.calibration.protocol import CalibrationSample
    from gazeos.inference.gaze import OnnxGazeEstimator
    from gazeos.pipeline import FrameProcessor
    from gazeos.settings import Settings
    from gazeos.vision.head_pose import CameraIntrinsics
    from gazeos.vision.landmarks import FaceLandmarker

    with open(recording, "rb") as fh:
        rec = [r for r in pickle.load(fh)["rec"] if r["part"] == "dots"]
    s = Settings()
    geo = ScreenGeometry.from_settings(s.screen)
    K = CameraIntrinsics.default(640, 480)
    ppd = geo.deg_to_px(1.0)
    lmk = FaceLandmarker()
    t0: dict[Any, float] = {}
    frames = []
    for r in rec:
        t0.setdefault(r["target"], r["t"])
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(r["jpg"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        frames.append((r["t"], r["target"], img, lmk.process(img, r["t"])))
    dots = list(t0)
    tgt = {d: np.array([d[0] * geo.width_px, d[1] * geo.height_px]) for d in dots}
    out: dict[str, Any] = {"recording": str(recording), "frames": len(frames)}
    for path in model_paths:
        est = OnnxGazeEstimator(path)
        proc = FrameProcessor(s, geo, K, None, est)
        rng = np.random.default_rng(0)
        items = []
        for t, d, img, lm in frames:
            pf = proc.process_landmarks(lm, t, img)
            if t - t0[d] >= settle_s and pf.observation is not None and pf.norm is not None and pf.estimate is not None:
                items.append((d, pf.norm, pf.estimate.pitchyaw_norm, pf.observation))
        consec = [np.degrees(np.abs(b[2] - a[2])) for a, b in zip(items, items[1:], strict=False) if a[0] == b[0]]
        sens = {}
        for kind in ("noise", "jpeg", "shift"):
            diffs = [np.degrees(np.abs(est.estimate(replace(n, face=_perturb(n.face, kind, rng))).pitchyaw_norm - p))
                     for _, n, p, _ in items[::3]]
            sens[kind] = [float(v) for v in np.median(diffs, 0)]
        per_dot = {d: [o for dd, _, _, o in items if dd == d] for d in dots}
        errs, spread = [], []
        for held in dots:
            p = make_personalizer("polynomial", geo)
            p.fit([CalibrationSample(o, tgt[d]) for d in dots if d != held for o in per_dot[d]])
            pred = np.array([p.predict_px(o) for o in per_dot[held]])
            errs.append(float(np.linalg.norm(np.median(pred, 0) - tgt[held]) / ppd))
            spread.append(math.sqrt(np.mean(np.sum((pred - np.median(pred, 0)) ** 2, 1))) / ppd)
        c = np.array(consec)
        out[Path(path).name] = {
            "frames": len(items),
            "frame_change_deg": {"pitch": float(np.median(c[:, 0])), "yaw": float(np.median(c[:, 1])),
                                 "rms": float(np.sqrt(np.mean(np.sum(c**2, 1))))},
            "perturbation_change_deg": sens,
            "calibrated_spread_deg": float(np.mean(spread)),
            "lodo_error_deg": {"mean": float(np.mean(errs)), "per_dot": errs},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = evaluate(a.recording, a.models)
    text = json.dumps(r, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
