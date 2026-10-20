"""Latency profiler and soak test (PRD 18.2, 23 'Performance').

Runs the real pipeline on the camera (or on synthetic landmarks with --simulate)
and reports per-stage timings, FPS, CPU and memory, against the budget.

    python -m tools.latency_profiler --seconds 60
    python -m tools.latency_profiler --seconds 600 --soak       # 10-minute soak: memory growth < 50 MB
    python -m tools.latency_profiler --simulate --seconds 20
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.control.engine import ControlEngine
from gazeos.os_backends.base import RecordingBackend
from gazeos.pipeline import FrameProcessor
from gazeos.settings import Settings
from gazeos.vision.head_pose import CameraIntrinsics

BUDGET_MS = {"landmarks": 8.0, "head_pose": 2.0, "normalize": 2.0, "gaze_net": 8.0, "gesture": 1.0, "engine": 1.0}


def profile(seconds: float, simulate: bool, model: str | None, camera: int) -> dict[str, object]:
    s = Settings()
    if model:
        s.gaze_model = model
    geometry = ScreenGeometry.from_settings(s.screen)
    K = CameraIntrinsics.default(s.camera.width, s.camera.height)
    gaze = None
    if model and Path(model).exists():
        from gazeos.inference.gaze import OnnxGazeEstimator

        gaze = OnnxGazeEstimator(model)
    proc_ = psutil.Process()
    timings: dict[str, list[float]] = defaultdict(list)
    engine = ControlEngine(s, geometry, RecordingBackend())
    rss0 = proc_.memory_info().rss
    proc_.cpu_percent(None)
    frames = 0
    t_end = time.monotonic() + seconds
    if simulate:
        from gazeos.sim import SimulatedUser, SyntheticFace

        user = SimulatedUser(geometry, SyntheticFace(K, noise_px=0.3))
        proc = FrameProcessor(s, geometry, K, None, gaze)
        rng = np.random.default_rng(0)
        while time.monotonic() < t_end:
            t = time.monotonic()
            lm = user.look_at(rng.uniform(0, geometry.width_px), rng.uniform(0, geometry.height_px))
            pf = proc.process_landmarks(lm, t)
            t1 = time.perf_counter()
            engine.update(pf.engine_input)
            pf.result.timings_ms["engine"] = (time.perf_counter() - t1) * 1000
            for k, v in pf.result.timings_ms.items():
                timings[k].append(v)
            frames += 1
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - t)))
    else:
        from gazeos.capture.camera import CaptureThread, OpenCVSource
        from gazeos.vision.landmarks import FaceLandmarker

        cap = CaptureThread(OpenCVSource(camera, s.camera.width, s.camera.height, s.camera.fps))
        cap.start()
        proc = FrameProcessor(s, geometry, K, FaceLandmarker(), gaze)
        while time.monotonic() < t_end:
            fr = cap.latest()
            if fr is None:
                continue
            pf = proc.process_image(fr.image_rgb, fr.t_capture)
            t1 = time.perf_counter()
            engine.update(pf.engine_input)
            pf.result.timings_ms["engine"] = (time.perf_counter() - t1) * 1000
            pf.result.timings_ms["capture_to_cursor"] = (time.monotonic() - fr.t_capture) * 1000
            for k, v in pf.result.timings_ms.items():
                timings[k].append(v)
            frames += 1
        cap.stop()
    report: dict[str, object] = {
        "frames": frames,
        "fps": frames / seconds,
        "cpu_percent_total": proc_.cpu_percent(None) / psutil.cpu_count(),
        "rss_mb": proc_.memory_info().rss / 2**20,
        "rss_growth_mb": (proc_.memory_info().rss - rss0) / 2**20,
        "stages_ms": {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                          "budget": BUDGET_MS.get(k)} for k, v in timings.items()},
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--soak", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = profile(a.seconds, a.simulate, a.model, a.camera)
    print(json.dumps(r, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2))
    if a.soak and float(r["rss_growth_mb"]) > 50:  # type: ignore[arg-type]
        raise SystemExit("soak test failed: memory grew by more than 50 MB")


if __name__ == "__main__":
    main()
