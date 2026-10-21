"""Experiment E7: which filter gives the best cursor? (PRD 12.3, 16.2)

Compares no filter, moving average, One Euro, Kalman and uncertainty-aware
Kalman on RMS jitter during fixations, lag after saccades (ms to reach 90% of
the jump) and dwell selection time, through the full control engine.

    python -m research.e7_filters --simulate --out results/e7_sim.json
    python -m research.e7_filters --log data/logs/session.jsonl --out results/e7.json   # recorded raw gaze
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.calibration.protocol import GazeObservation
from gazeos.control.engine import ControlEngine, EngineInput
from gazeos.os_backends.base import RecordingBackend
from gazeos.settings import Settings
from gazeos.types import TrackingStatus
from gazeos.vision.quality import QualityReport
from research.metrics import rms_precision

FILTERS = ("none", "moving_average", "one_euro", "kalman", "uncertainty_kalman")
OK = QualityReport(ok=True, status=TrackingStatus.TRACKING)


def scanpath(rng: np.random.Generator, geometry: ScreenGeometry, n_fix: int = 40, fps: float = 30.0,
             noise_deg: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixations of 0.6-1.5 s joined by instant saccades, with heteroscedastic noise.

    Returns time, noisy gaze (px), true gaze (px), per-sample sigma (px).
    """
    ppd = geometry.deg_to_px(1.0)
    ts, noisy, true, sig = [], [], [], []
    t = 0.0
    for _ in range(n_fix):
        target = np.array([rng.uniform(0.1, 0.9) * geometry.width_px, rng.uniform(0.1, 0.9) * geometry.height_px])
        for _ in range(int(rng.uniform(0.6, 1.5) * fps)):
            s = noise_deg * ppd * (3.0 if rng.random() < 0.1 else 1.0)  # occasional bad frames
            ts.append(t)
            true.append(target)
            noisy.append(target + rng.normal(0, s, 2))
            sig.append(s)
            t += 1 / fps
    return np.asarray(ts), np.asarray(noisy), np.asarray(true), np.asarray(sig)


def run_filter(kind: str, data: tuple[np.ndarray, ...], geometry: ScreenGeometry) -> dict[str, float]:
    ts, noisy, true, sig = data
    s = Settings()
    s.filter.kind = kind  # type: ignore[assignment]
    s.snap.enabled = False
    s.system.rest_reminder_min = 0
    backend = RecordingBackend()
    eng = ControlEngine(s, geometry, backend)
    eng.error_px = float(np.median(sig))
    out = []
    fire_times = []
    for t, p in zip(ts, noisy, strict=True):
        o = eng.update(EngineInput(float(t), GazeObservation(np.zeros(2), np.zeros(3), p), OK))
        out.append(o.cursor)
        if any(e.startswith("dwell:") for e in o.events):
            fire_times.append(t)
        eng.error_px = float(np.median(sig))
    cur = np.asarray(out, float)
    # Segment fixations by changes of the true target.
    starts = [0] + [i for i in range(1, len(true)) if not np.allclose(true[i], true[i - 1])]
    ends = starts[1:] + [len(true)]
    jitter, lags, sel = [], [], []
    ppd = geometry.deg_to_px(1.0)
    for a, b in zip(starts, ends, strict=True):
        seg = cur[a:b]
        settled = seg[len(seg) // 2:]
        jitter.append(rms_precision(settled) / ppd)
        if a > 0:
            jump = np.linalg.norm(true[a] - true[a - 1])
            dist = np.linalg.norm(seg - true[a], axis=1)
            reached = np.nonzero(dist <= 0.1 * jump)[0]
            lags.append((ts[a + reached[0]] - ts[a]) * 1000 if len(reached) else (ts[b - 1] - ts[a]) * 1000)
        fired = [ft for ft in fire_times if ts[a] <= ft < ts[b - 1]]
        if fired:
            sel.append((fired[0] - ts[a]) * 1000)
    return {"rms_jitter_deg": float(np.mean(jitter)), "lag_ms": float(np.mean(lags)),
            "selection_ms": float(np.mean(sel)) if sel else float("nan"),
            "selection_rate": len(sel) / len(starts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    geometry = ScreenGeometry.from_settings(Settings().screen)
    results: dict[str, dict[str, float]] = {}
    for kind in FILTERS:
        runs = [run_filter(kind, scanpath(np.random.default_rng(seed), geometry), geometry) for seed in range(a.seeds)]
        results[kind] = {k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        print(f"{k:<20} jitter {v['rms_jitter_deg']:.3f}°  lag {v['lag_ms']:6.0f} ms  "
              f"selection {v['selection_ms']:6.0f} ms  rate {v['selection_rate']:.2f}")


if __name__ == "__main__":
    main()
