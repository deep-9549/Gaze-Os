"""Experiment E6: which personalization works, with how few points? (PRD 11.4, 16.2)

Runs P0-P4 (plus P5 when a meta-learned init is given) with 1, 3, 5, 9 and 16
calibration points and reports angular error on held-out targets.

    # On GazeOS-Webcam recordings (session 1-2 calibrate, session 3 test):
    python -m research.e6_personalization --recordings "data/recordings/P*/session*.npz" --out results/e6.json
    # On simulated users (no data needed; checks the method end to end):
    python -m research.e6_personalization --simulate 10 --out results/e6_sim.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.calibration.personalization import MetaLearnedAdaptation, make_personalizer
from gazeos.calibration.protocol import CalibrationSample, grid_targets
from gazeos.pipeline import FrameProcessor
from gazeos.settings import Settings
from gazeos.vision.head_pose import CameraIntrinsics

POINTS = (1, 3, 5, 9, 16)
METHODS = ("none", "offset", "polynomial", "gp", "finetune")
SUBSETS = {1: [4], 3: [0, 4, 8], 5: [0, 2, 4, 6, 8], 9: list(range(9))}  # indices into the 3x3 grid


def choose_points(targets: list[tuple[float, float]], n: int, geometry: ScreenGeometry) -> list[tuple[float, float]]:
    if n == 16:
        return grid_targets(geometry, 16)
    grid = grid_targets(geometry, 9)
    return [grid[i] for i in SUBSETS[n]]


def simulate_subject(seed: int, geometry: ScreenGeometry, settings: Settings) -> tuple[dict, list]:  # type: ignore[type-arg]
    """Per-target calibration samples and held-out test samples for one simulated person."""
    from gazeos.sim import SimulatedUser, SyntheticFace

    rng = np.random.default_rng(seed)
    K = CameraIntrinsics.default(640, 480)
    face = SyntheticFace(K, kappa_deg=tuple(rng.normal(0, 2.0, 2)), noise_px=0.3, seed=seed)
    user = SimulatedUser(geometry, face, distance_mm=float(rng.uniform(500, 750)),
                         head_pyr_deg=tuple(rng.normal(0, 4, 3)))
    proc = FrameProcessor(settings, geometry, K, None)
    t = 0.0
    by_target: dict[tuple[float, float], list[CalibrationSample]] = defaultdict(list)
    for tgt in set(grid_targets(geometry, 9)) | set(grid_targets(geometry, 16)):
        for _ in range(20):
            t += 1 / 30
            obs = proc.process_landmarks(user.look_at(*tgt), t).observation
            if obs is not None:
                by_target[tgt].append(CalibrationSample(obs, np.asarray(tgt, float), t=t))
    test = []
    for _ in range(40):
        tgt = (float(rng.uniform(0.15, 0.85) * geometry.width_px), float(rng.uniform(0.15, 0.85) * geometry.height_px))
        for _ in range(5):
            t += 1 / 30
            obs = proc.process_landmarks(user.look_at(*tgt), t).observation
            if obs is not None:
                test.append(CalibrationSample(obs, np.asarray(tgt, float), t=t))
    return by_target, test


def run_subject(by_target: dict, test: list, geometry: ScreenGeometry, meta_init: str | None) -> dict[str, dict[int, float]]:  # type: ignore[type-arg]
    out: dict[str, dict[int, float]] = defaultdict(dict)
    methods = list(METHODS) + (["meta"] if meta_init else [])
    for n in POINTS:
        pts = choose_points(list(by_target), n, geometry)
        train = [s for p in pts for s in by_target.get(p, [])]
        if not train:
            continue
        for m in methods:
            p = MetaLearnedAdaptation.from_file(geometry, meta_init) if m == "meta" else make_personalizer(m, geometry)
            try:
                p.fit(train)
            except (ValueError, np.linalg.LinAlgError):
                continue
            errs = [geometry.px_to_deg(float(np.linalg.norm(p.predict_px(s.obs) - s.target_px))) for s in test]
            out[m][n] = float(np.mean(errs))
    return out


def summarize(per_subject: list[dict[str, dict[int, float]]]) -> dict[str, object]:
    agg: dict[str, dict[str, float]] = defaultdict(dict)
    sd: dict[str, dict[str, float]] = defaultdict(dict)
    for m in {k for s in per_subject for k in s}:
        for n in POINTS:
            vals = [s[m][n] for s in per_subject if n in s.get(m, {})]
            if vals:
                agg[m][str(n)] = float(np.mean(vals))
                sd[m][str(n)] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"error_deg": agg, "sd_deg": sd, "n_subjects": len(per_subject), "per_subject": per_subject}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", type=int, default=0, help="number of simulated subjects")
    ap.add_argument("--recordings", default=None)
    ap.add_argument("--meta-init", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", default=None)
    a = ap.parse_args()
    settings = Settings()
    geometry = ScreenGeometry.from_settings(settings.screen)
    per_subject = []
    if a.simulate:
        for seed in range(a.simulate):
            per_subject.append(run_subject(*simulate_subject(seed, geometry, settings), geometry, a.meta_init))
    elif a.recordings:
        per_subject = run_recordings(sorted(glob.glob(a.recordings)), settings, a.meta_init)
    result = summarize(per_subject)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    if a.figure:
        from research.figures import line_error_vs_points

        line_error_vs_points(result["error_deg"], a.figure)  # type: ignore[arg-type]
    print(json.dumps(result["error_deg"], indent=2))


def run_recordings(paths: list[str], settings: Settings, meta_init: str | None) -> list[dict[str, dict[int, float]]]:
    """GazeOS-Webcam: calibrate on sessions 1-2, test on session 3 of the same participant."""
    from gazeos.recording import RecordedSession

    by_participant: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        by_participant[Path(p).parent.name].append(p)
    results = []
    for files in by_participant.values():
        files = sorted(files)
        if len(files) < 2:
            continue
        sessions = [RecordedSession(f) for f in files]
        meta = sessions[0].meta
        s = settings.model_copy(deep=True)
        for k, v in meta["screen"].items():
            if hasattr(s.screen, k):
                setattr(s.screen, k, v)
        geometry = ScreenGeometry.from_settings(s.screen)
        K = CameraIntrinsics.default(int(meta["camera"]["width"]), int(meta["camera"]["height"]))
        proc = FrameProcessor(s, geometry, K, None)
        by_target: dict[tuple[float, float], list[CalibrationSample]] = defaultdict(list)
        test: list[CalibrationSample] = []
        for si, sess in enumerate(sessions):
            for fr in sess.frames():
                if fr.target_px is None:
                    continue
                obs = proc.process_landmarks(fr.landmarks, fr.t).observation
                if obs is None:
                    continue
                smp = CalibrationSample(obs, np.asarray(fr.target_px, float), t=fr.t)
                if si < len(sessions) - 1:
                    by_target[_snap(fr.target_px, geometry)].append(smp)
                else:
                    test.append(smp)
        random.Random(0).shuffle(test)
        results.append(run_subject(by_target, test[:2000], geometry, meta_init))
    return results


def _snap(p: tuple[float, float], geometry: ScreenGeometry) -> tuple[float, float]:
    """Map a recorded target to the nearest standard calibration point."""
    cands = set(grid_targets(geometry, 9)) | set(grid_targets(geometry, 16))
    return min(cands, key=lambda c: (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2)


if __name__ == "__main__":
    main()
