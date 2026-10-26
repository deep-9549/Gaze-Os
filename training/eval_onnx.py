"""Score any ONNX gaze model that follows the GazeOS contract on shards (E1, E8).

Runs the model through the same runtime wrapper the app uses
(`gazeos.inference.gaze.OnnxGazeEstimator`), so M1 (L2CS), M3, M4 and INT8 models
are all measured the same way. Reports the mean over subjects and the per-axis
correlation with the labels (a negative one means a convention mismatch).

    python -m training.eval_onnx data/weights/l2cs/l2cs_gaze360.onnx "data/shards/mpiifacegaze/*.h5" --every 10
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from gazeos.inference.gaze import OnnxGazeEstimator
from gazeos.vision.geometry import angular_error_deg
from gazeos.vision.normalization import NormalizationResult


def evaluate_onnx(model: str, pattern: str, every: int = 1, threads: int = 4) -> dict[str, Any]:
    import h5py

    est = OnnxGazeEstimator(model, threads=threads)
    errors: dict[int, list[float]] = {}
    pairs = []
    for path in sorted(glob.glob(pattern)):
        with h5py.File(path, "r") as f:
            for i in range(0, len(f["gaze"]), every):
                norm = NormalizationResult(face=f["face"][i], right_eye=f["right_eye"][i], left_eye=f["left_eye"][i],
                                           R_n=np.eye(3), S=np.eye(3), W=np.eye(3), head_pyr=tuple(f["head"][i]))
                pred, label = est.estimate(norm).pitchyaw_norm, f["gaze"][i]
                errors.setdefault(int(f["subject"][i]), []).append(float(angular_error_deg(pred, label)[0]))
                pairs.append((*pred, *label))
    p = np.array(pairs)
    per_subject = {k: float(np.mean(v)) for k, v in sorted(errors.items())}
    return {
        "model": model, "shards": pattern, "every": every, "n": len(p),
        "mean_of_subjects_deg": float(np.mean(list(per_subject.values()))),
        "per_subject_deg": per_subject,
        "corr_pitch": float(np.corrcoef(p[:, 0], p[:, 2])[0, 1]),
        "corr_yaw": float(np.corrcoef(p[:, 1], p[:, 3])[0, 1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("shards", help="glob of shard files")
    ap.add_argument("--every", type=int, default=1, help="use every n-th sample")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", help="write the report as JSON here")
    a = ap.parse_args()
    report = evaluate_onnx(a.model, a.shards, a.every, a.threads)
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
