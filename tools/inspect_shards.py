"""Sanity checks for packed gaze shards (docs/PLAN.md step 16).

Prints label statistics per shard set and the correlation between head and gaze
angles (people turn their head toward where they look, so both correlations should
be positive in every dataset; a negative one means a flipped head-pose sign), and
draws a grid of samples with their gaze drawn as arrows.

    python -m tools.inspect_shards "data/shards/mpiifacegaze/*.h5" "data/shards/rtgene/*.h5" --grid figures/shards.png
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


def load_labels(pattern: str, every: int = 20) -> dict[str, np.ndarray]:
    import h5py

    gaze, head, subj = [], [], []
    for p in sorted(glob.glob(pattern)):
        with h5py.File(p, "r") as f:
            gaze.append(f["gaze"][::every])
            head.append(f["head"][::every])
            subj.append(f["subject"][::every])
    return {"gaze": np.concatenate(gaze), "head": np.concatenate(head), "subject": np.concatenate(subj)}


def summary(pattern: str) -> dict[str, object]:
    d = load_labels(pattern)
    g, h = np.degrees(d["gaze"].astype(np.float64)), np.degrees(d["head"].astype(np.float64))
    return {
        "shards": len(glob.glob(pattern)),
        "subjects": sorted(int(s) for s in np.unique(d["subject"])),
        "sampled": int(len(g)),
        "gaze_mean_deg": g.mean(0).round(1).tolist(), "gaze_sd_deg": g.std(0).round(1).tolist(),
        "head_mean_deg": h.mean(0).round(1).tolist(), "head_sd_deg": h.std(0).round(1).tolist(),
        "corr_head_gaze_pitch": round(float(np.corrcoef(h[:, 0], g[:, 0])[0, 1]), 3),
        "corr_head_gaze_yaw": round(float(np.corrcoef(h[:, 1], g[:, 1])[0, 1]), 3),
    }


def grid(patterns: list[str], out: str, per_set: int = 8, seed: int = 0) -> None:
    import h5py

    rng = np.random.default_rng(seed)
    rows = []
    for pattern in patterns:
        tiles = []
        paths = sorted(glob.glob(pattern))
        for p in rng.choice(paths, per_set):
            with h5py.File(p, "r") as f:
                i = int(rng.integers(len(f["gaze"])))
                face = cv2.cvtColor(f["face"][i], cv2.COLOR_RGB2BGR)
                pitch, yaw = f["gaze"][i]
                right, left = f["right_eye"][i], f["left_eye"][i]
            # Arrow from the face centre along the gaze projected on the image (x right, y down).
            end = (int(112 - 90 * np.cos(pitch) * np.sin(yaw)), int(112 - 90 * np.sin(pitch)))
            cv2.arrowedLine(face, (112, 112), end, (0, 255, 255), 2, tipLength=0.2)
            eyes = cv2.cvtColor(np.hstack([right, left]), cv2.COLOR_GRAY2BGR)
            eyes = cv2.resize(eyes, (224, 67))
            tiles.append(np.vstack([face, eyes]))
        rows.append(np.hstack(tiles))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+", help="one glob per shard set, e.g. 'data/shards/rtgene/*.h5'")
    ap.add_argument("--grid", help="write a sample grid (faces with gaze arrows, eye crops below) to this PNG")
    a = ap.parse_args()
    print(json.dumps({p: summary(p) for p in a.patterns}, indent=2))
    if a.grid:
        grid(a.patterns, a.grid)
        print("wrote", a.grid)


if __name__ == "__main__":
    main()
