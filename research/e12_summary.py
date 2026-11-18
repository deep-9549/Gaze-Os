"""E12 (PRD 16): can the model tell when it is wrong? Summary over the E1 models.

Reads, for every E1 run (3 seeds x 15 leave-one-subject-out folds), the uncertainty-head
refit of `training.fit_uncertainty` (best_unc.json and best_unc.test.npz): Spearman rho
between predicted sigma and actual error on the held-out subject, and a reliability
diagram pooled over all held-out samples (deciles of predicted sigma).

    python -m research.e12_summary D:/GazeOS-cache/outputs/arms/d_eth_rtgene --out results/e12.json --figure figures/e12.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def run_dirs(arm: Path) -> list[tuple[int, int, Path]]:
    out = []
    for seed in (0, 1, 2):
        for fold in range(15):
            d = arm / f"fold{fold}" if seed == 0 else arm / f"seed{seed}" / f"fold{fold}"
            out.append((seed, fold, d))
    return out


def summarize(arm: Path, bins: int = 10) -> dict[str, Any]:
    rows, sig, err = [], [], []
    for seed, fold, d in run_dirs(arm):
        js, npz = d / "best_unc.json", d / "best_unc.test.npz"
        if not (js.exists() and npz.exists()):
            continue
        r = json.loads(js.read_text())
        rows.append({"seed": seed, "fold": fold, "spearman": r["test"]["spearman_after"],
                     "spearman_before": r["test"]["spearman_before"], "mean_error_deg": r["test"]["mean_error_deg"],
                     "median_sigma_deg": r["test"]["median_sigma_deg_after"], "z2": r["test"]["mean_z2_after"],
                     "fixed_range": "log_var_range" in r})
        z = np.load(npz)
        sig.append(np.degrees(np.sqrt(np.exp(z["log_var"]))).mean(1))
        err.append(z["angular_deg"])
    rho = np.array([r["spearman"] for r in rows])
    out: dict[str, Any] = {"runs": rows, "n_runs": len(rows),
                           "spearman_mean": float(rho.mean()) if len(rho) else None,
                           "spearman_sd": float(rho.std(ddof=1)) if len(rho) > 1 else None,
                           "spearman_positive_runs": int((rho > 0).sum())}
    if sig:
        s, e = np.concatenate(sig), np.concatenate(err)
        edges = np.quantile(s, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, bins - 1)
        out["mean_sigma"] = [float(s[idx == b].mean()) for b in range(bins)]
        out["mean_error"] = [float(e[idx == b].mean()) for b in range(bins)]
        from scipy.stats import spearmanr

        out["pooled_spearman"] = float(spearmanr(s, e).correlation)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", default=None)
    a = ap.parse_args()
    r = summarize(Path(a.arm))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items() if k != "runs"}, indent=2))
    if a.figure and "mean_sigma" in r:
        from research.figures import reliability_diagram

        reliability_diagram(r["mean_sigma"], r["mean_error"], a.figure)


if __name__ == "__main__":
    main()
