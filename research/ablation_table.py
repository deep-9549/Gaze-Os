"""E2 / E3 (PRD 16.2): one factor changed per arm, MPIIFaceGaze folds 0-2, E1's recipe.

Compares each arm's test error with E1 (arm d) on the same folds. E1 has three seeds, so
the per-fold spread between them is the noise band: a difference smaller than it says
nothing (Stage A re-runs alone moved fold 0 by 0.5 deg, PLAN.md log 26 Oct).

    python -m research.ablation_table D:/GazeOS-cache/outputs/e2 D:/GazeOS-cache/outputs/e3 --out results/ablations.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

FOLDS = (0, 1, 2)
E1 = Path("D:/GazeOS-cache/outputs/arms/d_eth_rtgene")


def test_error(run: Path) -> float | None:
    f = run / "result.json"
    return float(json.loads(f.read_text())["test"]["mean_error_deg"]) if f.exists() else None


def reference(e1: Path = E1) -> dict[str, Any]:
    seeds = [[test_error(e1 / f"fold{f}" if s == 0 else e1 / f"seed{s}" / f"fold{f}") for f in FOLDS] for s in (0, 1, 2)]
    per_fold = np.array([[np.nan if v is None else v for v in row] for row in seeds])
    return {"per_fold": seeds[0], "mean": float(np.mean(seeds[0])),
            "seed_means": [float(np.nanmean(r)) for r in per_fold],
            "seed_spread_deg": float(np.nanmax(per_fold.mean(1)) - np.nanmin(per_fold.mean(1)))}


def table(experiments: list[Path], e1: Path = E1) -> dict[str, Any]:
    ref = reference(e1)
    arms = {}
    for exp in experiments:
        for arm in sorted(p for p in exp.iterdir() if p.is_dir()):
            errs = [test_error(arm / f"fold{f}") for f in FOLDS]
            done = [(e, r) for e, r in zip(errs, ref["per_fold"], strict=True) if e is not None]
            arms[f"{exp.name}/{arm.name}"] = {
                "per_fold": errs,
                "mean": float(np.mean([e for e, _ in done])) if done else None,
                "delta_vs_e1": float(np.mean([e - r for e, r in done])) if done else None,
                "folds_done": len(done)}
    return {"e1_seed0": ref, "arms": arms}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--e1", default=str(E1))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = table([Path(p) for p in a.experiments], Path(a.e1))
    ref = r["e1_seed0"]
    print(f"E1 seed 0: {ref['mean']:.2f} deg (folds {', '.join(f'{v:.2f}' for v in ref['per_fold'])}); "
          f"seed means {', '.join(f'{v:.2f}' for v in ref['seed_means'])}, spread {ref['seed_spread_deg']:.2f} deg")
    for name, v in r["arms"].items():
        folds = ", ".join("-" if e is None else f"{e:.2f}" for e in v["per_fold"])
        if v["mean"] is None:
            print(f"{name:28s} no folds done")
        else:
            print(f"{name:28s} {v['mean']:.2f} deg ({folds}); vs E1 {v['delta_vs_e1']:+.2f} deg")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
