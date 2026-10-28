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


def seed_table(seed0: Path, extra: Path, e1: Path = E1) -> dict[str, Any]:
    """Arms re-run with seeds 1-2 (`extra/<arm>/seed<s>/fold<f>`; seed 0 in `seed0/<arm>/fold<f>`)
    against E1 run by run: same seed, same test subject. Paired t-test, Holm over the arms."""
    from scipy.stats import t as student_t
    from scipy.stats import ttest_rel

    from research.stats import holm

    out: dict[str, Any] = {}
    for arm in sorted(p for p in extra.iterdir() if p.is_dir()):
        pairs = []
        for s in (0, 1, 2):
            for f in FOLDS:
                a = test_error(seed0 / arm.name / f"fold{f}" if s == 0 else arm / f"seed{s}" / f"fold{f}")
                r = test_error(e1 / f"fold{f}" if s == 0 else e1 / f"seed{s}" / f"fold{f}")
                if a is not None and r is not None:
                    pairs.append((s, a, r))
        if not pairs:
            continue
        a_err, r_err = np.array([p[1] for p in pairs]), np.array([p[2] for p in pairs])
        d = a_err - r_err
        out[arm.name] = {"runs": len(pairs), "mean": float(a_err.mean()), "e1_mean": float(r_err.mean()),
                         "delta": float(d.mean()), "delta_sd": float(d.std(ddof=1)) if len(d) > 1 else None,
                         "delta_ci95": float(student_t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
                         if len(d) > 1 else None,
                         "better_in": int((d < 0).sum()),
                         "p": float(ttest_rel(a_err, r_err).pvalue) if len(d) > 2 else None,
                         "seed_means": {s: float(np.mean([p[1] for p in pairs if p[0] == s]))
                                        for s in sorted({p[0] for p in pairs})}}
    tested = [k for k, v in out.items() if v["p"] is not None]
    for k, ph in zip(tested, holm([out[k]["p"] for k in tested]), strict=True):
        out[k]["p_holm"] = ph
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--e1", default=str(E1))
    ap.add_argument("--seeds", nargs=2, metavar=("SEED0_DIR", "EXTRA_DIR"), default=None,
                    help="also compare the arms re-run with seeds 1-2, e.g. outputs/e2 outputs/e2_seeds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = table([Path(p) for p in a.experiments], Path(a.e1))
    if a.seeds:
        r["seeds"] = seed_table(Path(a.seeds[0]), Path(a.seeds[1]), Path(a.e1))
    ref = r["e1_seed0"]
    print(f"E1 seed 0: {ref['mean']:.2f} deg (folds {', '.join(f'{v:.2f}' for v in ref['per_fold'])}); "
          f"seed means {', '.join(f'{v:.2f}' for v in ref['seed_means'])}, spread {ref['seed_spread_deg']:.2f} deg")
    for name, v in r["arms"].items():
        folds = ", ".join("-" if e is None else f"{e:.2f}" for e in v["per_fold"])
        if v["mean"] is None:
            print(f"{name:28s} no folds done")
        else:
            print(f"{name:28s} {v['mean']:.2f} deg ({folds}); vs E1 {v['delta_vs_e1']:+.2f} deg")
    for name, v in r.get("seeds", {}).items():
        p = "" if v["p"] is None else f", paired t p {v['p']:.3f} (Holm {v.get('p_holm', v['p']):.3f})"
        print(f"{name} over {v['runs']} runs: {v['mean']:.2f} vs E1 {v['e1_mean']:.2f} deg, "
              f"{v['delta']:+.2f} deg, better in {v['better_in']} of {v['runs']}{p}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
