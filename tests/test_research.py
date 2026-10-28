from __future__ import annotations

import math

import numpy as np
import pytest

from research.metrics import (
    event_prf,
    fitts_throughput,
    iso_ring_targets,
    nasa_tlx,
    reliability_bins,
    rms_precision,
    sus_score,
    words_per_minute,
)
from research.stats import (
    cohens_d_paired,
    compare_methods,
    fairness_report,
    holm,
    repeated_measures,
)


def test_sus() -> None:
    assert sus_score([5, 1] * 5) == 100.0
    assert sus_score([1, 5] * 5) == 0.0
    assert sus_score([3] * 10) == 50.0
    with pytest.raises(ValueError):
        sus_score([3] * 9)


def test_nasa_tlx() -> None:
    r = dict.fromkeys(("mental", "physical", "temporal", "performance", "effort", "frustration"), 50.0)
    assert nasa_tlx(r) == 50.0
    w = {"mental": 5, "physical": 0, "temporal": 3, "performance": 3, "effort": 2, "frustration": 2}
    assert nasa_tlx(r, w) == 50.0


def test_fitts_throughput_textbook() -> None:
    """Perfectly accurate selections spread with SD 10 px along the axis."""
    rng = np.random.default_rng(0)
    n = 200
    starts = np.zeros((n, 2))
    targets = np.tile([400.0, 0.0], (n, 1))
    dx = rng.normal(0, 10, n)
    sel = targets + np.stack([dx, np.zeros(n)], 1)
    res = fitts_throughput(starts, targets, sel, [1.0] * n, [True] * n)
    assert res.effective_width == pytest.approx(41.33, rel=0.1)
    assert res.throughput_bps == pytest.approx(math.log2(400 / res.effective_width + 1), rel=0.05)


def test_iso_ring_alternates() -> None:
    t = iso_ring_targets(13, 100, (0, 0))
    assert len(t) == 13 and len(set(t)) == 13
    d = math.dist(t[0], t[1])
    assert d > 180  # consecutive targets are nearly opposite


def test_precision_wpm_prf_reliability() -> None:
    assert rms_precision([[0, 0], [3, 4]]) == 5.0
    assert words_per_minute("hello world", 60.0) == pytest.approx(2.0)
    r = event_prf([1.0, 2.05, 5.0], [1.1, 2.0, 3.0])
    assert r["tp"] == 2 and r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    ms, me, n = reliability_bins(np.arange(100.0), np.arange(100.0) * 2, bins=4)
    assert n.sum() == 100 and np.all(np.diff(me) > 0)


def test_holm_and_effect_sizes() -> None:
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert cohens_d_paired([2, 3, 4, 5], [1, 2, 3, 3]) > 1.0
    res = compare_methods([2.0, 2.1, 1.9, 2.2, 2.0, 2.3, 1.8], [1.5, 1.6, 1.4, 1.7, 1.5, 1.6, 1.4])
    assert res["p_t"] < 0.05 and res["mean_diff"] > 0


def test_repeated_measures_detects_difference() -> None:
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    rows = []
    for s in range(16):
        base = rng.normal(0, 0.3)
        for c, effect in (("eyes", 0.0), ("face", 0.2), ("mouse", 2.0)):
            rows.append({"subject": s, "condition": c, "value": 3 + effect + base + rng.normal(0, 0.2)})
    out = repeated_measures(pd.DataFrame(rows))
    assert out["omnibus"]["p"] < 0.001
    mouse_pairs = [p for p in out["pairwise"] if "mouse" in (p["a"], p["b"])]
    assert all(p["significant"] for p in mouse_pairs)


def test_fairness_report_flags_gaps() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"group": ["a"] * 5 + ["b"] * 5, "error_deg": [1.5] * 5 + [2.4] * 5})
    rep = fairness_report(df, "group")
    assert rep["flagged"] and rep["gap_deg"] == pytest.approx(0.9)


def test_e6_runner_on_simulated_subject(geometry, settings) -> None:
    from research.e6_personalization import run_subject, simulate_subject

    res = run_subject(*simulate_subject(0, geometry, settings), geometry, None)
    for m in ("offset", "polynomial", "gp", "finetune"):
        for n in (1, 3, 5, 9, 16):
            assert res[m][n] < res["none"][n], (m, n)


def test_e7_one_euro_beats_no_filter(geometry) -> None:
    from research.e7_filters import run_filter, scanpath

    data = scanpath(np.random.default_rng(0), geometry, n_fix=12, noise_deg=0.4)
    none, euro = run_filter("none", data, geometry), run_filter("one_euro", data, geometry)
    assert euro["rms_jitter_deg"] < 0.3 * none["rms_jitter_deg"]
    assert euro["selection_rate"] > 0.2


def test_figures_render(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from research import figures

    figures.line_error_vs_points({"none": {"1": 3.0, "9": 3.0}, "gp": {"1": 2.5, "9": 1.6}}, tmp_path / "a.png")
    figures.bar_error_by_method(["M0", "M1"], [5.0, 3.9], [0.4, 0.3], tmp_path / "b.png")
    figures.reliability_diagram([1, 2, 3], [1.1, 2.2, 2.9], tmp_path / "c.png")
    figures.grouped_breakdown({"Lighting": {"bright": 1.8, "dim": 2.4}}, tmp_path / "d.png")
    figures.scatter_speed_accuracy(["M3", "M4"], [20.0, 6.0], [1.7, 1.9], tmp_path / "e.png")
    assert len(list(tmp_path.glob("*.png"))) == 5


def test_ablation_table_compares_each_arm_with_e1_on_the_same_folds(tmp_path) -> None:
    import json

    from research.ablation_table import table

    def run(d, err):
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"test": {"mean_error_deg": err}}))

    e1 = tmp_path / "e1"
    for f, err in enumerate([3.0, 4.0, 5.0]):
        run(e1 / f"fold{f}", err)
        run(e1 / "seed1" / f"fold{f}", err + 0.3)
        run(e1 / "seed2" / f"fold{f}", err - 0.1)
    run(tmp_path / "e2" / "eyes_only" / "fold0", 3.5)  # only fold 0 done: compared on fold 0 only
    r = table([tmp_path / "e2"], e1)
    assert r["e1_seed0"]["seed_spread_deg"] == pytest.approx(0.4)
    arm = r["arms"]["e2/eyes_only"]
    assert arm["folds_done"] == 1 and arm["delta_vs_e1"] == pytest.approx(0.5)


def test_seed_table_pairs_each_run_with_e1_on_the_same_seed_and_subject(tmp_path) -> None:
    import json

    from research.ablation_table import seed_table

    def run(d, err):
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"test": {"mean_error_deg": err}}))

    e1 = tmp_path / "e1"
    for f, err in enumerate([3.0, 4.0, 5.0]):
        for s in (0, 1, 2):
            run(e1 / f"fold{f}" if s == 0 else e1 / f"seed{s}" / f"fold{f}", err + 0.1 * s)
            arm_err = err + 0.1 * s - 0.3 - 0.01 * f  # 0.3-0.32 deg better than E1 in every run
            run(tmp_path / "e2" / "no_head" / f"fold{f}" if s == 0 else
                tmp_path / "e2_seeds" / "no_head" / f"seed{s}" / f"fold{f}", arm_err)
    r = seed_table(tmp_path / "e2", tmp_path / "e2_seeds", e1)["no_head"]
    assert r["runs"] == 9 and r["better_in"] == 9
    assert r["delta"] == pytest.approx(-0.31)
    assert r["p"] < 0.001 and r["p_holm"] == pytest.approx(r["p"])
