"""Figures for the report and viva (PRD 28.2). Static matplotlib, light surface, print-safe.

Style: categorical hues in a fixed order (never cycled), thin marks, one y-axis
per chart, recessive grid, a legend whenever there are two or more series, text
in ink colours rather than series colours.

    python -m research.figures e6 results/e6.json figures/e6_calibration_points.png
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical palette (light mode), slot order fixed.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SEQUENTIAL_BLUE = ["#d6e6f8", "#a9cbf0", "#76aae5", "#4a8cdc", "#2a78d6", "#1d5ea9", "#16477f"]
MUTED = "#898781"  # de-emphasized marks: "the rest" in an emphasis chart, "no evidence" in a diverging one
BASELINE = "#c3c2b7"
NEUTRAL = "#f0efec"  # diverging midpoint
BETTER, WORSE = "#2a78d6", "#e34948"  # diverging pair (blue, red)


def style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def bar_error_by_method(methods: Sequence[str], means: Sequence[float], sds: Sequence[float], out: str | Path,
                        title: str = "Mean angular error by method (E1)") -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    x = np.arange(len(methods))
    # One measure across categories: a single hue; identity is in the axis labels.
    ax.bar(x, means, width=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, means, yerr=sds, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
    for xi, m in zip(x, means, strict=True):
        ax.text(xi, m + max(sds) * 0.15 + 0.05, f"{m:.2f}°", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(x, methods)
    ax.set_ylim(0, max(np.asarray(means) + np.asarray(sds)) * 1.25)
    style(ax, title, "", "Angular error (degrees)")
    return _save(fig, out)


METHOD_LABELS = {"none": "P0 none", "offset": "P1 offset", "polynomial": "P2 polynomial", "gp": "P3 GP",
                 "finetune": "P4 fine-tune", "meta": "P5 meta-learned"}


def line_error_vs_points(results: dict[str, dict[str, float]], out: str | Path,
                         title: str = "Error after calibration vs number of points (E6)") -> Path:
    """results: {method: {n_points: error_deg}} -> one line per method in fixed P0-P5 order.

    A legend sits outside the plot; lines are direct-labelled only when there are
    four or fewer, so labels never collide.
    """
    order = [m for m in METHOD_LABELS if m in results] + [m for m in results if m not in METHOD_LABELS]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    for i, method in enumerate(order):
        by_n = results[method]
        ns = sorted(int(k) for k in by_n)
        ys = [by_n[str(n)] for n in ns]
        label = METHOD_LABELS.get(method, method)
        ax.plot(ns, ys, color=SERIES[i] if i < len(SERIES) else INK_2, linewidth=2, marker="o", markersize=4,
                label=label)
        if len(order) <= 4:
            ax.annotate(label, (ns[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                        color=INK_2, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 3, 5, 9, 16], ["1", "3", "5", "9", "16"])
    ax.axhline(2.0, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(16, 2.0, "2° target ", va="bottom", ha="right", color=INK_2, fontsize=8)
    style(ax, title, "Calibration points", "Angular error (degrees)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, out)


def scatter_speed_accuracy(names: Sequence[str], ms: Sequence[float], err: Sequence[float], out: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)
    for i, (n, x, y) in enumerate(zip(names, ms, err, strict=True)):
        ax.scatter([x], [y], s=60, color=SERIES[i % 3], edgecolor=SURFACE, linewidth=2, label=n, zorder=3)
        ax.annotate(n, (x, y), xytext=(6, 4), textcoords="offset points", color=INK_2, fontsize=8)
    ax.axvline(8.0, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("budget: 8 ms (PRD)", (8.0, 1.0), xycoords=("data", "axes fraction"), xytext=(-4, -2),
                textcoords="offset points", ha="right", va="top", color=INK_2, fontsize=8)
    style(ax, "Speed vs accuracy (E8)", "Milliseconds per frame (CPU)", "Angular error (degrees)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    return _save(fig, out)


def reliability_diagram(mean_sigma: Sequence[float], mean_error: Sequence[float], out: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    lim = float(np.nanmax([*mean_sigma, *mean_error])) * 1.1
    ax.plot([0, lim], [0, lim], color=GRID, linewidth=1.5, label="perfectly calibrated")
    ax.plot(mean_sigma, mean_error, color=SERIES[0], linewidth=2, marker="o", markersize=4, label="GazeNet-GOS")
    style(ax, "Does the model know when it is wrong? (E12)", "Predicted uncertainty (degrees)",
          "Actual error (degrees)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    return _save(fig, out)


def grouped_breakdown(conditions: dict[str, dict[str, float]], out: str | Path,
                      title: str = "Error by condition (E9)") -> Path:
    """conditions: {factor: {level: error}} drawn as small multiples, one panel per factor."""
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), dpi=200, sharey=True, squeeze=False)
    top = max(v for d in conditions.values() for v in d.values()) * 1.25
    for ax, (factor, levels) in zip(axes[0], conditions.items(), strict=True):
        x = np.arange(len(levels))
        vals = list(levels.values())
        ax.bar(x, vals, width=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(x, vals, strict=True):
            ax.text(xi, v + top * 0.02, f"{v:.1f}°", ha="center", color=INK, fontsize=8)
        ax.set_xticks(x, list(levels))
        ax.set_ylim(0, top)
        style(ax, factor, "", "Angular error (degrees)" if ax is axes[0][0] else "")
    fig.suptitle(title, x=0.02, ha="left", color=INK, fontweight="bold")
    return _save(fig, out)


FILTER_LABELS = {"none": "Unfiltered", "moving_average": "Moving average", "one_euro": "One Euro",
                 "kalman": "Kalman", "uncertainty_kalman": "Uncertainty Kalman"}


def filter_tradeoff(results: dict[str, dict[str, float]], out: str | Path, chosen: str = "one_euro",
                    title: str = "Steadier against slower: cursor filters (E7)") -> Path:
    """E7: one dot per filter, jitter on a still gaze against the lag it adds over the
    unfiltered cursor (measured lag includes the person's reaction time, the same for all).
    Emphasis form: the app's filter in the accent hue, the others gray, all named."""
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=200)
    base = results["none"]["lag_ms"]
    for name, r in results.items():
        x, y = r["lag_ms"] - base, r["rms_jitter_deg"]
        ax.scatter([x], [y], s=64, color=SERIES[0] if name == chosen else MUTED, edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        label = FILTER_LABELS.get(name, name)
        if r.get("selection_rate", 1.0) < 1.0:
            label += f": {r['selection_rate']:.0%} of dwell clicks complete"
        ax.annotate(label, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK if name == chosen else INK_2)
    ax.set_xlim(-8, max(r["lag_ms"] - base for r in results.values()) * 1.6)
    ax.set_ylim(0, max(r["rms_jitter_deg"] for r in results.values()) * 1.15)
    style(ax, title, "Lag added over the unfiltered cursor (ms)", "Jitter on a still gaze (degrees RMS)")
    return _save(fig, out)


E11_LABELS = {"L0 word list": "Word list only", "L0+L1 seed 3-gram": "+ small built-in 3-gram",
              "L0+L1 trained 3-gram": "+ trained 3-gram (the app)", "L0+L1+L2 transformer": "+ trained 3-gram + transformer"}


def keystroke_savings(results: dict[str, dict[str, float]], out: str | Path,
                      title: str = "Keystrokes saved by word prediction (E11)") -> Path:
    """E11: one bar per prediction condition, value at the tip. One series, so one hue."""
    names = list(results)
    vals = [100 * results[n]["keystroke_savings"] for n in names]
    names = [E11_LABELS.get(n, n) for n in names]
    fig, ax = plt.subplots(figsize=(6.0, 0.5 * len(names) + 1.3), dpi=200)
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, height=0.5, color=SERIES[0], zorder=2)
    for yi, v in zip(y, vals, strict=True):
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 100)
    style(ax, title, "Keystrokes saved (% of characters)", "")
    ax.grid(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    return _save(fig, out)


ARM_LABELS = {"e2/eyes_only": "Inputs: eyes only, no face", "e2/face_only": "Inputs: face only, no eye crops",
              "e2/no_head": "Inputs: no head pose", "e2/concat": "Fusion: concatenation, not attention",
              "e3/regression": "Loss: regression only", "e3/classification": "Loss: bin classification only",
              "e3/beta2": "Loss: regression weight 2, not 1"}  # also the row order


def ablation_deltas(rows: Sequence[tuple[str, float, float | None]], band: float, out: str | Path,
                    title: str = "What each design choice is worth (E2, E3)") -> Path:
    """rows: (label, mean difference in error from E1 in degrees, 95% CI half-width or None).
    Diverging bars from zero: blue less error, red more; gray inside +-band (the spread
    between E1's own seeds on the same folds), where a difference is no evidence."""
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(rows) + 1.5), dpi=200)
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-band, band, color=NEUTRAL, zorder=0, linewidth=0)
    ax.axvline(0, color=BASELINE, linewidth=1, zorder=1)
    reach = max(abs(d) + (ci or 0) for _, d, ci in rows)
    for yi, (_, d, ci) in zip(y, rows, strict=True):
        color = MUTED if abs(d) < band else BETTER if d < 0 else WORSE
        ax.barh([yi], [d], height=0.5, color=color, zorder=2)
        if ci:
            ax.errorbar([d], [yi], xerr=[ci], fmt="none", ecolor=INK_2, elinewidth=1, capsize=3, zorder=3)
        tip = d + (ci or 0) if d >= 0 else d - (ci or 0)
        ax.text(tip + (0.04 if d >= 0 else -0.04) * reach, yi, f"{d:+.2f}°".replace("-", "−"),
                ha="left" if d >= 0 else "right",
                va="center", color=INK, fontsize=9)
    ax.text(band, len(rows) - 0.4, " E1 seed-to-seed spread", ha="left", va="bottom", color=INK_2, fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(-reach * 1.35, reach * 1.35)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    style(ax, title, "Difference in mean error from E1 (degrees; left is better)", "")
    ax.grid(False)
    return _save(fig, out)


def ablation_rows(table: dict[str, Any]) -> list[tuple[str, float, float | None]]:
    """Rows for `ablation_deltas` from `research.ablation_table` output: the 3-seed paired
    difference with its 95% CI where an arm was re-run with seeds 1-2, else seed 0's."""
    rows: list[tuple[str, float, float | None]] = []
    order = list(ARM_LABELS)
    for name, v in sorted(table["arms"].items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)):
        if v["delta_vs_e1"] is None:
            continue
        s = table.get("seeds", {}).get(name.split("/", 1)[1]) if name.startswith("e2/") else None
        label = ARM_LABELS.get(name, name)
        if s:
            rows.append((f"{label} ({s['runs']} runs)", s["delta"], s["delta_ci95"]))
        else:
            rows.append((f"{label} ({v['folds_done']} runs)", v["delta_vs_e1"], None))
    return rows


def _save(fig: plt.Figure, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.1)  # left titles can overrun the axes
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["e1", "e6", "e7", "e8", "e9", "e11", "e12", "ablation"])
    ap.add_argument("results")
    ap.add_argument("out")
    a = ap.parse_args()
    r = json.loads(Path(a.results).read_text())
    if a.kind == "e1":
        bar_error_by_method(r["methods"], r["means"], r["sds"], a.out)
    elif a.kind == "e6":
        line_error_vs_points(r["error_deg"], a.out)
    elif a.kind == "e7":
        filter_tradeoff(r, a.out)
    elif a.kind == "e11":
        keystroke_savings(r, a.out)
    elif a.kind == "ablation":
        ablation_deltas(ablation_rows(r), r["e1_seed0"]["seed_spread_deg"], a.out)
    elif a.kind == "e8":
        scatter_speed_accuracy(r["names"], r["ms"], r["error"], a.out)
    elif a.kind == "e9":
        grouped_breakdown(r, a.out)
    else:
        reliability_diagram(r["mean_sigma"], r["mean_error"], a.out)


if __name__ == "__main__":
    main()
