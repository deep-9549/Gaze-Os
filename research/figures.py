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


def _save(fig: plt.Figure, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["e1", "e6", "e8", "e9", "e12"])
    ap.add_argument("results")
    ap.add_argument("out")
    a = ap.parse_args()
    r = json.loads(Path(a.results).read_text())
    if a.kind == "e1":
        bar_error_by_method(r["methods"], r["means"], r["sds"], a.out)
    elif a.kind == "e6":
        line_error_vs_points(r["error_deg"], a.out)
    elif a.kind == "e8":
        scatter_speed_accuracy(r["names"], r["ms"], r["error"], a.out)
    elif a.kind == "e9":
        grouped_breakdown(r, a.out)
    else:
        reliability_diagram(r["mean_sigma"], r["mean_error"], a.out)


if __name__ == "__main__":
    main()
