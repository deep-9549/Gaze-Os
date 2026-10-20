"""Evaluation metrics (PRD 17.1, 17.4, 17.5, 14.6, 13.6)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def rms_precision(points: ArrayLike) -> float:
    """Root mean square of sample-to-sample distances during a fixation (lower is steadier)."""
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.sum(np.diff(p, axis=0) ** 2, axis=1))))


def on_screen_error(pred: ArrayLike, true: ArrayLike) -> np.ndarray:
    return np.linalg.norm(np.asarray(pred, float) - np.asarray(true, float), axis=-1)


@dataclass
class FittsResult:
    throughput_bps: float
    effective_width: float
    effective_id: float
    movement_time_s: float
    error_rate: float


def fitts_throughput(
    starts: ArrayLike, targets: ArrayLike, selections: ArrayLike, times_s: Sequence[float], hits: Sequence[bool]
) -> FittsResult:
    """ISO 9241-411 throughput for one condition (PRD 17.4).

    TP = IDe / MT, IDe = log2(De / We + 1), We = 4.133 x SDx where SDx is the
    spread of selection points projected on the movement axis and De the mean
    effective movement distance.
    """
    s, t, sel = (np.asarray(a, float) for a in (starts, targets, selections))
    axis = t - s
    dist = np.linalg.norm(axis, axis=1)
    u = axis / dist[:, None]
    # Signed overshoot along the movement axis.
    dx = np.sum((sel - t) * u, axis=1)
    we = 4.133 * float(np.std(dx, ddof=1)) if len(dx) > 1 else float("nan")
    de = float(np.mean(dist + dx))
    ide = math.log2(de / we + 1.0)
    mt = float(np.mean(times_s))
    return FittsResult(ide / mt, we, ide, mt, 1.0 - float(np.mean(hits)))


def iso_ring_targets(n: int = 13, radius: float = 300.0, center: tuple[float, float] = (960, 540)) -> list[tuple[float, float]]:
    """ISO 9241-411 multi-directional ring: targets visited in the standard alternating order."""
    pts = [(center[0] + radius * math.sin(2 * math.pi * i / n), center[1] - radius * math.cos(2 * math.pi * i / n))
           for i in range(n)]
    order, i = [], 0
    for _ in range(n):
        order.append(pts[i])
        i = (i + (n + 1) // 2) % n
    return order


def sus_score(answers: Sequence[int]) -> float:
    """System Usability Scale: 10 answers 1-5 -> 0-100 (odd items positive, even negative)."""
    if len(answers) != 10 or not all(1 <= a <= 5 for a in answers):
        raise ValueError("SUS needs 10 answers from 1 to 5")
    total = sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))
    return total * 2.5


NASA_TLX_SCALES = ("mental", "physical", "temporal", "performance", "effort", "frustration")


def nasa_tlx(ratings: dict[str, float], weights: dict[str, int] | None = None) -> float:
    """Raw TLX (mean of six 0-100 ratings), or weighted TLX with pairwise weights summing to 15."""
    if set(ratings) != set(NASA_TLX_SCALES):
        raise ValueError(f"ratings must cover {NASA_TLX_SCALES}")
    if weights is None:
        return float(np.mean([ratings[k] for k in NASA_TLX_SCALES]))
    if sum(weights.values()) != 15:
        raise ValueError("TLX weights must sum to 15")
    return float(sum(ratings[k] * weights[k] for k in NASA_TLX_SCALES) / 15.0)


def keystroke_savings(keystrokes: int, characters: int) -> float:
    return 1.0 - keystrokes / characters if characters else 0.0


def words_per_minute(text: str, seconds: float) -> float:
    """Standard text-entry WPM: (characters - 1) / 5 per minute."""
    return (max(len(text) - 1, 0) / 5.0) / (seconds / 60.0) if seconds > 0 else 0.0


def event_prf(detected: Sequence[float], labelled: Sequence[float], tolerance_s: float = 0.15) -> dict[str, float]:
    """Event-level precision/recall/F1: a detection is correct within 150 ms of a labelled event (PRD 13.6)."""
    det = sorted(detected)
    lab = sorted(labelled)
    used = [False] * len(lab)
    tp = 0
    for d in det:
        best, best_j = tolerance_s + 1e-9, -1
        for j, l in enumerate(lab):
            if not used[j] and abs(d - l) <= best:
                best, best_j = abs(d - l), j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    p = tp / len(det) if det else 0.0
    r = tp / len(lab) if lab else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp}


def false_activations_per_10min(n_events: int, seconds: float) -> float:
    return n_events / (seconds / 600.0) if seconds > 0 else 0.0


def reliability_bins(sigma: ArrayLike, error: ArrayLike, bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For the E12 reliability diagram: mean predicted sigma vs mean actual error per quantile bin."""
    s, e = np.asarray(sigma, float), np.asarray(error, float)
    edges = np.quantile(s, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, bins - 1)
    ms = np.array([s[idx == b].mean() if np.any(idx == b) else np.nan for b in range(bins)])
    me = np.array([e[idx == b].mean() if np.any(idx == b) else np.nan for b in range(bins)])
    n = np.array([int(np.sum(idx == b)) for b in range(bins)])
    return ms, me, n
