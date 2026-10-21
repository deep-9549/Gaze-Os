"""Bayesian target selection (PRD 12.6, FR-16).

P(T_j | g) is proportional to exp(-|g - c_j|^2 / 2 sigma^2) * pi_j, with a prior
favouring larger and recently used elements.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gazeos.types import UIElement

FloatArray = NDArray[np.float64]


@dataclass
class SnapDecision:
    element: UIElement | None
    probability: float
    point: tuple[float, float] | None  # where to snap the cursor, if snapping


class TargetSnapper:
    def __init__(self, threshold: float = 0.7, radius_sigmas: float = 2.0, recency_s: float = 120.0) -> None:
        self.threshold = threshold
        self.radius_sigmas = radius_sigmas
        self.recency_s = recency_s
        self._recent: dict[tuple[str, str, int, int], float] = {}

    @staticmethod
    def _key(e: UIElement) -> tuple[str, str, int, int]:
        return (e.role, e.name, e.rect.left, e.rect.top)

    def note_used(self, element: UIElement, t: float | None = None) -> None:
        self._recent[self._key(element)] = time.monotonic() if t is None else t

    def prior(self, element: UIElement, now: float) -> float:
        size = math.sqrt(max(element.rect.area, 1))
        p = math.log1p(size)
        used = self._recent.get(self._key(element))
        if used is not None and now - used < self.recency_s:
            p *= 1.5
        return p if element.invokable else p * 0.2

    def candidates(
        self, gaze: FloatArray, sigma_px: float, elements: Sequence[UIElement]
    ) -> list[UIElement]:
        radius = self.radius_sigmas * sigma_px
        out = []
        for e in elements:
            cx, cy = e.rect.center
            # Distance to the rectangle, not just its centre: large elements count when near.
            dx = max(e.rect.left - gaze[0], 0.0, gaze[0] - (e.rect.left + e.rect.width))
            dy = max(e.rect.top - gaze[1], 0.0, gaze[1] - (e.rect.top + e.rect.height))
            if math.hypot(dx, dy) <= radius:
                out.append(e)
        return out

    def decide(
        self,
        gaze: FloatArray,
        sigma_px: float,
        elements: Sequence[UIElement],
        now: float | None = None,
    ) -> SnapDecision:
        now = time.monotonic() if now is None else now
        cands = self.candidates(gaze, sigma_px, elements)
        if not cands:
            return SnapDecision(None, 0.0, None)
        scores = []
        for e in cands:
            cx, cy = e.rect.center
            d2 = (gaze[0] - cx) ** 2 + (gaze[1] - cy) ** 2
            scores.append(math.exp(-d2 / (2.0 * sigma_px**2)) * self.prior(e, now))
        total = sum(scores)
        if total <= 0:
            return SnapDecision(None, 0.0, None)
        probs = np.asarray(scores) / total
        best = int(np.argmax(probs))
        p = float(probs[best])
        element = cands[best]
        if p >= self.threshold:
            return SnapDecision(element, p, element.rect.center)
        return SnapDecision(element, p, None)

    def should_click_directly(self, decision: SnapDecision, error_radius_px: float) -> bool:
        """PRD 7.3 step 2: click directly when a confident target is larger than the error radius."""
        e = decision.element
        if e is None or decision.point is None:
            return False
        return min(e.rect.width, e.rect.height) >= error_radius_px
