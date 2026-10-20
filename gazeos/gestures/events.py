"""Event logic on top of the learned gesture classifier (PRD 13.5).

An event fires when one class stays above probability 0.8 for its minimum
duration and the face then returns to neutral. A 300 ms refractory period
follows any event.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

GESTURE_CLASSES: tuple[str, ...] = (
    "none",
    "normal_blink",
    "long_blink",
    "left_wink",
    "right_wink",
    "eyebrow_raise",
    "mouth_open",
    "smile",
    "talking_or_yawning",
)
NEUTRAL_CLASSES = frozenset({"none", "normal_blink", "talking_or_yawning"})

DEFAULT_MIN_DURATION_MS: dict[str, float] = {
    "long_blink": 400.0,
    "left_wink": 300.0,
    "right_wink": 300.0,
    "eyebrow_raise": 250.0,
    "mouth_open": 300.0,
    "smile": 300.0,
}


class GestureEventDetector:
    def __init__(
        self,
        threshold: float = 0.8,
        refractory_ms: float = 300.0,
        min_duration_ms: Mapping[str, float] | None = None,
        enabled: set[str] | None = None,
    ) -> None:
        self.threshold = threshold
        self.refractory_s = refractory_ms / 1000.0
        self.min_duration_s = {k: v / 1000.0 for k, v in (min_duration_ms or DEFAULT_MIN_DURATION_MS).items()}
        self.enabled = enabled
        self._active: str | None = None
        self._active_start = 0.0
        self._qualified = False
        self._last_event_t = -1e9

    def update(self, t: float, probs: Mapping[str, float]) -> str | None:
        if t - self._last_event_t < self.refractory_s:
            return None
        top = max(probs, key=lambda k: probs[k])
        p = probs[top]
        confident_gesture = (
            p >= self.threshold
            and top not in NEUTRAL_CLASSES
            and (self.enabled is None or top in self.enabled)
        )
        if confident_gesture:
            if self._active != top:
                self._active, self._active_start, self._qualified = top, t, False
            elif t - self._active_start >= self.min_duration_s.get(top, 0.3):
                self._qualified = True
            return None
        # Not confidently showing the active gesture any more.
        if self._active is not None:
            fired = self._active if self._qualified and top in NEUTRAL_CLASSES else None
            if fired is not None or top in NEUTRAL_CLASSES:
                self._active, self._qualified = None, False
            if fired is not None:
                self._last_event_t = t
                return fired
        return None

    def reset(self) -> None:
        self._active, self._qualified = None, False


class FeatureWindow:
    """Sliding 30-frame window of 17-d features for the temporal classifier."""

    def __init__(self, length: int = 30, num_features: int = 17) -> None:
        self.length = length
        self.num_features = num_features
        self._buf: deque[NDArray[np.float64]] = deque(maxlen=length)

    def push(self, features: NDArray[np.float64]) -> None:
        self._buf.append(np.asarray(features, dtype=np.float64))

    @property
    def full(self) -> bool:
        return len(self._buf) == self.length

    def array(self) -> NDArray[np.float32]:
        """(1, length, num_features), zero-padded at the front until full."""
        out = np.zeros((self.length, self.num_features), dtype=np.float32)
        if self._buf:
            data = np.stack(self._buf).astype(np.float32)
            out[-len(data):] = data
        return out[None]


def scale_by_gesture_size(probs: Mapping[str, float], gesture_size: float) -> dict[str, float]:
    """User 'gesture size' setting: >1 demands stronger gestures (as in Project Gameface).

    Implemented as a temperature on non-neutral classes, renormalized.
    """
    out = {}
    for k, v in probs.items():
        out[k] = v ** gesture_size if k not in NEUTRAL_CLASSES else v
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}
