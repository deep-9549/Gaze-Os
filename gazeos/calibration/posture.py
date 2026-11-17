"""Warn when the head has moved away from where the calibration was made.

A calibration maps gaze to the screen for one head position. On 18 Nov 2026 the user
calibrated, then moved about 19 cm closer to type, and the cursor landed about 5 cm off
(two keys); fitting at one distance and testing at the other cost 5-17 mm on its own. The
app sees the head position on every frame, so it can say so instead of failing silently.

Positions are the gaze origin in camera millimetres as the app estimates them (x to the
image right, y down, z away from the camera); the scale depends on the camera model, so
the limit is in the same units and is only used to compare positions with each other.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class PostureGuard:
    def __init__(self, limit_mm: float = 90.0, hold_s: float = 3.0, repeat_s: float = 60.0) -> None:
        self.limit_mm = limit_mm
        self.hold_s = hold_s
        self.repeat_s = repeat_s
        self.reference: FloatArray | None = None
        self._away_since: float | None = None
        self._last_warning = -1e9

    def set_reference(self, origins: FloatArray) -> None:
        """Head positions recorded during the calibration (N, 3); their median is the reference."""
        o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        o = o[np.isfinite(o).all(axis=1)]
        self.reference = np.median(o, axis=0) if len(o) else None
        self._away_since = None

    def update(self, t: float, origin_mm: FloatArray) -> str | None:
        """A message to show, or None. Shown once the head has been away for `hold_s`, then at
        most every `repeat_s` while it stays away."""
        if self.reference is None:
            return None
        delta = np.asarray(origin_mm, dtype=np.float64) - self.reference
        if not np.isfinite(delta).all() or float(np.linalg.norm(delta)) <= self.limit_mm:
            self._away_since = None
            return None
        if self._away_since is None:
            self._away_since = t
        if t - self._away_since < self.hold_s or t - self._last_warning < self.repeat_s:
            return None
        self._last_warning = t
        return (f"You have moved {describe(delta)} since calibrating, so the cursor may be off. "
                "Sit where you were, or calibrate again: look at the top-right corner, then Calibrate.")


def describe(delta_mm: FloatArray) -> str:
    """The main directions of a head movement, in the user's words. The camera faces the user:
    image right (+x) is the user's left; +y is down; +z is away from the screen."""
    dx, dy, dz = (float(v) for v in delta_mm)
    parts = []
    if abs(dz) >= max(abs(dx), abs(dy)) * 0.5:
        parts.append("closer to the screen" if dz < 0 else "further from the screen")
    if abs(dx) >= 40.0:
        parts.append("to your left" if dx > 0 else "to your right")
    if abs(dy) >= 40.0:
        parts.append("lower" if dy > 0 else "higher")
    return " and ".join(parts) if parts else "from where you were"
