"""Glance aim: a one-second re-aim after each typed word.

Why (20 Nov 2026 live runs): "every time I use Fix cursor it feels really good for a few
seconds before getting ruined again". Fix cursor itself was accurate (two presses a few seconds
apart agreed within about 20 px), but the offset kept moving as the head shifted (the horizontal
error followed the sideways head position, correlation 0.57-0.77 within a sitting). Learning from
corrected mistakes could not keep up: in 128 recorded key presses a clean "wrong key, Del,
neighbouring key" happened once or twice.

So after Space or a chosen prediction a small dot is shown in the middle of the keyboard for
about a second; the user glances at it and goes on. Each word starts freshly aimed, where the
keys are. Nothing is pressed while the dot is up. A glance that did not happen (the gaze was
scattered, or the correction is too large to be drift) changes nothing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class GlanceAim:
    def __init__(
        self,
        target_px: tuple[float, float],
        t_start: float,
        sample_from_s: float = 0.35,
        duration_s: float = 1.0,
        min_samples: int = 8,
        max_shift_px: float = 150.0,
        max_spread_px: float = 110.0,
    ) -> None:
        self.target = np.asarray(target_px, dtype=np.float64)
        self.t_start = t_start
        self.sample_from_s = sample_from_s
        self.duration_s = duration_s
        self.min_samples = min_samples
        self.max_shift_px = max_shift_px
        self.max_spread_px = max_spread_px
        self._points: list[FloatArray] = []
        self.origins: list[FloatArray] = []  # gaze origins (app mm) of the samples used
        self.reason = ""  # why the last result() gave no correction

    def add(self, t: float, predicted_px: FloatArray, origin: FloatArray | None = None) -> None:
        """A gaze sample (where the cursor would be now, without posture compensation) at time t."""
        since = t - self.t_start
        if self.sample_from_s <= since <= self.duration_s:
            self._points.append(np.asarray(predicted_px, dtype=np.float64))
            if origin is not None:
                self.origins.append(np.asarray(origin, dtype=np.float64))

    def result(self) -> FloatArray | None:
        """How far to move the cursor (px), or None when the glance did not give a clear answer."""
        if len(self._points) < self.min_samples:
            self.reason = f"too few samples ({len(self._points)})"
            return None
        pts = np.stack(self._points)
        med = np.median(pts, axis=0)
        spread = float(np.median(np.linalg.norm(pts - med, axis=1)))
        if spread > self.max_spread_px:
            self.reason = f"gaze not on the dot (spread {spread:.0f} px)"
            return None
        shift = self.target - med
        if float(np.linalg.norm(shift)) > self.max_shift_px:
            self.reason = f"correction too large to be drift ({np.linalg.norm(shift):.0f} px)"
            return None
        self.reason = ""
        return np.asarray(shift, dtype=np.float64)
