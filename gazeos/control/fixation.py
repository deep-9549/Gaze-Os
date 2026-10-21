"""Fixation and saccade detection (PRD 12.4, FR-15).

I-VT: angular velocity above 30 deg/s is a saccade.
I-DT: samples staying within 1 degree for at least 100 ms form a fixation;
during a fixation the cursor sits at the running median of its samples.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class EyeEvent(StrEnum):
    MOVING = "moving"
    SACCADE = "saccade"
    FIXATION = "fixation"


@dataclass
class FixationOutput:
    event: EyeEvent
    point: FloatArray  # where the cursor should be
    saccade_started: bool = False


class FixationDetector:
    def __init__(
        self,
        px_per_deg: float,
        saccade_velocity_deg_s: float = 30.0,
        dispersion_deg: float = 1.0,
        min_fixation_ms: float = 100.0,
        max_window: int = 90,
        hold_window_s: float = 0.5,
        settle_s: float = 0.15,
    ) -> None:
        self.px_per_deg = px_per_deg
        self.v_saccade = saccade_velocity_deg_s
        self.dispersion_px = dispersion_deg * px_per_deg
        self.min_fix_s = min_fixation_ms / 1000.0
        self.hold_window_s = hold_window_s
        self.settle_s = settle_s
        self._window: deque[tuple[float, FloatArray]] = deque(maxlen=max_window)
        self._last: tuple[float, FloatArray] | None = None
        self.in_fixation = False

    def reset(self) -> None:
        self._window.clear()
        self._last = None
        self.in_fixation = False

    def update(self, t: float, raw: FloatArray, filtered: FloatArray) -> FixationOutput:
        """raw: unfiltered gaze point (the jump target after a saccade);
        filtered: smoothed point (used for velocity, so per-frame model noise
        is not mistaken for a saccade)."""
        raw = np.asarray(raw, dtype=np.float64)
        filtered = np.asarray(filtered, dtype=np.float64)
        saccade = False
        if self._last is not None:
            dt = t - self._last[0]
            if dt > 0:
                v_deg = float(np.linalg.norm(filtered - self._last[1])) / self.px_per_deg / dt
                saccade = v_deg > self.v_saccade
        self._last = (t, filtered)

        if saccade:
            self._window.clear()
            self.in_fixation = False
            return FixationOutput(EyeEvent.SACCADE, raw.copy(), saccade_started=True)

        self._window.append((t, filtered.copy()))
        pts = np.stack([p for _, p in self._window])
        # Shrink the window from the front until it satisfies the dispersion rule.
        while len(pts) > 1 and self._dispersion(pts) > self.dispersion_px:
            self._window.popleft()
            pts = pts[1:]
            self.in_fixation = False
        # A deliberate small correction (the gaze settling somewhere new within the
        # dispersion limit) restarts the fixation instead of being swallowed.
        recent = [p for ts, p in self._window if t - ts <= self.settle_s]
        older = [p for ts, p in self._window if t - ts > self.settle_s]
        if len(recent) >= 3 and len(older) >= 3:
            shift = float(np.linalg.norm(np.median(np.stack(recent), 0) - np.median(np.stack(older), 0)))
            if shift > self.dispersion_px / 2.0:
                while self._window and t - self._window[0][0] > self.settle_s:
                    self._window.popleft()
        duration = self._window[-1][0] - self._window[0][0]
        if duration >= self.min_fix_s:
            self.in_fixation = True
            held = np.stack([p for ts, p in self._window if t - ts <= self.hold_window_s])
            return FixationOutput(EyeEvent.FIXATION, np.median(held, axis=0))
        return FixationOutput(EyeEvent.MOVING, filtered.copy())

    @staticmethod
    def _dispersion(pts: FloatArray) -> float:
        span = pts.max(axis=0) - pts.min(axis=0)
        return float(span[0] + span[1]) / 2.0
