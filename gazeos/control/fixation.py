"""Fixation and saccade detection (PRD 12.4, FR-15).

Saccade: the raw gaze lands more than 7 degrees from the smoothed cursor on two
consecutive frames in the same direction, or more than 14 degrees on one frame.
The PRD's velocity rule (30 deg/s) fires on webcam noise alone; see ADR 0006.
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
        saccade_jump_deg: float = 7.0,
        dispersion_deg: float = 1.0,
        min_fixation_ms: float = 100.0,
        max_window: int = 90,
        hold_window_s: float = 0.5,
        settle_s: float = 0.15,
        confirm_frames: int = 2,
        hold_deadband_deg: float = 0.0,
        hold_ease: float = 1.0,
        hold_lock_after_s: float = 0.0,
    ) -> None:
        self.px_per_deg = px_per_deg
        self.jump_px = saccade_jump_deg * px_per_deg
        self.confirm_frames = confirm_frames
        self.deadband_px = hold_deadband_deg * px_per_deg
        self.hold_ease = hold_ease
        # The dead band locks only once the fixation has settled, so it cannot stop the
        # cursor short while the median is still catching up with the eye.
        self.hold_lock_after_s = hold_lock_after_s
        self._held_since = 0.0
        self._shown: FloatArray | None = None  # point held during the current fixation
        self._jumps: list[FloatArray] = []  # offsets of the current saccade candidate
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
        self._jumps = []
        self._shown = None
        self.in_fixation = False

    def update(self, t: float, raw: FloatArray, filtered: FloatArray) -> FixationOutput:
        """raw: unfiltered gaze point (the jump target after a saccade);
        filtered: smoothed point (used for velocity, so per-frame model noise
        is not mistaken for a saccade)."""
        raw = np.asarray(raw, dtype=np.float64)
        filtered = np.asarray(filtered, dtype=np.float64)
        saccade = False
        if self._last is not None:
            jump = raw - self._last[1]  # raw gaze vs the cursor before this frame
            norm = float(np.linalg.norm(jump))
            same_way = all(
                float(np.dot(jump, j)) > 0.5 * norm * float(np.linalg.norm(j)) for j in self._jumps
            )
            if norm <= self.jump_px:
                self._jumps = []
            else:
                self._jumps = [*self._jumps, jump] if same_way else [jump]
            # A jump of twice the threshold is too large for noise: no need to wait.
            saccade = len(self._jumps) >= self.confirm_frames or norm > 2 * self.jump_px
        self._last = (t, filtered)

        if saccade:
            self._window.clear()
            self._jumps = []
            self._shown = None
            self._last = (t, raw.copy())  # the engine restarts its filter at the raw point
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
            point = np.median(held, axis=0)
            # Dead band: keep the shown point until the median moves clearly, then glide to it,
            # so the cursor sits still instead of trembling with the network's noise.
            if self._shown is None:
                self._held_since = t
            elif float(np.linalg.norm(point - self._shown)) >= self.deadband_px:
                self._held_since = t  # a real move: follow it fully before locking again
                point = self._shown + self.hold_ease * (point - self._shown)
            elif t - self._held_since < self.hold_lock_after_s:
                point = self._shown + self.hold_ease * (point - self._shown)
            else:
                point = self._shown
            self._shown = point.copy()
            return FixationOutput(EyeEvent.FIXATION, point.copy())
        self._shown = None
        return FixationOutput(EyeEvent.MOVING, filtered.copy())

    @staticmethod
    def _dispersion(pts: FloatArray) -> float:
        span = pts.max(axis=0) - pts.min(axis=0)
        return float(span[0] + span[1]) / 2.0
