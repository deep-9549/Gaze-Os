"""Dwell click state machine (PRD 12.7, FR-19).

Idle -> Dwelling (cursor stays within the dwell radius of an anchor) -> Fired
(dwell time reached) -> Cooldown (no new dwell at the same spot for 500 ms).
Leaving the radius at any time returns to Idle and cancels the ring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gazeos.types import ControlState

FloatArray = NDArray[np.float64]


@dataclass
class DwellUpdate:
    state: ControlState
    progress: float  # 0..1, for the progress ring
    fired: bool = False
    anchor: FloatArray | None = None


class DwellClicker:
    def __init__(self, dwell_ms: float = 800.0, radius_px: float = 60.0, cooldown_ms: float = 500.0) -> None:
        self.dwell_s = dwell_ms / 1000.0
        self.radius_px = radius_px
        self.cooldown_s = cooldown_ms / 1000.0
        self.state = ControlState.IDLE
        self._anchor: FloatArray | None = None
        self._start: float = 0.0
        self._fired_at: float = -1e9
        self._fired_point: FloatArray | None = None
        self._points: list[FloatArray] = []

    def reset(self) -> None:
        self.state = ControlState.IDLE
        self._anchor = None

    def rearm(self) -> None:
        """Allow a new dwell at the same spot once the cooldown time has passed
        (used when the magnifier opens right where the last dwell fired)."""
        self._fired_point = None

    def update(self, t: float, xy: FloatArray, enabled: bool = True) -> DwellUpdate:
        xy = np.asarray(xy, dtype=np.float64)
        if not enabled:
            self.reset()
            return DwellUpdate(self.state, 0.0)

        if self.state in (ControlState.FIRED, ControlState.COOLDOWN):
            near_fired = (
                self._fired_point is not None
                and float(np.linalg.norm(xy - self._fired_point)) <= self.radius_px
            )
            if t - self._fired_at < self.cooldown_s:
                self.state = ControlState.COOLDOWN
                return DwellUpdate(self.state, 0.0)
            if near_fired:
                # Stay in cooldown until the gaze leaves the spot that just fired,
                # so a long stare does not click again and again.
                self.state = ControlState.COOLDOWN
                return DwellUpdate(self.state, 0.0)
            self.state = ControlState.IDLE
            self._anchor = None

        if self._anchor is None or float(np.linalg.norm(xy - self._anchor)) > self.radius_px:
            self._anchor = xy.copy()
            self._start = t
            self._points = [xy.copy()]
            self.state = ControlState.IDLE
            return DwellUpdate(self.state, 0.0)

        self._points.append(xy.copy())
        elapsed = t - self._start
        if elapsed >= self.dwell_s:
            # Click at the median of the second half of the dwell: the first
            # samples may still be settling after the eye arrived.
            settled = self._points[len(self._points) // 2 :]
            anchor = np.median(np.stack(settled), axis=0)
            self.state = ControlState.FIRED
            self._fired_at = t
            self._fired_point = anchor.copy()
            self._anchor = None
            return DwellUpdate(self.state, 1.0, fired=True, anchor=anchor)
        self.state = ControlState.DWELLING
        return DwellUpdate(self.state, elapsed / self.dwell_s, anchor=self._anchor)
