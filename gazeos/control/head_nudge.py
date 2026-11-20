"""Head nudge: the eyes place the cursor, a small head movement moves it the last bit.

Why (measured 20 Nov 2026 on the developer's 9-dot recordings, `D:\\GazeOS-cache\\calib_study`):
after calibrating, webcam gaze is about 16 mm off on average (1.5 degrees), a calibration from
another sitting about 28 mm, and the offset wanders by 1-2 degrees within minutes even with
the head still. No calibration method tried (screen or angle space, head pose as an input)
did better. A key is about 31 mm, so the cursor will often sit next to the key looked at.

The head signal is the nose tip in the camera image, not the head angles: its frame-to-frame
jitter is 0.32 px (about 0.13 degrees of head rotation), against 0.5-1.2 degrees for the head
angles from the face model. Movement is measured in eye distances (the pixel distance between
the outer eye corners), so the gain does not depend on how far the user sits.

While the eyes hold still on something, the nose's movement since the gaze arrived moves the
cursor (`gain` screen widths per eye distance). Sway is absorbed by a dead band: on recorded
fixations the cursor then moves under a third of a key (95th percentile) without any intent,
while a key's worth of correction takes about a 2.5 degree tilt. The reference is taken
`settle_s` after the gaze arrives, since the head keeps turning toward a new target for a
moment after the eyes land.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gazeos.control.filters import OneEuroFilter

FloatArray = NDArray[np.float64]


class HeadNudge:
    def __init__(
        self,
        screen_width_px: float,
        max_px: float,
        gain: float = 1.5,
        deadband: float = 0.010,
        settle_s: float = 0.4,
        min_cutoff_hz: float = 0.5,
    ) -> None:
        self.px_per_unit = gain * screen_width_px
        self.max_px = max_px
        self.deadband = deadband
        self.settle_s = settle_s
        self._filter = OneEuroFilter(min_cutoff_hz, 0.05, 1.0)
        self._anchor: FloatArray | None = None  # filtered nose position the offset is measured from
        self._since: float | None = None  # when the gaze arrived (the reference is taken settle_s later)
        self._base = np.zeros(2)  # offset carried over from before a re-anchor (kept after a click)
        self.offset = np.zeros(2)  # current cursor offset, screen px
        self.speed_px_s = 0.0
        self._prev: tuple[float, FloatArray] | None = None

    def reset(self) -> None:
        """The gaze jumped somewhere new (or tracking was lost): start again from the gaze."""
        self._anchor = None
        self._since = None
        self._base = np.zeros(2)
        self.offset = np.zeros(2)
        self.speed_px_s = 0.0
        self._prev = None

    def rebase(self) -> None:
        """Keep the cursor where it is but measure further nudges from the current head position
        (after a click, whose nudge has been learned into the correction)."""
        self._base = np.zeros(2)
        self.offset = np.zeros(2)
        self._anchor = None
        self._since = None

    def update(self, t: float, nose_px: tuple[float, float], eye_distance_px: float) -> FloatArray:
        """The cursor offset in screen pixels for this frame."""
        p = np.asarray(self._filter(t, np.asarray(nose_px, dtype=np.float64)), dtype=np.float64)
        if self._since is None:
            self._since = t
        if self._anchor is None or t - self._since < self.settle_s:
            # Still settling: follow the head, so the turn that goes with the eye movement
            # does not push the cursor past the target.
            self._anchor = p.copy()
            off = self._base.copy()
        else:
            d = (p - self._anchor) / max(eye_distance_px, 1.0)
            n = float(np.linalg.norm(d))
            d = d * max(0.0, 1.0 - self.deadband / n) if n > 0 else d
            # The camera faces the user: the nose moving to the image's left is the user's right.
            off = self._base + np.array([-d[0], d[1]]) * self.px_per_unit
        size = float(np.linalg.norm(off))
        if size > self.max_px:
            off = off * (self.max_px / size)
        if self._prev is not None and t > self._prev[0]:
            inst = float(np.linalg.norm(off - self._prev[1])) / (t - self._prev[0])
            self.speed_px_s = 0.7 * self.speed_px_s + 0.3 * inst
        self._prev = (t, off.copy())
        self.offset = off
        return off
