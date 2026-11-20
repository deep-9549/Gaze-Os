"""Implicit (continuous) recalibration and drift correction (PRD 11.5, 11.6, FR-11, FR-12)."""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gazeos.calibration.personalization import Personalizer
from gazeos.calibration.protocol import CalibrationSample, GazeObservation

FloatArray = NDArray[np.float64]


@dataclass
class TimedObservation:
    t: float
    obs: GazeObservation


class GazeHistory:
    """Ring buffer of recent observations, so a click can look back 300 ms."""

    def __init__(self, seconds: float = 2.0) -> None:
        self.seconds = seconds
        self._items: deque[TimedObservation] = deque()

    def add(self, t: float, obs: GazeObservation) -> None:
        self._items.append(TimedObservation(t, obs))
        while self._items and t - self._items[0].t > self.seconds:
            self._items.popleft()

    def window(self, t_end: float, duration: float) -> list[GazeObservation]:
        return [it.obs for it in self._items if t_end - duration <= it.t <= t_end]


class ImplicitRecalibrator:
    """Turns confirmed clicks on known UI elements into weighted calibration samples."""

    def __init__(
        self,
        personalizer: Personalizer,
        base_samples: list[CalibrationSample],
        refit_every: int = 20,
        decay_minutes: float = 30.0,
        lookback_s: float = 0.3,
        background: bool = True,
        on_refit: Callable[[Personalizer], None] | None = None,
    ) -> None:
        self.personalizer = personalizer
        self.base_samples = list(base_samples)
        self.refit_every = refit_every
        self.decay_s = decay_minutes * 60.0
        self.lookback_s = lookback_s
        self.background = background
        self.on_refit = on_refit
        self.implicit: list[CalibrationSample] = []
        self._pending = 0
        self._lock = threading.Lock()
        self._residuals: deque[float] = deque(maxlen=50)
        self.baseline_error_px: float | None = None
        self.refits = 0

    def current_error_px(self) -> float:
        if self._residuals:
            return float(np.median(self._residuals))
        return self.baseline_error_px or 80.0

    def on_click(
        self, t: float, history: GazeHistory, element_center: tuple[float, float]
    ) -> int:
        """Add samples for a confirmed click; returns how many were accepted."""
        observations = history.window(t, self.lookback_s)
        if not observations:
            return 0
        target = np.asarray(element_center, dtype=np.float64)
        preds = np.stack([self.personalizer.predict_px(o) for o in observations])
        residual = float(np.linalg.norm(np.median(preds, axis=0) - target))
        # Reject clicks whose residual is beyond twice the current error:
        # the user probably clicked something they were not looking at.
        if residual > 2.0 * self.current_error_px():
            return 0
        self._residuals.append(residual)
        with self._lock:
            for o in observations:
                self.implicit.append(CalibrationSample(obs=o, target_px=target, t=t))
            self._pending += 1
            should_refit = self._pending >= self.refit_every
        if should_refit:
            self._pending = 0
            self.refit(t)
        return len(observations)

    def _weighted(self, now: float) -> list[CalibrationSample]:
        out = list(self.base_samples)
        for s in self.implicit:
            w = math.exp(-(now - s.t) / self.decay_s)
            if w > 0.02:
                out.append(CalibrationSample(obs=s.obs, target_px=s.target_px, weight=w, t=s.t))
        return out

    def refit(self, now: float) -> None:
        with self._lock:
            # Drop samples that no longer carry weight.
            self.implicit = [s for s in self.implicit if now - s.t < 4.0 * self.decay_s]
            samples = self._weighted(now)

        def job() -> None:
            self.personalizer.fit(samples)
            self.refits += 1
            if self.on_refit is not None:
                self.on_refit(self.personalizer)

        if self.background:
            threading.Thread(target=job, name="gazeos-refit", daemon=True).start()
        else:
            job()

    def drift_suspected(self) -> bool:
        """True when the running residual has grown more than 50% over the baseline."""
        if self.baseline_error_px is None or len(self._residuals) < 10:
            return False
        return self.current_error_px() > 1.5 * self.baseline_error_px


class CorrectionField:
    """A screen-space correction learned from head nudges: where the user moved the cursor by
    head before a press, the gaze was that far off there.

    Unlike labels taken from the cursor itself (ADR 0009: a press on a key only says the cursor
    was on it, so refitting on those pulled the bottom row 9-10 mm up), a nudge is the user's own
    correction, so it cannot reinforce the tracker's error. Webcam gaze error differs across the
    screen and changes within minutes (20 Nov 2026 analysis), so each correction counts near
    where it was made (Gaussian, `radius_frac` of the screen width) and fades with time
    (`half_life_s`). `prior` acts as that many corrections of zero at every point, so one
    nudge moves the field only part of the way and a stray one cannot move it far.
    """

    def __init__(self, screen_width_px: float, radius_frac: float = 0.25, half_life_s: float = 300.0,
                 prior: float = 0.5, max_samples: int = 60) -> None:
        self.sigma_px = radius_frac * screen_width_px
        self.half_life_s = half_life_s
        self.prior = prior
        self.max_samples = max_samples
        self._samples: list[tuple[float, FloatArray, FloatArray]] = []  # (t, where, correction)
        self.now = 0.0

    def __len__(self) -> int:
        return len(self._samples)

    def reset(self) -> None:
        self._samples = []

    def add(self, t: float, where_px: FloatArray, correction_px: FloatArray) -> None:
        self.now = max(self.now, t)
        self._samples.append((t, np.asarray(where_px, dtype=np.float64).copy(),
                              np.asarray(correction_px, dtype=np.float64).copy()))
        del self._samples[:-self.max_samples]

    def at(self, px: FloatArray) -> FloatArray:
        if not self._samples:
            return np.zeros(2)
        p = np.asarray(px, dtype=np.float64)
        num, den = np.zeros(2), self.prior
        for t, where, corr in self._samples:
            w = 0.5 ** ((self.now - t) / self.half_life_s)
            w *= math.exp(-float(np.sum((p - where) ** 2)) / (2.0 * self.sigma_px**2))
            num += w * corr
            den += w
        return num / den


@dataclass
class KeyPress:
    t: float
    name: str  # "key_d", "key_backspace", ...
    center: FloatArray  # the key's centre, screen px
    size: float  # the key's larger side, px
    where: FloatArray  # gaze before the learned correction during the dwell (median), px
    cursor: FloatArray  # cursor during the dwell (median), px


class MistakeLearner:
    """Learns where the cursor is off from the user's own corrections while typing.

    A letter pressed, deleted with one Del within `delete_within_s`, then a neighbouring letter
    pressed within `retype_within_s`: the user was looking at the second key while the cursor
    sat on the first. The cursor then needed to move from where it was during the first press to
    the second key's centre. On 20 Nov 2026 Fix cursor felt right for a few seconds and then the
    cursor slid onto neighbouring keys again; this follows the slide from the mistakes themselves.

    It cannot reinforce the tracker's error the way click labels did (ADR 0009): the label is the
    key the user chose to retype, not where the cursor was. Several Dels in a row, a key far from
    the deleted one, or the same key again teach nothing (editing, not a near miss).
    """

    def __init__(self, delete_within_s: float = 6.0, retype_within_s: float = 8.0, max_keys_apart: float = 1.6) -> None:
        self.delete_within_s = delete_within_s
        self.retype_within_s = retype_within_s
        self.max_keys_apart = max_keys_apart
        self._last: KeyPress | None = None  # the last letter pressed
        self._deleted: tuple[KeyPress, float] | None = None  # a letter deleted once, and when

    @staticmethod
    def is_letter(name: str) -> bool:
        return name.startswith("key_") and len(name) == 5 and name[4].isalpha()

    def on_press(self, p: KeyPress) -> tuple[FloatArray, FloatArray] | None:
        """Feed every keyboard press in order. Returns (where, correction) when a near miss was
        corrected: at gaze point `where` the cursor should move by `correction` px."""
        if p.name == "key_backspace":
            if self._deleted is not None:  # a second Del: editing, not a near miss
                self._deleted = None
            elif self._last is not None and p.t - self._last.t <= self.delete_within_s:
                self._deleted = (self._last, p.t)
            self._last = None
            return None
        out = None
        if self.is_letter(p.name):
            if self._deleted is not None:
                wrong, t_del = self._deleted
                apart = float(np.linalg.norm(p.center - wrong.center))
                if (p.name != wrong.name and p.t - t_del <= self.retype_within_s
                        and apart <= self.max_keys_apart * max(wrong.size, p.size)):
                    out = (wrong.where.copy(), p.center - wrong.cursor)
            self._last = p
        else:
            self._last = None
        self._deleted = None
        return out


class DriftCorrector:
    """One-point correction: an additive screen-space offset (PRD 11.6), plus the local
    correction learned from head nudges (`field`)."""

    def __init__(self, field: CorrectionField | None = None) -> None:
        self.offset = np.zeros(2)
        self.field = field

    def fit(self, predicted_px: list[FloatArray], target_px: tuple[float, float]) -> FloatArray:
        if predicted_px:
            med = np.median(np.stack(predicted_px), axis=0)
            self.offset = np.asarray(target_px, dtype=np.float64) - (med - self.offset)
        return self.offset

    def apply(self, px: FloatArray) -> FloatArray:
        out = np.asarray(px, dtype=np.float64) + self.offset
        if self.field is not None:
            out = out + self.field.at(out)
        return out

    def reset(self) -> None:
        self.offset = np.zeros(2)
        if self.field is not None:
            self.field.reset()
