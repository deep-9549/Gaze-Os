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


class DriftCorrector:
    """One-point correction: an additive screen-space offset (PRD 11.6)."""

    def __init__(self) -> None:
        self.offset = np.zeros(2)

    def fit(self, predicted_px: list[FloatArray], target_px: tuple[float, float]) -> FloatArray:
        if predicted_px:
            med = np.median(np.stack(predicted_px), axis=0)
            self.offset = np.asarray(target_px, dtype=np.float64) - (med - self.offset)
        return self.offset

    def apply(self, px: FloatArray) -> FloatArray:
        return np.asarray(px, dtype=np.float64) + self.offset

    def reset(self) -> None:
        self.offset = np.zeros(2)
