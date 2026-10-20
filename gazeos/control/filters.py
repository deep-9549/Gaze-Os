"""Cursor smoothing filters (PRD 12.2, 12.3; compared in experiment E7)."""

from __future__ import annotations

import math
from collections import deque
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from gazeos.settings import FilterSettings

FloatArray = NDArray[np.float64]


class PointFilter(Protocol):
    def __call__(self, t: float, xy: FloatArray, sigma_px: float | None = None) -> FloatArray: ...
    def reset(self) -> None: ...


class NoFilter:
    def __call__(self, t: float, xy: FloatArray, sigma_px: float | None = None) -> FloatArray:
        return np.asarray(xy, dtype=np.float64)

    def reset(self) -> None:
        pass


class MovingAverage:
    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._buf: deque[FloatArray] = deque(maxlen=window)

    def __call__(self, t: float, xy: FloatArray, sigma_px: float | None = None) -> FloatArray:
        self._buf.append(np.asarray(xy, dtype=np.float64))
        return np.asarray(np.mean(np.stack(self._buf), axis=0), dtype=np.float64)

    def reset(self) -> None:
        self._buf.clear()


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter1D:
    """One Euro filter (Casiez, Roussel and Vogel, CHI 2012)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    def __call__(self, t: float, x: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        if dt <= 0:
            return self._x
        self._t = t
        dx = (x - self._x) / dt
        self._dx = self._dx + _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x = self._x + _alpha(cutoff, dt) * (x - self._x)
        return self._x

    def reset(self) -> None:
        self._x, self._t, self._dx = None, None, 0.0


class OneEuroFilter:
    """Applied separately to screen x and y.

    `units_per_px` converts pixels to the units the parameters are tuned in.
    The engine passes 1 / (pixels per degree), so beta is per degree/second and
    the same settings work on any screen size or resolution.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0,
                 units_per_px: float = 1.0) -> None:
        self.fx = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.k = units_per_px

    def __call__(self, t: float, xy: FloatArray, sigma_px: float | None = None) -> FloatArray:
        x = self.fx(t, float(xy[0]) * self.k)
        y = self.fy(t, float(xy[1]) * self.k)
        return np.array([x / self.k, y / self.k], dtype=np.float64)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()


class KalmanFilter2D:
    """Constant-velocity Kalman filter with state (x, y, vx, vy).

    With `uncertainty_aware=True` the measurement noise for each frame comes
    from the gaze network's predicted variance, so uncertain frames move the
    cursor less (PRD 12.3).
    """

    def __init__(
        self,
        process_noise: float = 2000.0,
        measurement_noise_px: float = 40.0,
        uncertainty_aware: bool = False,
    ) -> None:
        self.q = process_noise
        self.r = measurement_noise_px
        self.uncertainty_aware = uncertainty_aware
        self.x: FloatArray | None = None
        self.P = np.eye(4)
        self._t: float | None = None
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    def __call__(self, t: float, xy: FloatArray, sigma_px: float | None = None) -> FloatArray:
        z = np.asarray(xy, dtype=np.float64)
        if self.x is None or self._t is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([self.r**2, self.r**2, 1e6, 1e6])
            self._t = t
            return z.copy()
        dt = max(t - self._t, 1e-4)
        self._t = t
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        # Discrete white-noise acceleration model.
        g = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        Q = g @ g.T * self.q**2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        r = self.r
        if self.uncertainty_aware and sigma_px is not None and sigma_px > 0:
            r = float(sigma_px)
        R = np.eye(2) * r**2
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return np.asarray(self.x[:2].copy(), dtype=np.float64)

    def reset(self) -> None:
        self.x = None
        self._t = None


def make_filter(s: FilterSettings, px_per_deg: float = 1.0) -> PointFilter:
    if s.kind == "none":
        return NoFilter()
    if s.kind == "moving_average":
        return MovingAverage(s.moving_average_window)
    if s.kind == "one_euro":
        return OneEuroFilter(s.min_cutoff_hz, s.beta, s.d_cutoff_hz, units_per_px=1.0 / px_per_deg)
    if s.kind == "kalman":
        return KalmanFilter2D(s.kalman_process_noise, s.kalman_measurement_noise_px)
    return KalmanFilter2D(s.kalman_process_noise, s.kalman_measurement_noise_px, uncertainty_aware=True)
