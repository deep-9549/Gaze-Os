"""Calibration protocol: targets, sample windows, outlier rejection, validation (PRD 11.2)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.vision.geometry import vector_to_pitchyaw

FloatArray = NDArray[np.float64]


@dataclass
class GazeObservation:
    """What the network and front-end produce for one frame, before personalization."""

    pitchyaw: FloatArray  # (2,) camera-frame gaze, radians
    origin: FloatArray  # (3,) gaze origin, camera mm
    screen_px: FloatArray  # (2,) raw ray-screen intersection, pixels
    head_pyr: FloatArray = field(default_factory=lambda: np.zeros(3))  # radians
    features: FloatArray | None = None  # penultimate network features (for P4/P5)
    sigma: FloatArray | None = None  # (2,) predicted std of pitch/yaw, radians


@dataclass
class CalibrationSample:
    obs: GazeObservation
    target_px: FloatArray  # (2,)
    weight: float = 1.0
    t: float = 0.0


def grid_targets(geometry: ScreenGeometry, num_points: int = 9) -> list[tuple[float, float]]:
    """9 points at 10/50/90% of width and height, or 16 points on a 4x4 grid."""
    if num_points == 9:
        fractions: Sequence[float] = (0.1, 0.5, 0.9)
    elif num_points == 16:
        fractions = (0.1, 0.1 + 0.8 / 3, 0.1 + 1.6 / 3, 0.9)
    else:
        raise ValueError("num_points must be 9 or 16")
    return [(fx * geometry.width_px, fy * geometry.height_px) for fy in fractions for fx in fractions]


def validation_targets(geometry: ScreenGeometry) -> list[tuple[float, float]]:
    """Four positions not used for calibration."""
    return [
        (fx * geometry.width_px, fy * geometry.height_px)
        for fx, fy in ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7))
    ]


def quick_targets(geometry: ScreenGeometry) -> list[tuple[float, float]]:
    """Single centre dot for one-point drift correction (PRD 11.6)."""
    return [(geometry.width_px / 2.0, geometry.height_px / 2.0)]


def shuffled(targets: list[tuple[float, float]], seed: int | None = None) -> list[tuple[float, float]]:
    out = list(targets)
    random.Random(seed).shuffle(out)
    return out


def target_radius(t_since_show: float, duration_s: float = 1.5, r_max: float = 40.0, r_min: float = 6.0) -> float:
    """Radius of the shrinking target circle at a given time (draws the eye to its centre)."""
    frac = min(max(t_since_show / duration_s, 0.0), 1.0)
    return r_max + (r_min - r_max) * frac


def in_sample_window(t_since_show: float, keep_from_s: float = 0.5, duration_s: float = 1.5) -> bool:
    """Keep frames from 0.5 to 1.5 s after the target appears (skips the eye's travel)."""
    return keep_from_s <= t_since_show <= duration_s


def mad_inliers(values: FloatArray, threshold: float = 3.0) -> NDArray[np.bool_]:
    """Mask of rows within `threshold` median absolute deviations of the median.

    values: (N, D). A row is an outlier if any dimension is beyond the threshold.
    """
    v = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if len(v) < 3:
        return np.ones(len(v), dtype=bool)
    med = np.median(v, axis=0)
    mad = np.median(np.abs(v - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-9, 1e-9, mad)
    return np.asarray(np.all(np.abs(v - med) <= threshold * mad, axis=1), dtype=np.bool_)


def filter_point_samples(
    samples: list[CalibrationSample], threshold: float = 3.0
) -> list[CalibrationSample]:
    """Drop outliers among the samples collected for one target."""
    if not samples:
        return []
    values = np.stack([s.obs.pitchyaw for s in samples])
    mask = mad_inliers(values, threshold)
    return [s for s, keep in zip(samples, mask, strict=True) if keep]


def true_pitchyaw(sample: CalibrationSample, geometry: ScreenGeometry) -> FloatArray:
    """Gaze direction from the eye to the target, in the camera frame."""
    target_mm = geometry.px_to_mm(float(sample.target_px[0]), float(sample.target_px[1]))
    target_cam = geometry.screen_mm_to_camera(*target_mm)
    return vector_to_pitchyaw(target_cam - sample.obs.origin)


@dataclass
class ValidationResult:
    per_point_px: list[float]
    per_point_deg: list[float]
    mean_px: float
    mean_deg: float
    mean_cm: float
    score: int  # 0-100, shown to the user

    @staticmethod
    def score_from_deg(mean_deg: float) -> int:
        """100 at <= 1 degree, 0 at >= 6 degrees, linear between."""
        return int(round(100.0 * min(max((6.0 - mean_deg) / 5.0, 0.0), 1.0)))


def validate(
    predicted_px: Sequence[Sequence[float]],
    target_px: Sequence[Sequence[float]],
    geometry: ScreenGeometry,
) -> ValidationResult:
    """Mean error per validation point, in pixels, degrees and centimetres (FR-10)."""
    errs_px = [
        float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(t, dtype=np.float64)))
        for p, t in zip(predicted_px, target_px, strict=True)
    ]
    errs_deg = [geometry.px_to_deg(e) for e in errs_px]
    mean_px = float(np.mean(errs_px)) if errs_px else 0.0
    mean_deg = float(np.mean(errs_deg)) if errs_deg else 0.0
    sx, sy = geometry.px_per_mm
    mean_cm = mean_px / ((sx + sy) / 2.0) / 10.0
    return ValidationResult(
        per_point_px=errs_px,
        per_point_deg=errs_deg,
        mean_px=mean_px,
        mean_deg=mean_deg,
        mean_cm=mean_cm,
        score=ValidationResult.score_from_deg(mean_deg),
    )
