"""Gaze ray to screen point and unit conversions (PRD 11.3, 12.5).

Screen coordinates are as the user sees the screen: origin at the top-left,
x to the right, y down, in millimetres or pixels. The camera sits at the
top centre of the screen, in the screen plane, offset by `camera_offset_mm`.
Because the camera faces the user, camera +x is screen -x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gazeos.settings import ScreenSettings

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ScreenGeometry:
    width_px: int
    height_px: int
    width_mm: float
    height_mm: float
    camera_offset_mm: tuple[float, float, float] = (0.0, -10.0, 0.0)
    viewing_distance_mm: float = 600.0

    @classmethod
    def from_settings(cls, s: ScreenSettings) -> ScreenGeometry:
        return cls(
            width_px=s.width_px,
            height_px=s.height_px,
            width_mm=s.width_mm,
            height_mm=s.height_mm,
            camera_offset_mm=s.camera_offset_mm,
            viewing_distance_mm=s.viewing_distance_mm,
        )

    @property
    def px_per_mm(self) -> tuple[float, float]:
        return self.width_px / self.width_mm, self.height_px / self.height_mm

    @property
    def camera_screen_mm(self) -> tuple[float, float]:
        """Camera position in screen millimetre coordinates."""
        return self.width_mm / 2.0 + self.camera_offset_mm[0], self.camera_offset_mm[1]

    def camera_to_screen_mm(self, p_cam: FloatArray) -> tuple[float, float]:
        cx, cy = self.camera_screen_mm
        return cx - float(p_cam[0]), cy + float(p_cam[1])

    def screen_mm_to_camera(self, x_mm: float, y_mm: float) -> FloatArray:
        cx, cy = self.camera_screen_mm
        return np.array([cx - x_mm, y_mm - cy, self.camera_offset_mm[2]], dtype=np.float64)

    def mm_to_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        sx, sy = self.px_per_mm
        return x_mm * sx, y_mm * sy

    def px_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        sx, sy = self.px_per_mm
        return x_px / sx, y_px / sy

    def intersect(self, origin: FloatArray, gaze: FloatArray) -> tuple[float, float] | None:
        """Intersect the gaze ray o + t g with the screen plane z = camera z.

        Returns the screen point in millimetres, or None if the ray points away
        from the screen.
        """
        plane_z = self.camera_offset_mm[2]
        if gaze[2] >= -1e-9:
            return None
        t = (plane_z - origin[2]) / gaze[2]
        if t <= 0:
            return None
        p = origin + t * gaze
        return self.camera_to_screen_mm(p)

    def gaze_to_px(self, origin: FloatArray, gaze: FloatArray) -> tuple[float, float] | None:
        mm = self.intersect(origin, gaze)
        return None if mm is None else self.mm_to_px(*mm)

    def deg_to_px(self, degrees: float, distance_mm: float | None = None) -> float:
        """Approximate on-screen size in pixels of an angle at the viewing distance."""
        d = distance_mm or self.viewing_distance_mm
        mm = d * math.tan(math.radians(degrees))
        sx, sy = self.px_per_mm
        return mm * (sx + sy) / 2.0

    def px_to_deg(self, pixels: float, distance_mm: float | None = None) -> float:
        d = distance_mm or self.viewing_distance_mm
        sx, sy = self.px_per_mm
        mm = pixels / ((sx + sy) / 2.0)
        return math.degrees(math.atan2(mm, d))

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, 0.0), self.width_px - 1.0),
            min(max(y, 0.0), self.height_px - 1.0),
        )
