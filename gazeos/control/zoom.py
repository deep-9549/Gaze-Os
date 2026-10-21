"""Zoom-click geometry (PRD 7.3, FR-17).

A magnifier shows the region around the gaze at `factor` x; a dwell inside it
selects a point that maps back to the exact original screen location.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoomRegion:
    # Source region on the real screen.
    src_left: float
    src_top: float
    src_size: float
    # Where the magnified view is drawn.
    view_left: float
    view_top: float
    factor: float

    @property
    def view_size(self) -> float:
        return self.src_size * self.factor

    def view_to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.src_left + (x - self.view_left) / self.factor,
            self.src_top + (y - self.view_top) / self.factor,
        )

    def screen_to_view(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.view_left + (x - self.src_left) * self.factor,
            self.view_top + (y - self.src_top) * self.factor,
        )

    def in_view(self, x: float, y: float) -> bool:
        return (
            self.view_left <= x <= self.view_left + self.view_size
            and self.view_top <= y <= self.view_top + self.view_size
        )


def make_zoom_region(
    center: tuple[float, float],
    screen_w: int,
    screen_h: int,
    src_size: float = 240.0,
    factor: float = 3.0,
) -> ZoomRegion:
    """Centre the source region on the gaze and the magnified view on the screen
    as close to the gaze as fits."""
    half = src_size / 2.0
    src_left = min(max(center[0] - half, 0.0), screen_w - src_size)
    src_top = min(max(center[1] - half, 0.0), screen_h - src_size)
    view = src_size * factor
    view = min(view, float(min(screen_w, screen_h)))
    factor = view / src_size
    view_left = min(max(center[0] - view / 2.0, 0.0), screen_w - view)
    view_top = min(max(center[1] - view / 2.0, 0.0), screen_h - view)
    return ZoomRegion(src_left, src_top, src_size, view_left, view_top, factor)
