"""Pixel mapping: clamping and magnetic screen edges (PRD 12.5)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def magnetic_edges(xy: FloatArray, width: int, height: int, margin_px: float) -> FloatArray:
    """Clamp to the screen; within `margin_px` of an edge, snap onto that edge.

    Makes the taskbar and window close buttons easy to hit.
    """
    x = float(min(max(xy[0], 0.0), width - 1.0))
    y = float(min(max(xy[1], 0.0), height - 1.0))
    if x < margin_px:
        x = 0.0
    elif x > width - 1.0 - margin_px:
        x = width - 1.0
    if y < margin_px:
        y = 0.0
    elif y > height - 1.0 - margin_px:
        y = height - 1.0
    return np.array([x, y], dtype=np.float64)


def make_dpi_aware() -> None:
    """Per-monitor DPI awareness on Windows so cursor pixels are physical (PRD 20.2)."""
    import sys

    if sys.platform != "win32":
        return
    import ctypes

    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
