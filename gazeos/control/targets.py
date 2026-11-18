"""Registry of GazeOS's own on-screen targets (keyboard keys, action bar, menus).

Dwell on one of these calls its callback directly instead of injecting an OS
click, so GazeOS windows never steal focus from the app being typed into
(PRD 7.5, 20.2), on every platform.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from gazeos.types import Rect


@dataclass
class GazeTarget:
    key: str
    rect: Rect
    callback: Callable[[], None]
    group: str = ""
    dwell_ms: float | None = None  # per-target dwell override
    # Pressed again while the gaze stays on it (typing keys: the second l of "hello", holding Del).
    repeat: bool = False
    # Dwell time as a multiple of the user's: longer for keys whose press is hard to undo (Enter).
    dwell_factor: float = 1.0
    # Which window is in front where two GazeOS windows overlap (higher = in front).
    layer: int = 0


def covered_fraction(a: Rect, b: Rect) -> float:
    """How much of `a` lies under `b` (0..1)."""
    w = min(a.left + a.width, b.left + b.width) - max(a.left, b.left)
    h = min(a.top + a.height, b.top + b.height) - max(a.top, b.top)
    return (w * h) / a.area if w > 0 and h > 0 and a.area > 0 else 0.0


class TargetRegistry:
    def __init__(self) -> None:
        self._targets: dict[str, GazeTarget] = {}
        self._lock = threading.Lock()

    def register(self, target: GazeTarget) -> None:
        with self._lock:
            self._targets[target.key] = target

    def unregister(self, key: str) -> None:
        with self._lock:
            self._targets.pop(key, None)

    def clear_group(self, group: str) -> None:
        with self._lock:
            for k in [k for k, t in self._targets.items() if t.group == group]:
                del self._targets[k]

    def hit(self, x: float, y: float) -> GazeTarget | None:
        with self._lock:
            hits = [t for t in self._targets.values() if t.rect.contains(x, y)]
        if not hits:
            return None
        # The window in front wins, then the innermost target: a side-bar button behind the
        # keyboard was smaller than the key over it and took its presses (18 Nov 2026).
        return max(hits, key=lambda t: (t.layer, -t.rect.area))

    def all(self) -> list[GazeTarget]:
        with self._lock:
            return list(self._targets.values())
