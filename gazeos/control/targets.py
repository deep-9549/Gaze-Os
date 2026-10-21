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
        return min(hits, key=lambda t: t.rect.area)  # innermost wins

    def all(self) -> list[GazeTarget]:
        with self._lock:
            return list(self._targets.values())
