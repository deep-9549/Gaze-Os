"""Opt-in experiment logging and gaze heatmaps (FR-38, FR-41, PRD 21.1-21.2).

Off by default. Typed text is never logged while the focused field is a
password field.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from numpy.typing import NDArray


class ExperimentLogger:
    def __init__(self, directory: str | Path, session: str | None = None) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        name = session or time.strftime("session-%Y%m%d-%H%M%S")
        self.path = self.dir / f"{name}.jsonl"
        self._f: TextIO | None = self.path.open("a", encoding="utf-8")

    def log(self, kind: str, **fields: Any) -> None:
        if self._f is None:
            return
        rec = {"t": time.monotonic(), "wall": time.time(), "kind": kind, **fields}
        self._f.write(json.dumps(rec, default=_jsonable) + "\n")

    def log_frame(self, t: float, gaze_px: tuple[int, int] | None, state: str, status: str,
                  timings_ms: dict[str, float], sigma_px: float) -> None:
        self.log("frame", t_capture=t, gaze=gaze_px, state=state, status=status,
                 timings=timings_ms, sigma_px=round(sigma_px, 2))

    def log_text(self, text: str, in_password_field: bool) -> None:
        if in_password_field:
            self.log("text", text=None, redacted=True, length=len(text))
        else:
            self.log("text", text=text)

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def _jsonable(o: object) -> object:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def read_log(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


class HeatmapRecorder:
    """Accumulates gaze points into a coarse grid; stored locally only."""

    def __init__(self, width: int, height: int, cell_px: int = 16) -> None:
        self.cell = cell_px
        self.grid = np.zeros((height // cell_px + 1, width // cell_px + 1), dtype=np.float64)

    def add(self, x: float, y: float, weight: float = 1.0) -> None:
        r, c = int(y) // self.cell, int(x) // self.cell
        if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
            self.grid[r, c] += weight

    def normalized(self) -> NDArray[np.float64]:
        from scipy.ndimage import gaussian_filter

        g = gaussian_filter(self.grid, sigma=2.0)
        m = g.max()
        return np.asarray(g / m if m > 0 else g, dtype=np.float64)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p.with_suffix(".npy"), self.grid)
        try:
            import cv2

            gray = (self.normalized() * 255).astype(np.uint8)
            img = cv2.applyColorMap(cv2.resize(gray, None, fx=self.cell, fy=self.cell), cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(p.with_suffix(".png")), img)
        except Exception:
            pass
        return p
