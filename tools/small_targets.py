"""Small-target test: can GazeOS hit Windows-sized buttons (snapping and zoom-click, PRD 7.3)?

A full-screen window shows a toolbar row of seven real buttons of one size (16, 24, 32 or
48 px, logical; Windows toolbar icons are 16-32) and highlights one. Look at it and let
GazeOS click as usual. Real buttons are visible to UI Automation, so snapping can find
them (the circles of tools.fitts_task are not). The window covers the screen and stays on
top, so every click lands inside it; Esc ends the test.

Per trial: size, whether the first click hit the target, wrong clicks, time to the hit
(20 s without one counts as a miss). Run GazeOS without --dry-run alongside:

    python -m tools.small_targets --out data/small_targets.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

SIZES = (16, 24, 32, 48)
BUTTONS = 7
GAP = 8  # px between toolbar buttons
TIMEOUT_S = 20.0


def make_trials(per_size: int = 6, seed: int = 0) -> list[dict[str, Any]]:
    """Each size `per_size` times in random order, with a random target index and toolbar
    position (as fractions of the screen, away from the edges)."""
    rng = random.Random(seed)
    trials = [{"size": s} for s in SIZES for _ in range(per_size)]
    rng.shuffle(trials)
    for t in trials:
        t.update(target=rng.randrange(BUTTONS), fx=rng.uniform(0.15, 0.7), fy=rng.uniform(0.15, 0.8))
    return trials


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in SIZES:
        rows = [r for r in results if r["size"] == s]
        if not rows:
            continue
        hits = [r for r in rows if r["hit"]]
        out[str(s)] = {
            "trials": len(rows),
            "first_click_correct": sum(r["first_click_correct"] for r in rows) / len(rows),
            "hit_within_timeout": len(hits) / len(rows),
            "wrong_clicks_per_trial": sum(r["wrong_clicks"] for r in rows) / len(rows),
            "mean_time_s": sum(r["time_s"] for r in hits) / len(hits) if hits else None,
        }
    return out


class SmallTargetTest(QWidget):
    def __init__(self, trials: list[dict[str, Any]], out: Path | None = None) -> None:
        super().__init__(None, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.trials, self.out = trials, out
        self.results: list[dict[str, Any]] = []
        self.buttons: list[QPushButton] = []
        self.info = QLabel(self)
        self.info.setStyleSheet("font-size: 18px; color: #222;")
        self.setStyleSheet("background: #C9CBD1;")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._check_timeout)
        self.timer.start(200)
        self.i = -1

    def start(self) -> None:
        self._next()

    def _next(self) -> None:
        for b in self.buttons:
            b.deleteLater()
        self.buttons = []
        self.i += 1
        if self.i >= len(self.trials):
            self.finish()
            return
        t = self.trials[self.i]
        size = t["size"]
        x0, y0 = int(t["fx"] * self.width()), int(t["fy"] * self.height())
        for k in range(BUTTONS):
            b = QPushButton("", self)
            b.setObjectName(f"button{k}")
            b.setAccessibleName(f"button {k}")
            b.setGeometry(x0 + k * (size + GAP), y0, size, size)
            target = k == t["target"]
            b.setStyleSheet(f"background: {'#E06C00' if target else '#5A6270'}; border: none;")
            b.clicked.connect(lambda _=False, k=k: self._clicked(k))
            b.show()
            self.buttons.append(b)
        self.info.setText(f"Trial {self.i + 1} of {len(self.trials)}: click the orange button ({size} px). Esc ends.")
        self.info.adjustSize()
        self.info.move(20, 20)
        self.t_start, self.wrong, self.first = time.monotonic(), 0, None

    def _clicked(self, k: int) -> None:
        hit = k == self.trials[self.i]["target"]
        if self.first is None:
            self.first = hit
        if not hit:
            self.wrong += 1
            return
        self._record(hit=True)

    def _check_timeout(self) -> None:
        if 0 <= self.i < len(self.trials) and time.monotonic() - self.t_start > TIMEOUT_S:
            self._record(hit=False)

    def _record(self, hit: bool) -> None:
        t = self.trials[self.i]
        self.results.append({"size": t["size"], "target": t["target"], "hit": hit,
                             "first_click_correct": bool(self.first), "wrong_clicks": self.wrong,
                             "time_s": time.monotonic() - self.t_start})
        self._next()

    def finish(self) -> None:
        self.timer.stop()
        report = {"results": self.results, "summary": summarize(self.results)}
        if self.out is not None:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report["summary"], indent=2))
        self.close()

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-size", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/small_targets.json")
    a = ap.parse_args()
    app = QApplication([])
    w = SmallTargetTest(make_trials(a.per_size, a.seed), Path(a.out))
    w.showFullScreen()
    QTimer.singleShot(300, w.start)  # after the window has its full-screen size
    app.exec()


if __name__ == "__main__":
    main()
