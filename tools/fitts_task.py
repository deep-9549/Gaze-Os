"""ISO 9241-411 multi-directional target-acquisition task (user-study task T2, PRD 17.3-17.4).

Shows 13 circular targets in a ring; the highlighted one must be selected with
whatever input is under test (GazeOS clicks arrive as ordinary mouse clicks,
so the same app measures the mouse baseline). Widths 32, 64 and 128 px.

    python -m tools.fitts_task --participant P01 --condition eyes_only --out data/study
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from research.metrics import fitts_throughput, iso_ring_targets

WIDTHS = (32, 64, 128)


class FittsTask(QWidget):
    def __init__(self, participant: str, condition: str, out: Path, radius: float, n: int = 13) -> None:
        super().__init__()
        self.participant, self.condition = participant, condition
        self.out = out
        self.radius, self.n = radius, n
        self.blocks = list(WIDTHS)
        self.width_px = self.blocks.pop(0)
        self.rows: list[dict[str, object]] = []
        self._new_block()

    def _new_block(self) -> None:
        c = (self.width() / 2 or 960, self.height() / 2 or 540)
        self.targets = iso_ring_targets(self.n, self.radius, c)
        self.i = 0
        self.t_last = time.monotonic()
        self.start = self.targets[0]

    def resizeEvent(self, _: object) -> None:  # noqa: N802
        self._new_block()

    def mousePressEvent(self, e) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        now = time.monotonic()
        x, y = e.position().x(), e.position().y()
        tx, ty = self.targets[self.i]
        hit = math.hypot(x - tx, y - ty) <= self.width_px / 2
        if self.i > 0:  # the first selection only starts the block
            self.rows.append({"participant": self.participant, "condition": self.condition,
                              "width": self.width_px, "trial": self.i, "start_x": self.start[0],
                              "start_y": self.start[1], "target_x": tx, "target_y": ty, "sel_x": x, "sel_y": y,
                              "time_s": now - self.t_last, "hit": int(hit)})
        self.start, self.t_last = (x, y), now
        self.i += 1
        if self.i >= len(self.targets):
            if self.blocks:
                self.width_px = self.blocks.pop(0)
                self._new_block()
            else:
                self.finish()
        self.update()

    def finish(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / f"fitts_{self.participant}_{self.condition}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        r = self.rows
        res = fitts_throughput([(q["start_x"], q["start_y"]) for q in r], [(q["target_x"], q["target_y"]) for q in r],
                               [(q["sel_x"], q["sel_y"]) for q in r], [float(q["time_s"]) for q in r],  # type: ignore[arg-type]
                               [bool(q["hit"]) for q in r])
        print(f"saved {path}; throughput {res.throughput_bps:.2f} bits/s, error rate {res.error_rate:.1%}")
        QApplication.quit()

    def keyPressEvent(self, e) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        if e.key() == Qt.Key.Key_Escape and self.rows:
            self.finish()

    def paintEvent(self, _: object) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#fcfcfb"))
        for j, (tx, ty) in enumerate(self.targets):
            active = j == self.i
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor("#2a78d6" if active else "#d6e6f8")))
            p.drawEllipse(QPointF(tx, ty), self.width_px / 2, self.width_px / 2)
        f = QFont()
        f.setPointSize(14)
        p.setFont(f)
        p.setPen(QColor("#52514e"))
        p.drawText(20, 30, f"Select the dark circle · width {self.width_px}px · {self.i}/{len(self.targets)}")
        p.end()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--out", default="data/study")
    ap.add_argument("--radius", type=float, default=350.0)
    a = ap.parse_args()
    app = QApplication(sys.argv)
    w = FittsTask(a.participant, a.condition, Path(a.out), a.radius)
    w.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
