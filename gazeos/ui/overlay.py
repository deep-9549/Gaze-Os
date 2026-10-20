"""Always-on-top, click-through overlay (PRD 7.4, FR-34, FR-17).

Draws the gaze cursor (a soft circle whose size shows confidence), the dwell
ring, the status chip (colour plus a text label, never colour alone), the
zoom-click magnifier and banners.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QFont, QGuiApplication, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from gazeos.control.engine import EngineOutput
from gazeos.control.zoom import ZoomRegion
from gazeos.types import TrackingStatus
from gazeos.ui.common import STATUS_COLORS, apply_no_activate, color, device_ratio, overlay_flags


class Overlay(QWidget):
    def __init__(self, theme_name: str = "default") -> None:
        super().__init__(None, overlay_flags())
        self.theme_name = theme_name
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.out: EngineOutput | None = None
        self.show_cursor = True
        self.banner: str | None = None
        self.toast: tuple[str, float] | None = None
        self._zoom_pixmap: QPixmap | None = None
        self._zoom_key: ZoomRegion | None = None
        self._ratio = 1.0
        self._repaint = QTimer(self)
        self._repaint.timeout.connect(self.update)
        self._repaint.start(16)  # ~60 Hz redraw (PRD 8.1)

    def showEvent(self, event: object) -> None:  # noqa: N802
        super().showEvent(event)  # type: ignore[arg-type]
        apply_no_activate(self, click_through=True)
        self._ratio = device_ratio(self)

    def set_output(self, out: EngineOutput) -> None:
        self.out = out
        if out.zoom is not None and out.zoom != self._zoom_key:
            self._grab_zoom(out.zoom)
        elif out.zoom is None:
            self._zoom_pixmap, self._zoom_key = None, None

    def show_toast(self, text: str, seconds: float = 4.0) -> None:
        self.toast = (text, time.monotonic() + seconds)

    def _grab_zoom(self, zoom: ZoomRegion) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        r = self._ratio
        # Hide ourselves from the grab by grabbing the desktop window behind the overlay.
        self._zoom_pixmap = screen.grabWindow(
            0, int(zoom.src_left / r), int(zoom.src_top / r), int(zoom.src_size / r), int(zoom.src_size / r)
        )
        self._zoom_key = zoom

    def _p(self, x: float, y: float) -> QPointF:
        """Physical pixels from the engine -> overlay logical coordinates."""
        g = self.geometry()
        return QPointF(x / self._ratio - g.x(), y / self._ratio - g.y())

    def paintEvent(self, event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        out = self.out
        th = self.theme_name
        if out is not None and out.zoom is not None and self._zoom_pixmap is not None:
            z = out.zoom
            tl = self._p(z.view_left, z.view_top)
            size = z.view_size / self._ratio
            target = QRectF(tl.x(), tl.y(), size, size)
            painter.drawPixmap(target, self._zoom_pixmap, QRectF(self._zoom_pixmap.rect()))
            painter.setPen(QPen(color(th, "accent"), 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(target)

        if out is not None and out.cursor is not None and self.show_cursor:
            c = self._p(*out.cursor)
            radius = max(10.0, min(out.sigma_px / self._ratio, 120.0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color(th, "cursor", 60 if not out.paused else 25)))
            painter.drawEllipse(c, radius, radius)
            painter.setBrush(QBrush(color(th, "cursor", 200 if not out.paused else 80)))
            painter.drawEllipse(c, 4.0, 4.0)
            if out.dwell_progress > 0 and not out.paused:
                ring = 26.0
                pen = QPen(color(th, "ring"), 6)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                rect = QRectF(c.x() - ring, c.y() - ring, 2 * ring, 2 * ring)
                # Fills clockwise from 12 o'clock (Qt angles are 1/16 degree, counter-clockwise).
                painter.drawArc(rect, 90 * 16, int(-360 * 16 * out.dwell_progress))

        self._draw_chip(painter)
        if self.banner:
            self._draw_text_box(painter, self.banner, top=True, key="warn")
        if self.toast and time.monotonic() < self.toast[1]:
            self._draw_text_box(painter, self.toast[0], top=False, key="accent")
        painter.end()

    def _draw_chip(self, painter: QPainter) -> None:
        out = self.out
        status = out.status if out is not None else TrackingStatus.NO_FACE
        if out is not None and out.paused:
            status = TrackingStatus.PAUSED
        label = status.value
        if out is not None and not out.paused:
            label += f" · {out.action.value.title()}" + (" (dragging)" if out.dragging else "")
        key = STATUS_COLORS.get(status, "muted")
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        painter.setFont(font)
        w = painter.fontMetrics().horizontalAdvance(label) + 44
        x = (self.width() - w) / 2
        rect = QRectF(x, 8, w, 34)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color(self.theme_name, "bg", 220)))
        painter.drawRoundedRect(rect, 17, 17)
        painter.setBrush(QBrush(color(self.theme_name, key)))
        painter.drawEllipse(QPointF(x + 18, 25), 7, 7)
        painter.setPen(color(self.theme_name, "fg"))
        painter.drawText(rect.adjusted(30, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter, label)

    def _draw_text_box(self, painter: QPainter, text: str, top: bool, key: str) -> None:
        font = QFont()
        font.setPointSize(15)
        painter.setFont(font)
        w = painter.fontMetrics().horizontalAdvance(text) + 40
        y = 52 if top else self.height() - 90
        rect = QRectF((self.width() - w) / 2, y, w, 44)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color(self.theme_name, key)))
        painter.drawRoundedRect(rect, 10, 10)
        painter.setPen(color(self.theme_name, "accent_fg"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
