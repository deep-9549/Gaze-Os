"""Small gaze-operable panels: main menu, shortcuts, confirmation, privacy (FR-26, US-15, PRD 21.1).

Every GazeOS button is pressed by looking at it for the dwell time, so whatever a person
glances at that long gets pressed. Two rules keep stray glances harmless:

- A panel closes itself when nothing in it was pressed for a while (`GazePanel`), so one
  opened by accident does not stay open.
- An action that cannot be undone by gaze asks in a `ConfirmPanel`: its Yes is well away from
  the button that asked, where the eye still is (No goes there), and no answer is No.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QHideEvent, QScreen, QShowEvent
from PySide6.QtWidgets import (
    QAbstractButton,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gazeos.profiles.store import Profile
from gazeos.ui.common import apply_no_activate, panel_flags, stylesheet

MENU_IDLE_S = 15.0  # the menu and the shortcuts close after this long without a press
READING_IDLE_S = 30.0  # panels with text to read (privacy) wait longer
CONFIRM_TIMEOUT_S = 10.0  # a confirmation not answered by then is answered No
# Space between a confirmation's No and Yes, and the least clearance (logical px) kept between
# Yes and the button that asked: far beyond how much a webcam gaze cursor wanders during one
# stare (1-2 degrees, under 120 px at 60 cm), so the stare that pressed it cannot reach Yes.
YES_GAP_PX = 300
MIN_YES_CLEARANCE_PX = 200


def global_rect(widget: QWidget) -> QRect:
    """A widget's rectangle in global logical coordinates (also while its window is hidden)."""
    return QRect(widget.mapToGlobal(QPoint(0, 0)), widget.size())


def rect_gap(a: QRect, b: QRect) -> float:
    """Distance between two rectangles' nearest edges; 0 when they touch or overlap."""
    dx = max(b.left() - (a.left() + a.width()), a.left() - (b.left() + b.width()), 0)
    dy = max(b.top() - (a.top() + a.height()), a.top() - (b.top() + b.height()), 0)
    return math.hypot(dx, dy)


def place_confirmation(avail: QRect, size: QSize, no: QRect, yes: QRect, avoid: QRect | None) -> QPoint:
    """Top-left corner on screen for a confirmation panel of `size` whose No and Yes buttons sit
    at `no` and `yes` inside it. No goes over `avoid`, the button that asked; where the screen is
    too narrow for Yes to stay `MIN_YES_CLEARANCE_PX` clear of it there, the panel goes to the
    top or bottom edge, whichever is further from it."""

    def clamp(p: QPoint) -> QPoint:
        x = min(max(p.x(), avail.left()), avail.left() + avail.width() - size.width())
        y = min(max(p.y(), avail.top()), avail.top() + avail.height() - size.height())
        return QPoint(x, y)

    if avoid is None:
        return clamp(QPoint(avail.left() + (avail.width() - size.width()) // 2,
                            avail.top() + (avail.height() - size.height()) // 2))
    beside = clamp(avoid.center() - no.center())
    if rect_gap(yes.translated(beside), avoid) >= MIN_YES_CLEARANCE_PX:
        return beside
    lower_half = avoid.center().y() > avail.center().y()
    edge = clamp(QPoint(beside.x(), avail.top() if lower_half else avail.top() + avail.height()))
    return max((beside, edge), key=lambda p: rect_gap(yes.translated(p), avoid))


class GazePanel(QWidget):
    """A GazeOS window that hides itself after `idle_s` seconds without a press (0: never).

    The countdown starts when the window is shown and restarts on every press of one of its
    buttons, so a panel opened by a stray glance does not stay open."""

    def __init__(self, flags: Qt.WindowType, theme_name: str, idle_s: float) -> None:
        super().__init__(None, flags)
        self.theme_name = theme_name
        self.setStyleSheet(stylesheet(theme_name))
        self.idle_timer = QTimer(self)
        self.idle_timer.setSingleShot(True)
        self.idle_timer.setInterval(round(idle_s * 1000))
        self.idle_timer.timeout.connect(self._on_idle)

    def watch_presses(self) -> None:
        """Restart the countdown on a press of any button in the panel (call once it is built)."""
        for b in self.findChildren(QAbstractButton):
            b.clicked.connect(lambda _=False: self.touch())

    def touch(self) -> None:
        """Something in the panel was used: count the idle time from now."""
        if self.isVisible() and self.idle_timer.interval() > 0:
            self.idle_timer.start()

    def _on_idle(self) -> None:
        self.hide()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if self.idle_timer.interval() > 0:
            self.idle_timer.start()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        super().hideEvent(event)
        self.idle_timer.stop()


class ButtonPanel(GazePanel):
    """A titled grid of large buttons plus Close, centred on screen; never takes focus."""

    def __init__(self, title: str, items: list[tuple[str, str, Callable[[], None]]], theme_name: str = "default",
                 columns: int = 3, close_after: bool = True, idle_s: float = MENU_IDLE_S) -> None:
        super().__init__(panel_flags(), theme_name, idle_s)
        root = QVBoxLayout(self)
        t = QLabel(title)
        t.setProperty("role", "title")
        root.addWidget(t)
        grid = QGridLayout()
        root.addLayout(grid)
        self.buttons: dict[str, QPushButton] = {}
        for i, (label, name, cb) in enumerate(items):
            b = QPushButton(label)
            b.setObjectName(name)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setMinimumSize(200, 80)

            def run(_: bool = False, f: Callable[[], None] = cb) -> None:
                if not self.isVisible():
                    return  # a press that arrives after the panel closed (a stale gaze target)
                if close_after:
                    self.hide()
                f()

            b.clicked.connect(run)
            grid.addWidget(b, i // columns, i % columns)
            self.buttons[name] = b
        close = QPushButton("Close")
        close.setObjectName("panel_close")
        close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close.clicked.connect(self.hide)
        grid.addWidget(close, (len(items) + columns - 1) // columns, 0, 1, columns)
        self.watch_presses()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        apply_no_activate(self)
        screen = self.screen()
        if screen is not None:
            g = screen.availableGeometry()
            self.adjustSize()
            self.move(g.left() + (g.width() - self.width()) // 2, g.top() + (g.height() - self.height()) // 2)


class ConfirmPanel(GazePanel):
    """A question a stray glance cannot answer Yes.

    `avoid` is the button that asked (global logical rectangle), where the eye still is: No is
    put over it and Yes `YES_GAP_PX` further on the side with more room, so the long stare that
    pressed the button cannot go on to press Yes. No answer within `timeout_s` is No."""

    def __init__(self, question: str, on_yes: Callable[[], None], theme_name: str = "default", *,
                 yes_label: str = "Yes", no_label: str = "No", detail: str = "",
                 on_no: Callable[[], None] | None = None, avoid: QRect | None = None,
                 timeout_s: float = CONFIRM_TIMEOUT_S) -> None:
        super().__init__(panel_flags(), theme_name, timeout_s)
        self.on_yes = on_yes
        self.on_no = on_no
        self.avoid = avoid
        self.answer: bool | None = None
        root = QVBoxLayout(self)
        title = QLabel(question)
        title.setProperty("role", "title")
        title.setWordWrap(True)
        root.addWidget(title)
        if detail:
            text = QLabel(detail)
            text.setWordWrap(True)
            root.addWidget(text)
        hint = QLabel(f"No answer within {timeout_s:g} s counts as No.")
        hint.setProperty("role", "muted")
        root.addWidget(hint)
        self.yes_btn = self._button(yes_label, "confirm_yes", True)
        self.no_btn = self._button(no_label, "confirm_no", False)
        row = QHBoxLayout()
        root.addLayout(row)
        yes_left = self._yes_goes_left()
        row.addWidget(self.yes_btn if yes_left else self.no_btn)
        row.addSpacing(YES_GAP_PX)
        row.addWidget(self.no_btn if yes_left else self.yes_btn)

    def _button(self, label: str, name: str, answer: bool) -> QPushButton:
        b = QPushButton(label)
        b.setObjectName(name)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setMinimumSize(200, 80)
        b.clicked.connect(lambda _=False: self._answer(answer))
        return b

    def _screen(self) -> QScreen | None:
        at = QGuiApplication.screenAt(self.avoid.center()) if self.avoid is not None else None
        return at or QGuiApplication.primaryScreen()

    def _yes_goes_left(self) -> bool:
        """Yes goes on whichever side of the asking button has more room on screen."""
        screen = self._screen()
        if self.avoid is None or screen is None:
            return True
        g = screen.availableGeometry()
        x = self.avoid.center().x()
        return x - g.left() >= g.left() + g.width() - x

    def yes_clearance(self) -> float:
        """Gap between Yes and the button that asked, in logical px (inf when none was given)."""
        return math.inf if self.avoid is None else rect_gap(global_rect(self.yes_btn), self.avoid)

    def _answer(self, yes: bool) -> None:
        if not self.isVisible():
            return  # a press that arrives after a timeout (a stale gaze target)
        self.answer = yes
        self.hide()
        callback = self.on_yes if yes else self.on_no
        if callback is not None:
            callback()

    def _on_idle(self) -> None:
        self._answer(False)

    def dismiss(self) -> None:
        """Take the question away without an answer (another one replaces it)."""
        self.hide()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        apply_no_activate(self)
        screen = self._screen()
        if screen is None:
            return
        self.adjustSize()
        layout = self.layout()
        if layout is not None:
            layout.activate()
        self.move(place_confirmation(screen.availableGeometry(), self.size(), self.no_btn.geometry(),
                                     self.yes_btn.geometry(), self.avoid))


class PrivacyPanel(GazePanel):
    """Lists exactly what is stored and offers 'Delete all my data', which asks in a separate
    `ConfirmPanel` first. `present` shows that panel (the app also makes it gaze-operable)."""

    def __init__(self, profile: Profile, on_deleted: Callable[[], None], theme_name: str = "default",
                 present: Callable[[ConfirmPanel], None] | None = None, idle_s: float = READING_IDLE_S) -> None:
        super().__init__(Qt.WindowType.WindowStaysOnTopHint, theme_name, idle_s)
        self.setWindowTitle("Privacy")
        self.profile = profile
        self.on_deleted = on_deleted
        self.present: Callable[[ConfirmPanel], None] = present or ConfirmPanel.show
        self.confirm: ConfirmPanel | None = None
        root = QVBoxLayout(self)
        title = QLabel("Your data stays on this computer")
        title.setProperty("role", "title")
        root.addWidget(title)
        root.addWidget(QLabel(
            "GazeOS makes no network calls by default. Camera frames are processed in memory and never saved.\n"
            "Research recording, experiment logs and heatmaps are off unless you turn them on."))
        self.items = QLabel()
        self.items.setProperty("role", "muted")
        root.addWidget(self.items)
        self.delete_btn = QPushButton("Delete all my data")
        self.delete_btn.setObjectName("privacy_delete")
        self.delete_btn.clicked.connect(self._ask_delete)
        root.addWidget(self.delete_btn)
        close = QPushButton("Close")
        close.setObjectName("privacy_close")
        close.clicked.connect(self.close)
        root.addWidget(close)
        self.refresh()
        self.watch_presses()

    def refresh(self) -> None:
        rows = self.profile.stored_items()
        text = "\n".join(f"• {label}: {path.name} ({size / 1024:.1f} KB)" for label, path, size in rows)
        self.items.setText(f"Stored for profile “{self.profile.name}”:\n{text or '• nothing yet'}")

    def _ask_delete(self) -> None:
        """Ask in a separate panel; this one hides meanwhile and comes back on No."""
        if not self.isVisible():
            return
        avoid = global_rect(self.delete_btn)
        self.hide()
        self.confirm = ConfirmPanel(
            "Delete all your GazeOS data?", self._delete, self.theme_name, yes_label="Yes, delete everything",
            detail=f"Settings, calibration, dictionary and phrases of profile “{self.profile.name}” "
                   "are removed for good.",
            on_no=self.show, avoid=avoid)
        self.present(self.confirm)

    def _delete(self) -> None:
        self.profile.delete_all()
        self.refresh()
        self.close()
        self.on_deleted()
