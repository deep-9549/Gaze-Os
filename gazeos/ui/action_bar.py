"""Slim dockable action bar: Left, Right, Double, Drag, Scroll, Keyboard, Shortcuts, Pause (PRD 7.4)."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QBoxLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from gazeos.control.engine import Action
from gazeos.ui.common import apply_no_activate, panel_flags, stylesheet


class ActionBar(QWidget):
    def __init__(
        self,
        on_action: Callable[[Action], None],
        on_keyboard: Callable[[], None],
        on_shortcuts: Callable[[], None],
        on_pause: Callable[[], None],
        on_menu: Callable[[], None],
        theme_name: str = "default",
        dock: str = "right",
    ) -> None:
        super().__init__(None, panel_flags())
        self.setStyleSheet(stylesheet(theme_name))
        self.dock = dock
        vertical = dock in ("left", "right")
        layout: QBoxLayout = QVBoxLayout(self) if vertical else QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.buttons: dict[str, QPushButton] = {}
        for action in Action:
            b = self._button(action.value.title(), f"action_{action.value}")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, a=action: on_action(a))
            layout.addWidget(b)
            self.buttons[action.value] = b
        for label, name, cb in (("Keyboard", "keyboard", on_keyboard), ("Shortcuts", "shortcuts", on_shortcuts),
                                ("Pause", "pause", on_pause), ("Menu", "menu", on_menu)):
            b = self._button(label, name)
            b.clicked.connect(lambda _=False, f=cb: f())
            layout.addWidget(b)
            self.buttons[name] = b
        self.set_action(Action.LEFT)
        self._place()

    def _button(self, label: str, name: str) -> QPushButton:
        b = QPushButton(label)
        b.setObjectName(name)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setMinimumSize(110, 64)
        return b

    def _place(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        g = screen.availableGeometry()
        self.adjustSize()
        if self.dock == "right":
            self.move(g.right() - self.width(), g.top() + (g.height() - self.height()) // 2)
        elif self.dock == "left":
            self.move(g.left(), g.top() + (g.height() - self.height()) // 2)
        else:
            self.move(g.left() + (g.width() - self.width()) // 2, g.bottom() - self.height())

    def showEvent(self, event: object) -> None:  # noqa: N802
        super().showEvent(event)  # type: ignore[arg-type]
        apply_no_activate(self)

    def set_action(self, action: Action) -> None:
        for a in Action:
            self.buttons[a.value].setChecked(a == action)

    def set_paused(self, paused: bool) -> None:
        self.buttons["pause"].setText("Resume" if paused else "Pause")
