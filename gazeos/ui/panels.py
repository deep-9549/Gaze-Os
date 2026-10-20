"""Small gaze-operable panels: main menu, shortcuts, privacy (FR-26, US-15, PRD 21.1)."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gazeos.profiles.store import Profile
from gazeos.ui.common import apply_no_activate, panel_flags, stylesheet


class ButtonPanel(QWidget):
    """A grid of large buttons; never takes focus."""

    def __init__(self, title: str, items: list[tuple[str, str, Callable[[], None]]], theme_name: str = "default",
                 columns: int = 3, close_after: bool = True) -> None:
        super().__init__(None, panel_flags())
        self.setStyleSheet(stylesheet(theme_name))
        root = QVBoxLayout(self)
        t = QLabel(title)
        t.setProperty("role", "title")
        root.addWidget(t)
        grid = QGridLayout()
        root.addLayout(grid)
        for i, (label, name, cb) in enumerate(items):
            b = QPushButton(label)
            b.setObjectName(name)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setMinimumSize(200, 80)

            def run(_: bool = False, f: Callable[[], None] = cb) -> None:
                if close_after:
                    self.hide()
                f()

            b.clicked.connect(run)
            grid.addWidget(b, i // columns, i % columns)
        close = QPushButton("Close")
        close.setObjectName("panel_close")
        close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close.clicked.connect(self.hide)
        grid.addWidget(close, (len(items) + columns - 1) // columns, 0, 1, columns)

    def showEvent(self, event: object) -> None:  # noqa: N802
        super().showEvent(event)  # type: ignore[arg-type]
        apply_no_activate(self)
        screen = self.screen()
        if screen is not None:
            g = screen.availableGeometry()
            self.adjustSize()
            self.move(g.left() + (g.width() - self.width()) // 2, g.top() + (g.height() - self.height()) // 2)


class ConfirmPanel(ButtonPanel):
    def __init__(self, question: str, on_yes: Callable[[], None], theme_name: str = "default") -> None:
        super().__init__(question, [("Yes", "confirm_yes", on_yes), ("No", "confirm_no", lambda: None)],
                         theme_name, columns=2)


class PrivacyPanel(QWidget):
    """Lists exactly what is stored and offers 'Delete all my data'."""

    def __init__(self, profile: Profile, on_deleted: Callable[[], None], theme_name: str = "default") -> None:
        super().__init__(None, Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("Privacy")
        self.setStyleSheet(stylesheet(theme_name))
        self.profile = profile
        self.on_deleted = on_deleted
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
        self.delete_btn.clicked.connect(self._confirm)
        root.addWidget(self.delete_btn)
        close = QPushButton("Close")
        close.setObjectName("privacy_close")
        close.clicked.connect(self.close)
        root.addWidget(close)
        self._armed = False
        self.refresh()

    def refresh(self) -> None:
        rows = self.profile.stored_items()
        text = "\n".join(f"• {label}: {path.name} ({size / 1024:.1f} KB)" for label, path, size in rows)
        self.items.setText(f"Stored for profile “{self.profile.name}”:\n{text or '• nothing yet'}")

    def _confirm(self) -> None:
        if not self._armed:
            self._armed = True
            self.delete_btn.setText("Look here again to confirm: delete everything")
            return
        self.profile.delete_all()
        self.refresh()
        self.close()
        self.on_deleted()
