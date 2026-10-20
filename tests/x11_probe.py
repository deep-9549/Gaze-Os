"""Child process for test_backends: a Qt window records the events that the X11
input backend injects. Prints the received events as JSON."""

from __future__ import annotations

import json
import os
import sys
import threading
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget

from gazeos.os_backends.linux_x11 import X11Backend


class Recorder(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[list[object]] = []
        self.setMouseTracking(True)
        self.setGeometry(0, 0, 640, 480)

    def mousePressEvent(self, e) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        self.events.append(["press", e.button().name, int(e.position().x()), int(e.position().y())])

    def mouseDoubleClickEvent(self, e) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        self.events.append(["double", e.button().name, int(e.position().x()), int(e.position().y())])

    def wheelEvent(self, e) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        self.events.append(["wheel", int(e.angleDelta().y())])

    def keyPressEvent(self, e) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        self.events.append(["key", e.text(), int(e.modifiers().value)])


def main() -> None:
    app = QApplication(sys.argv)
    w = Recorder()
    w.show()
    w.activateWindow()
    w.setFocus(Qt.FocusReason.OtherFocusReason)

    def drive() -> None:
        b = X11Backend()
        time.sleep(0.5)
        b.move(100, 120)
        time.sleep(0.1)
        b.click("left")
        time.sleep(0.1)
        b.move(300, 200)
        time.sleep(0.1)
        b.click("right")
        time.sleep(0.1)
        b.scroll(0, 240)
        time.sleep(0.1)
        b.type_text("hi")
        time.sleep(0.1)
        b.hotkey("ctrl", "a")
        time.sleep(0.5)
        print(json.dumps({"events": w.events, "size": b.screen_size()}))
        sys.stdout.flush()
        os._exit(0)  # a Qt quit from this worker thread would not reach the event loop

    QTimer.singleShot(500, lambda: threading.Thread(target=drive, daemon=True).start())
    app.exec()


if __name__ == "__main__":
    main()
