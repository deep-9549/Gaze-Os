"""Gaze keyboard window with prediction bar and phrase bank (PRD 7.5, FR-27-FR-30).

The window never takes focus, so typed characters go to the app the user was in.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gazeos.text_entry.layouts import Key, Layout, key_unit_px, symbols_layout
from gazeos.text_entry.phrases import PhraseBank
from gazeos.text_entry.session import TypingSession
from gazeos.text_entry.tts import Speaker
from gazeos.ui.common import apply_no_activate, device_ratio, panel_flags, stylesheet


class KeyboardWindow(QWidget):
    def __init__(
        self,
        session: TypingSession,
        phrases: PhraseBank,
        speaker: Speaker,
        error_radius_px: float,
        on_close: Callable[[], None],
        theme_name: str = "default",
        num_predictions: int = 4,
    ) -> None:
        super().__init__(None, panel_flags())
        self.session = session
        self.phrases = phrases
        self.speaker = speaker
        self.on_close = on_close
        self.num_predictions = num_predictions
        self.last_gaze_px: tuple[int, int] | None = None
        self.setStyleSheet(stylesheet(theme_name))
        self.unit = 0.0
        self.error_radius_px = error_radius_px
        self.current_layout = session.layout  # what the keys show: the letters or a page of symbols

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        self.preview = QLabel("")
        self.preview.setProperty("role", "muted")
        root.addWidget(self.preview)
        pred_row = QHBoxLayout()
        self.pred_buttons: list[QPushButton] = []
        for i in range(num_predictions):
            b = QPushButton("")
            b.setObjectName(f"pred{i}")
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, idx=i: self._choose(idx))
            pred_row.addWidget(b)
            self.pred_buttons.append(b)
        self.sentence_button = QPushButton("")
        self.sentence_button.setObjectName("sentence")
        self.sentence_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sentence_button.clicked.connect(self._choose_sentence)
        self.sentence_button.hide()
        pred_row.addWidget(self.sentence_button)
        root.addLayout(pred_row)
        self.keys_host = QWidget()
        root.addWidget(self.keys_host)
        self.phrase_host = QWidget()
        self.phrase_host.hide()
        root.addWidget(self.phrase_host)
        self.key_buttons: dict[str, QPushButton] = {}
        self.build(session.layout, error_radius_px)
        self._update_predictions()

    # -- layout ----------------------------------------------------------------
    def build(self, layout: Layout, error_radius_px: float) -> None:
        for b in self.key_buttons.values():
            b.hide()  # hidden buttons are not gaze targets; deleted once the click that called this returns
            b.deleteLater()
        self.key_buttons.clear()
        self.current_layout = layout
        ratio = device_ratio(self)
        self.error_radius_px = error_radius_px
        screen = QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        avail_w = avail.width() if avail else 1280
        avail_h = avail.height() if avail else 720
        # As large as the screen allows, whatever the calibration score: the score is measured
        # with the head still, right after calibrating, and typing error was 2-3 times larger
        # (18 Nov 2026). Sized from the score, a good calibration (0.5 cm) gave 12 mm keys whose
        # labels no longer fit. Up to 85% of the screen height, less the prediction strip.
        unit = min((avail_w - 40) / layout.width, (0.85 * avail_h - 130) / layout.height)
        unit = max(unit, key_unit_px(error_radius_px / ratio))  # never smaller than 2.5x the error
        unit = min(unit, (avail_w - 40) / layout.width)  # but always on screen
        self.unit = unit
        self.keys_host.setFixedSize(int(layout.width * unit), int(layout.height * unit))
        for key in layout.keys:
            b = QPushButton(key.label.replace("&", "&&"), self.keys_host)  # a lone & marks a shortcut
            b.setObjectName(f"key_{key.id}")
            if key.letters or key.text or key.id in ("space", "backspace"):
                b.setProperty("gaze_repeat", True)  # keep looking to type it again
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setGeometry(int(key.x * unit) + 2, int(key.y * unit) + 2, int(key.w * unit) - 4, int(key.h * unit) - 4)
            b.clicked.connect(lambda _=False, k=key: self._press(k))
            b.show()
            self.key_buttons[key.id] = b
        self.adjustSize()
        self._place()

    def _place(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        g = screen.availableGeometry()
        self.move(g.left() + (g.width() - self.width()) // 2, g.bottom() - self.height())

    def showEvent(self, event: object) -> None:  # noqa: N802
        super().showEvent(event)  # type: ignore[arg-type]
        apply_no_activate(self)

    # -- typing ----------------------------------------------------------------
    def _gaze_in_key_units(self) -> tuple[float, float] | None:
        if self.last_gaze_px is None or self.unit <= 0:
            return None
        ratio = device_ratio(self)
        origin = self.keys_host.mapToGlobal(QPoint(0, 0))
        x = (self.last_gaze_px[0] / ratio - origin.x()) / self.unit
        y = (self.last_gaze_px[1] / ratio - origin.y()) / self.unit
        return (x, y)

    def _press(self, key: Key) -> None:
        if key.id == "speak":
            if self.session.text.strip():
                self.speaker.speak(self.session.text.strip())
            return
        if key.id == "phrases":
            self._toggle_phrases()
            return
        if key.id == "close":
            self.on_close()
            return
        if key.id in ("numbers", "symbols"):  # "123" and "#+=": the pages of digits and symbols
            page = 1 if key.id == "symbols" else 0
            self.build(symbols_layout(self.session.layout, page), self.error_radius_px)
            return
        if key.id == "letters":  # "ABC": back to the letters the user came from
            self.build(self.session.layout, self.error_radius_px)
            return
        point = self._gaze_in_key_units() if key.letters else None
        self.session.press(key, point)
        if key.id == "enter":
            self.phrases.note_typed(self.session.text.strip().splitlines()[-1] if self.session.text.strip() else "")
        self._update_predictions()

    def _choose(self, idx: int) -> None:
        words = self.session.suggestions.words
        if idx < len(words):
            self.session.choose(words[idx])
            self._update_predictions()

    def _choose_sentence(self) -> None:
        sentence = self.session.suggestions.sentence
        if sentence:
            typed = self.session.text
            rest = sentence[len(typed):] if sentence.lower().startswith(typed.lower()) else sentence
            self.session.insert_phrase(rest)
            self._update_predictions()

    def _update_predictions(self) -> None:
        sugg = self.session.suggestions if self.session.suggestions.words else self.session.refresh()
        for i, b in enumerate(self.pred_buttons):
            b.setText(sugg.words[i] if i < len(sugg.words) else "")
            b.setEnabled(i < len(sugg.words))
        if sugg.sentence:
            self.sentence_button.setText(sugg.sentence)
            self.sentence_button.show()
        else:
            self.sentence_button.hide()
        tail = self.session.text[-60:].replace("\n", " ⏎ ")
        self.preview.setText(tail or "Look at a key to type. Predictions appear above the keys.")

    # -- phrase bank -----------------------------------------------------------
    def _toggle_phrases(self) -> None:
        if self.phrase_host.isVisible():
            self.phrase_host.hide()
            self.keys_host.show()
            return
        old = self.phrase_host.layout()
        if old is not None:
            QWidget().setLayout(old)  # reparent the old layout away
        grid = QGridLayout(self.phrase_host)
        for i, phrase in enumerate(self.phrases.ranked()[:12]):
            b = QPushButton(phrase)
            b.setObjectName(f"phrase{i}")
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, p=phrase: self._insert_phrase(p))
            grid.addWidget(b, i // 3, i % 3)
        back = QPushButton("Back to keys")
        back.setObjectName("phrases_back")
        back.clicked.connect(self._toggle_phrases)
        grid.addWidget(back, 4, 0, 1, 3)
        self.keys_host.hide()
        self.phrase_host.show()

    def _insert_phrase(self, phrase: str) -> None:
        self.phrases.used(phrase)
        self.session.insert_phrase(phrase + " ")
        self._toggle_phrases()
        self._update_predictions()
