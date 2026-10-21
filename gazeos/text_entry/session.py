"""Typing session state: the text, the current word's key presses, and edits (FR-27, FR-28)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from gazeos.text_entry.layouts import Key, Layout
from gazeos.text_entry.predictor import Predictor, Suggestions


@dataclass
class TypingSession:
    layout: Layout
    predictor: Predictor
    emit: Callable[[str], None] = lambda _t: None  # inject text into the focused app
    emit_key: Callable[[str], None] = lambda _k: None  # inject a named key (backspace, enter)
    text: str = ""
    shift: bool = False
    # Letters chosen for the current word and the gaze points (key units) of each press.
    letters: list[str] = field(default_factory=list)
    points: list[tuple[float, float]] = field(default_factory=list)
    keystrokes: int = 0
    suggestions: Suggestions = field(default_factory=lambda: Suggestions([]))

    @property
    def current_word(self) -> str:
        return "".join(self.letters)

    @property
    def text_before_word(self) -> str:
        return self.text[: len(self.text) - len(self.current_word)]

    def refresh(self) -> Suggestions:
        self.suggestions = self.predictor.suggest(self.text_before_word, self.current_word, self.points)
        return self.suggestions

    def press(self, key: Key, gaze_point: tuple[float, float] | None = None) -> Suggestions:
        self.keystrokes += 1
        if key.letters:
            ch = key.letters[0]
            if self.layout.grouped and gaze_point is None:
                gaze_point = key.center
            if self.shift:
                ch = ch.upper()
                self.shift = False
            if ch in ".,?":
                self._commit_word_boundary()
                self._type(ch + " ")
            else:
                self.letters.append(ch)
                self.points.append(gaze_point or key.center)
                self._type(ch)
        elif key.id == "space":
            self._commit_word_boundary()
            self._type(" ")
        elif key.id == "backspace":
            self.backspace()
        elif key.id == "enter":
            self._commit_word_boundary()
            self.text += "\n"
            self.emit_key("enter")
        elif key.id == "shift":
            self.shift = not self.shift
        return self.refresh()

    def _type(self, s: str) -> None:
        self.text += s
        self.emit(s)

    def _commit_word_boundary(self) -> None:
        if self.letters:
            self.predictor.learn(self.current_word)
        self.letters.clear()
        self.points.clear()

    def backspace(self) -> None:
        if not self.text:
            return
        self.text = self.text[:-1]
        self.emit_key("backspace")
        if self.letters:
            self.letters.pop()
            self.points.pop()

    def choose(self, word: str) -> Suggestions:
        """Replace the current (possibly mistyped) word with a suggestion, plus a space."""
        self.keystrokes += 1
        typed = self.current_word
        for _ in typed:
            self.emit_key("backspace")
        self.text = self.text[: len(self.text) - len(typed)]
        self.letters.clear()
        self.points.clear()
        self._type(word + " ")
        self.predictor.learn(word)
        return self.refresh()

    def insert_phrase(self, phrase: str) -> Suggestions:
        self.keystrokes += 1
        self._commit_word_boundary()
        self._type(phrase)
        return self.refresh()

    def clear(self) -> None:
        self.text = ""
        self.letters.clear()
        self.points.clear()


def keystroke_savings_rate(keystrokes: int, characters: int) -> float:
    """KSR = 1 - keystrokes / characters (PRD 14.6)."""
    if characters <= 0:
        return 0.0
    return 1.0 - keystrokes / characters
