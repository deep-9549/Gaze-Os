"""Phrase bank of saved sentences (FR-29, PRD 14.4)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DEFAULT_PHRASES = [
    "I need water.",
    "Please call my family.",
    "I am in pain.",
    "Yes.",
    "No.",
    "Thank you.",
    "Please wait a moment.",
    "I need to rest.",
    "Can you help me?",
    "I am hungry.",
    "Please adjust my position.",
    "Good morning.",
]


class PhraseBank:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.phrases: list[str] = list(DEFAULT_PHRASES)
        self.usage: Counter[str] = Counter()
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.phrases = data.get("phrases", self.phrases)
            self.usage = Counter(data.get("usage", {}))

    def add(self, phrase: str) -> None:
        phrase = phrase.strip()
        if phrase and phrase not in self.phrases:
            self.phrases.append(phrase)
            self.save()

    def remove(self, phrase: str) -> None:
        if phrase in self.phrases:
            self.phrases.remove(phrase)
            self.save()

    def used(self, phrase: str) -> None:
        self.usage[phrase] += 1
        self.save()

    def ranked(self) -> list[str]:
        order = {p: i for i, p in enumerate(self.phrases)}
        return sorted(self.phrases, key=lambda p: (-self.usage[p], order[p]))

    def note_typed(self, sentence: str, threshold: int = 3) -> bool:
        """Offer to save sentences the user types often. Returns True when added."""
        sentence = sentence.strip()
        if not sentence or sentence in self.phrases:
            return False
        self.usage[sentence] += 1
        if self.usage[sentence] >= threshold:
            self.add(sentence)
            return True
        self.save()
        return False

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"phrases": self.phrases, "usage": dict(self.usage)}, indent=2), encoding="utf-8"
        )
