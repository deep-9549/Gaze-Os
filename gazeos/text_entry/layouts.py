"""Gaze keyboard layouts and key geometry (PRD 7.5, FR-27).

Geometry is in abstract key units (1 unit = one standard key); the UI scales
it so keys are at least 2.5x the user's measured error radius.
"""

from __future__ import annotations

from dataclasses import dataclass

SPECIAL_KEYS = ("space", "backspace", "speak", "phrases", "enter", "shift", "numbers", "close")


@dataclass(frozen=True)
class Key:
    id: str  # "a", "space", "g:abc" (group), ...
    label: str
    x: float  # left, key units
    y: float  # top, key units
    w: float = 1.0
    h: float = 1.0
    letters: str = ""  # letters this key can produce (several for grouped keys)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass(frozen=True)
class Layout:
    name: str
    keys: tuple[Key, ...]
    width: float
    height: float

    def key_for_letter(self, letter: str) -> Key | None:
        for k in self.keys:
            if letter in k.letters:
                return k
        return None

    def letter_centers(self) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for k in self.keys:
            for ch in k.letters:
                out[ch] = k.center
        return out

    def key_at(self, x: float, y: float) -> Key | None:
        for k in self.keys:
            if k.x <= x < k.x + k.w and k.y <= y < k.y + k.h:
                return k
        return None

    @property
    def grouped(self) -> bool:
        return any(len(k.letters) > 1 for k in self.keys)


def _special_row(y: float, width: float) -> list[Key]:
    specs = [("shift", "Shift", 1.5), ("numbers", "123", 1.5), ("space", "Space", width - 9.0),
             ("backspace", "Del", 1.5), ("enter", "Enter", 1.5), ("speak", "Speak", 1.5),
             ("phrases", "Phrases", 1.5)]
    total = sum(w for _, _, w in specs)
    scale = width / total
    keys, x = [], 0.0
    for kid, label, w in specs:
        keys.append(Key(kid, label, x, y, w * scale, 1.0))
        x += w * scale
    return keys


def _rows_layout(name: str, rows: list[str], offsets: list[float]) -> Layout:
    keys: list[Key] = []
    width = float(max(len(r) + off for r, off in zip(rows, offsets, strict=True)))
    for row_i, (row, off) in enumerate(zip(rows, offsets, strict=True)):
        for col, ch in enumerate(row):
            label = {"'": "'", ".": ".", ",": ",", "?": "?"}.get(ch, ch.upper())
            keys.append(Key(ch, label, off + col, float(row_i), 1.0, 1.0, letters=ch))
    keys += _special_row(float(len(rows)), width)
    keys.append(Key("close", "Close", width, 0.0, 1.0, 1.0))
    return Layout(name, tuple(keys), width + 1.0, float(len(rows) + 1))


def qwerty() -> Layout:
    return _rows_layout("qwerty", ["qwertyuiop", "asdfghjkl'", "zxcvbnm,.?"], [0.0, 0.25, 0.5])


def alphabetical() -> Layout:
    return _rows_layout("alphabetical", ["abcdefghi", "jklmnopqr", "stuvwxyz'", ".,?"], [0.0, 0.0, 0.0, 3.0])


def large6() -> Layout:
    """Six big grouped keys for low accuracy; the decoder resolves the ambiguity."""
    groups = ["abcd", "efgh", "ijklm", "nopq", "rstu", "vwxyz'"]
    keys = []
    for i, g in enumerate(groups):
        col, row = i % 3, i // 3
        keys.append(Key(f"g:{g}", g.upper(), col * 2.0, row * 2.0, 2.0, 2.0, letters=g))
    y = 4.0
    keys += [
        Key("space", "Space", 0.0, y, 2.0, 1.5),
        Key("backspace", "Del", 2.0, y, 2.0, 1.5),
        Key("speak", "Speak", 4.0, y, 2.0, 1.5),
        Key("phrases", "Phrases", 0.0, y + 1.5, 2.0, 1.0),
        Key("enter", "Enter", 2.0, y + 1.5, 2.0, 1.0),
        Key("close", "Close", 4.0, y + 1.5, 2.0, 1.0),
    ]
    return Layout("large6", tuple(keys), 6.0, 6.5)


LAYOUTS = {"qwerty": qwerty, "alphabetical": alphabetical, "large6": large6}


def get_layout(name: str) -> Layout:
    return LAYOUTS[name]()


def key_unit_px(error_radius_px: float, multiple: float = 2.5, min_px: float = 48.0) -> float:
    """Size of one key unit so keys are >= `multiple` x the error radius."""
    return max(error_radius_px * multiple, min_px)
