"""Rule-based blink and wink detection (baseline B0; PRD 7.2, 13.2, FR-20, FR-21).

Events fire when the eyes reopen, so their duration is known:
- long blink: both eyes closed 400-1000 ms -> click
- wink: one eye closed >= 300 ms while the other stays open
- both eyes closed >= 2 s -> pause / resume
Normal blinks (about 100-300 ms) never fire.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from gazeos.settings import BlinkSettings


class BlinkEvent(StrEnum):
    NORMAL_BLINK = "normal_blink"
    LONG_BLINK = "long_blink"
    LEFT_WINK = "left_wink"
    RIGHT_WINK = "right_wink"
    PAUSE_TOGGLE = "pause_toggle"


@dataclass
class EyeState:
    left_closed: bool
    right_closed: bool

    @property
    def both_closed(self) -> bool:
        return self.left_closed and self.right_closed

    @property
    def any_closed(self) -> bool:
        return self.left_closed or self.right_closed


def eye_state(
    blendshapes: dict[str, float] | None,
    ear_left: float,
    ear_right: float,
    ear_threshold: float = 0.21,
    blink_score_threshold: float = 0.5,
) -> EyeState:
    """Closed-eye decision from MediaPipe blink scores, falling back to EAR."""
    if blendshapes and "eyeBlinkLeft" in blendshapes and "eyeBlinkRight" in blendshapes:
        return EyeState(
            left_closed=blendshapes["eyeBlinkLeft"] > blink_score_threshold,
            right_closed=blendshapes["eyeBlinkRight"] > blink_score_threshold,
        )
    return EyeState(left_closed=ear_left < ear_threshold, right_closed=ear_right < ear_threshold)


class BlinkDetector:
    def __init__(self, settings: BlinkSettings | None = None) -> None:
        self.s = settings or BlinkSettings()
        self._both_start: float | None = None
        self._left_start: float | None = None
        self._right_start: float | None = None
        self._both_seen = False  # both eyes closed at some point in this closure
        # Duration of the most recent both-eyes closure, for blink tuning (PRD 7.6).
        self.last_closure_ms: float | None = None

    @property
    def eyes_closed_since(self) -> float | None:
        starts = [s for s in (self._both_start, self._left_start, self._right_start) if s is not None]
        return min(starts) if starts else None

    def update(self, t: float, state: EyeState) -> BlinkEvent | None:
        event: BlinkEvent | None = None
        # Track each eye's closure start.
        if state.left_closed and self._left_start is None:
            self._left_start = t
        if state.right_closed and self._right_start is None:
            self._right_start = t
        if state.both_closed:
            if self._both_start is None:
                self._both_start = t
            self._both_seen = True

        closure_over = not state.any_closed and (
            self._left_start is not None or self._right_start is not None
        )
        if closure_over:
            event = self._classify(t)
            self._both_start = self._left_start = self._right_start = None
            self._both_seen = False
        return event

    def _classify(self, t: float) -> BlinkEvent | None:
        def ms(start: float | None) -> float:
            return (t - start) * 1000.0 if start is not None else 0.0

        if self._both_seen:
            dur = ms(self._both_start)
            self.last_closure_ms = dur
            if dur >= self.s.pause_close_ms:
                return BlinkEvent.PAUSE_TOGGLE
            if self.s.long_blink_min_ms <= dur <= self.s.long_blink_max_ms:
                return BlinkEvent.LONG_BLINK
            if dur < self.s.long_blink_min_ms:
                return BlinkEvent.NORMAL_BLINK
            return None  # between long-blink max and pause: ignore
        if not self.s.winks_enabled:
            return None
        left, right = ms(self._left_start), ms(self._right_start)
        if left >= self.s.wink_min_ms and right == 0.0:
            return BlinkEvent.LEFT_WINK
        if right >= self.s.wink_min_ms and left == 0.0:
            return BlinkEvent.RIGHT_WINK
        return None

    def reset(self) -> None:
        self._both_start = self._left_start = self._right_start = None
        self._both_seen = False


def tune_long_blink_threshold(
    normal_ms: Sequence[float], long_ms: Sequence[float], floor: float = 250.0, ceiling: float = 800.0
) -> float:
    """Personal threshold from five normal and five long blinks (PRD 7.6 step 3).

    Halfway between the slowest normal blink and the quickest long blink,
    clamped to a safe range.
    """
    if not normal_ms or not long_ms:
        return 400.0
    lo, hi = max(normal_ms), min(long_ms)
    threshold = (lo + hi) / 2.0 if hi > lo else lo * 1.3
    return float(min(max(threshold, floor), ceiling))
