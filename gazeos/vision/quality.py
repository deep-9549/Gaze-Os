"""Quality gates that decide whether a frame may update the gaze (PRD 9.6, FR-03)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gazeos.settings import QualitySettings
from gazeos.types import TrackingStatus


@dataclass
class QualityReport:
    ok: bool  # frame may update the gaze estimate
    status: TrackingStatus
    hold_cursor: bool = False
    eye_closed: bool = False
    lower_confidence: bool = False
    auto_pause: bool = False


class QualityGate:
    def __init__(self, settings: QualitySettings | None = None) -> None:
        self.s = settings or QualitySettings()
        self._last_face_t: float | None = None

    def check(
        self,
        t: float,
        num_faces: int,
        inter_eye_px: float = 0.0,
        head_pitch: float = 0.0,
        head_yaw: float = 0.0,
        blink_score: float = 0.0,
        brightness: float = 255.0,
    ) -> QualityReport:
        if num_faces == 0:
            lost_for = 0.0 if self._last_face_t is None else t - self._last_face_t
            return QualityReport(
                ok=False,
                status=TrackingStatus.NO_FACE,
                hold_cursor=True,
                auto_pause=self._last_face_t is not None and lost_for > self.s.face_lost_pause_s,
            )
        self._last_face_t = t
        if num_faces > 1:
            return QualityReport(ok=False, status=TrackingStatus.MULTIPLE_FACES, hold_cursor=True)
        if brightness < self.s.min_brightness:
            # Tracking continues in low light; the chip warns the user.
            low = QualityReport(ok=True, status=TrackingStatus.LOW_LIGHT, lower_confidence=True)
        else:
            low = None
        if inter_eye_px < self.s.min_inter_eye_px_hard:
            return QualityReport(ok=False, status=TrackingStatus.MOVE_CLOSER, hold_cursor=True)
        if inter_eye_px < self.s.min_inter_eye_px and low is None:
            # Far, but trackable: keep tracking and ask the user to come closer.
            low = QualityReport(ok=True, status=TrackingStatus.MOVE_CLOSER, lower_confidence=True)
        max_angle = math.radians(self.s.max_head_angle_deg)
        if abs(head_pitch) > max_angle or abs(head_yaw) > max_angle:
            return QualityReport(
                ok=False,
                status=low.status if low else TrackingStatus.TRACKING,
                hold_cursor=True,
                lower_confidence=True,
            )
        if blink_score > self.s.blink_score_threshold:
            return QualityReport(
                ok=False,
                status=low.status if low else TrackingStatus.TRACKING,
                eye_closed=True,
                hold_cursor=True,
            )
        return low or QualityReport(ok=True, status=TrackingStatus.TRACKING)


class LidDropMonitor:
    """Says when the lid gate drops most frames for a while: the eyelids hide the eyes when
    the user looks down, usually because the screen sits low against their eyes. On 18 Nov
    2026 it dropped 73-78% of the frames on the lower keyboard rows in one sitting, and the
    cursor felt sluggish there for no visible reason."""

    def __init__(self, window_s: float = 10.0, rate: float = 0.5, repeat_s: float = 300.0) -> None:
        self.window_s, self.rate, self.repeat_s = window_s, rate, repeat_s
        self._frames: list[tuple[float, bool]] = []
        self._last_warning = -1e9

    def update(self, t: float, lid_down: bool, tracking: bool) -> str | None:
        if not tracking:
            self._frames.clear()
            return None
        self._frames.append((t, lid_down))
        while self._frames and self._frames[0][0] < t - self.window_s:
            self._frames.pop(0)
        if t - self._frames[0][0] < self.window_s * 0.9 or t - self._last_warning < self.repeat_s:
            return None
        if sum(d for _, d in self._frames) / len(self._frames) < self.rate:
            return None
        self._last_warning = t
        return ("Your eyelids are covering your eyes when you look down, so tracking keeps stopping. "
                "Raise the screen a little, or sit a little lower.")
