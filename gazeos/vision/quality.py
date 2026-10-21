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
        if inter_eye_px < self.s.min_inter_eye_px:
            return QualityReport(ok=False, status=TrackingStatus.MOVE_CLOSER, hold_cursor=True)
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
