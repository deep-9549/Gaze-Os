"""Shared data types: the per-frame contract (PRD Section 8.2) and UI elements."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ControlState(str, Enum):
    """Dwell / control state reported on every frame (PRD 12.7)."""

    IDLE = "idle"
    DWELLING = "dwelling"
    FIRED = "fired"
    COOLDOWN = "cooldown"
    PAUSED = "paused"


class TrackingStatus(str, Enum):
    """Status chip values (PRD 7.4). Always shown with a text label, never colour alone."""

    TRACKING = "Tracking"
    PAUSED = "Paused"
    NO_FACE = "No face"
    LOW_LIGHT = "Low light"
    MOVE_CLOSER = "Move closer"
    MULTIPLE_FACES = "More than one face"


@dataclass
class FrameResult:
    """One typed record that every pipeline stage reads and adds to (PRD 8.2)."""

    t_capture: float
    face_found: bool = False
    landmarks: FloatArray | None = None  # (478, 3), pixels for x/y
    blendshapes: dict[str, float] = field(default_factory=dict)
    head_R: FloatArray | None = None  # (3, 3) rotation
    head_t: FloatArray | None = None  # (3,) translation, mm
    gaze_pitch_yaw: tuple[float, float] | None = None  # radians, camera frame
    gaze_sigma: tuple[float, float] | None = None  # predicted uncertainty, radians
    gesture_probs: dict[str, float] = field(default_factory=dict)
    screen_xy: tuple[int, int] | None = None
    state: str = ControlState.IDLE.value
    timings_ms: dict[str, float] = field(default_factory=dict)
    # Extra fields used by later stages; not part of the minimal PRD contract.
    num_faces: int = 0
    image_size: tuple[int, int] | None = None  # (width, height)
    quality_ok: bool = True
    status: str = TrackingStatus.TRACKING.value
    eye_closed: bool = False
    features: FloatArray | None = None  # penultimate gaze features, for P4


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.left + self.width and self.top <= y < self.top + self.height


@dataclass(frozen=True)
class UIElement:
    """A clickable element read from the OS accessibility tree (PRD 12.6, 20.1)."""

    rect: Rect
    role: str = "button"
    name: str = ""
    invokable: bool = True
    is_password: bool = False
