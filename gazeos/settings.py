"""Single settings schema that validates every adjustable value (PRD 22, FR-36).

Defaults follow the PRD; the settings screen edits these and each profile stores
its own copy.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ControlMode(StrEnum):
    EYES_ONLY = "eyes_only"
    EYES_FACE = "eyes_face"
    HYBRID = "hybrid"
    HEAD_FINE_TUNE = "head_fine_tune"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CameraSettings(_Section):
    device_index: int = Field(0, ge=0)
    width: Literal[640, 1280] = 640
    height: Literal[480, 720] = 480
    fps: int = Field(30, ge=15, le=60)
    lock_exposure: bool = True
    # Optional accurate intrinsics from a checkerboard calibration (PRD 9.2).
    intrinsics_file: str | None = None


class ScreenSettings(_Section):
    """Physical screen geometry. Read from the OS when possible; editable here."""

    width_px: int = Field(1920, gt=0)
    height_px: int = Field(1080, gt=0)
    width_mm: float = Field(344.0, gt=0)
    height_mm: float = Field(194.0, gt=0)
    # Camera position relative to the top-centre of the screen, mm (PRD 11.3).
    camera_offset_mm: tuple[float, float, float] = (0.0, -10.0, 0.0)
    viewing_distance_mm: float = Field(600.0, gt=0)


class QualitySettings(_Section):
    min_inter_eye_px: float = 60.0
    max_head_angle_deg: float = 35.0
    blink_score_threshold: float = 0.5
    # No gaze sample while a lid hides the iris or moves fast (a blink starting or ending),
    # and for a few frames after; a lid lowered to look down is fine (ADR 0007).
    lid_hidden_level: float = Field(0.45, ge=0.0, le=1.0)  # either eye's blink score
    lid_speed: float = Field(0.1, gt=0.0)  # change of the blink score in one frame
    lid_hold_frames: int = Field(2, ge=0)
    min_brightness: float = 50.0
    face_lost_pause_s: float = 1.0
    head_pose_disagreement_deg: float = 10.0

    @model_validator(mode="before")
    @classmethod
    def _drop_old_lid_rule(cls, data: Any) -> Any:
        """Profiles saved on 27 Oct 2026 carry the replaced fixed lid threshold."""
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k not in ("gaze_lid_threshold", "gaze_lid_hold_frames")}
        return data


class FilterSettings(_Section):
    kind: Literal["none", "moving_average", "one_euro", "kalman", "uncertainty_kalman"] = (
        "one_euro"
    )
    min_cutoff_hz: float = Field(1.0, gt=0)
    beta: float = Field(0.007, ge=0)
    d_cutoff_hz: float = Field(1.0, gt=0)
    moving_average_window: int = Field(5, ge=1)
    kalman_process_noise: float = Field(2000.0, gt=0)
    kalman_measurement_noise_px: float = Field(40.0, gt=0)


class FixationSettings(_Section):
    # A saccade is a jump of the raw gaze away from the cursor, confirmed on consecutive
    # frames; the PRD's 30 deg/s velocity rule fires on webcam noise (ADR 0006).
    saccade_jump_deg: float = Field(7.0, gt=0)
    saccade_confirm_frames: int = Field(2, ge=1)
    dispersion_deg: float = 1.0
    min_fixation_ms: float = 100.0

    @model_validator(mode="before")
    @classmethod
    def _drop_velocity_rule(cls, data: Any) -> Any:
        """Profiles saved before ADR 0006 still carry the old velocity setting."""
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != "saccade_velocity_deg_s"}
        return data


class DwellSettings(_Section):
    dwell_ms: float = Field(800.0, ge=300.0, le=2000.0)
    radius_deg: float = Field(1.5, gt=0)
    cooldown_ms: float = Field(500.0, ge=0)
    enabled: bool = True


class BlinkSettings(_Section):
    long_blink_min_ms: float = Field(400.0, ge=150.0)
    long_blink_max_ms: float = Field(1000.0, ge=300.0)
    wink_min_ms: float = Field(300.0, ge=100.0)
    pause_close_ms: float = Field(2000.0, ge=1000.0)
    # Off by default: looking down at the keyboard reads as closed eyes on a laptop webcam
    # (REVIEW.md item 12), so a 2 s glance toggled pause. The hotkey and corner still pause.
    eyes_closed_pause: bool = False
    ear_threshold: float = Field(0.21, gt=0, lt=1)
    winks_enabled: bool = False


class GestureSettings(_Section):
    enabled: bool = False
    probability_threshold: float = Field(0.8, gt=0, lt=1)
    refractory_ms: float = 300.0
    gesture_size: float = Field(1.0, gt=0.1, le=3.0)
    mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "eyebrow_raise": "double",
            "mouth_open": "drag_toggle",
            "smile": "none",
            "left_wink": "left",
            "right_wink": "right",
            "long_blink": "left",
        }
    )


class SnapSettings(_Section):
    enabled: bool = True
    probability_threshold: float = Field(0.7, gt=0, lt=1)
    radius_sigmas: float = 2.0
    zoom_factor: float = Field(3.0, ge=1.5, le=6.0)
    zoom_region_px: int = Field(240, ge=80)


class ScrollSettings(_Section):
    zone_fraction: float = Field(0.06, gt=0, lt=0.3)
    zone_dwell_ms: float = 400.0
    step: int = Field(120, gt=0)
    rate_hz: float = Field(6.0, gt=0)
    # Zones are active when the Scroll action is selected, or always if set.
    always_on: bool = False


class SystemSettings(_Section):
    corner_menu_dwell_ms: float = 1500.0
    corner_pause_dwell_ms: float = 1500.0  # top-left corner toggles pause (FR-33)
    corner_size_px: int = 60
    emergency_hotkey: str = "ctrl+alt+g"
    pause_hotkey: str = "ctrl+alt+p"
    rest_reminder_min: float = 20.0
    start_with_os: bool = False
    show_overlay: bool = True
    auto_resume_after_face_lost: bool = True
    head_fine_tune_gain_px_per_deg: float = 6.0
    confirm_destructive_shortcuts: bool = True


class TypingSettings(_Section):
    layout: Literal["qwerty", "alphabetical", "large6"] = "qwerty"
    num_predictions: int = Field(4, ge=3, le=5)
    use_transformer: bool = False
    use_llm: bool = False
    llm_model: str = "qwen2.5:1.5b"
    tts_engine: Literal["pyttsx3", "piper", "none"] = "pyttsx3"
    piper_voice: str | None = None
    min_key_error_multiple: float = 2.5


class CalibrationSettings(_Section):
    num_points: Literal[9, 16] = 9
    target_duration_s: float = 1.5
    keep_from_s: float = 0.5
    mad_threshold: float = 3.0
    method: Literal["none", "offset", "polynomial", "gp", "finetune"] = "polynomial"
    implicit_enabled: bool = True
    implicit_refit_every: int = 20
    implicit_decay_min: float = 30.0


class PrivacySettings(_Section):
    research_recording: bool = False
    experiment_logging: bool = False
    heatmaps: bool = False


class Settings(_Section):
    mode: ControlMode = ControlMode.EYES_ONLY
    camera: CameraSettings = Field(default_factory=CameraSettings)
    screen: ScreenSettings = Field(default_factory=ScreenSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    filter: FilterSettings = Field(default_factory=FilterSettings)
    fixation: FixationSettings = Field(default_factory=FixationSettings)
    dwell: DwellSettings = Field(default_factory=DwellSettings)
    blink: BlinkSettings = Field(default_factory=BlinkSettings)
    gestures: GestureSettings = Field(default_factory=GestureSettings)
    snap: SnapSettings = Field(default_factory=SnapSettings)
    scroll: ScrollSettings = Field(default_factory=ScrollSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    typing: TypingSettings = Field(default_factory=TypingSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    # fp32 M4: 2 ms/frame on the laptop CPU, inside the 8 ms budget; static INT8 lost 1.2° and
    # ran slower, so it was rejected (docs/PLAN.md step 22).
    gaze_model: str = "models/gazenet_lite.onnx"
    gesture_model: str | None = "models/gesture_tcn.onnx"
    theme: Literal["default", "high_contrast", "large_text"] = "default"
