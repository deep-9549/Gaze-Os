"""Control engine: turns per-frame gaze and face events into OS actions.

Pipeline per frame (PRD 8, 11, 12): personalization -> drift offset -> adaptive
filter -> fixation hold -> magnetic edges -> snapping -> cursor. Clicking comes
from dwell, long blinks, winks, gestures or a hybrid switch. The engine never
sends input while paused (PRD 18.3) and auto-pauses on face loss (FR-40).
"""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.calibration.implicit import DriftCorrector, GazeHistory, ImplicitRecalibrator
from gazeos.calibration.personalization import NoPersonalization, Personalizer
from gazeos.calibration.protocol import GazeObservation
from gazeos.control.dwell import DwellClicker
from gazeos.control.filters import PointFilter, make_filter
from gazeos.control.fixation import EyeEvent, FixationDetector
from gazeos.control.mapping import magnetic_edges
from gazeos.control.snapping import TargetSnapper
from gazeos.control.targets import GazeTarget, TargetRegistry
from gazeos.control.zoom import ZoomRegion, make_zoom_region
from gazeos.gestures.blink import BlinkEvent
from gazeos.os_backends.base import InputBackend
from gazeos.os_backends.keys import DESTRUCTIVE_SHORTCUTS, shortcut_keys
from gazeos.settings import ControlMode, Settings
from gazeos.types import ControlState, TrackingStatus
from gazeos.vision.quality import QualityReport

log = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


class Action(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    DOUBLE = "double"
    DRAG = "drag"
    SCROLL = "scroll"


@dataclass
class EngineInput:
    t: float
    obs: GazeObservation | None
    quality: QualityReport
    blink_event: BlinkEvent | None = None
    gesture_event: str | None = None
    eyes_closed: bool = False


@dataclass
class EngineOutput:
    cursor: tuple[int, int] | None
    state: ControlState
    status: TrackingStatus
    dwell_progress: float = 0.0
    sigma_px: float = 0.0
    action: Action = Action.LEFT
    zoom: ZoomRegion | None = None
    events: list[str] = field(default_factory=list)
    paused: bool = False
    dragging: bool = False


class ControlEngine:
    # Frames without a gaze sample, by cause. Tracking lost (no face, more than one face, too
    # far, head turned away) or eyes closed: the time does not count toward a dwell, and
    # after MAX_DWELL_GAP_S of it the dwell starts again, so a dwell can never finish on the
    # first frame back. Face tracked with the eyes open but the sample dropped by the lid gate
    # (lids lowered to look at the lower screen, a blink starting): the user is still looking
    # where they were, so the time counts, for up to MAX_LID_GAP_S at a stretch. Not counting
    # it froze the dwell on the lower keyboard rows, where the lid gate dropped 13-78% of the
    # frames (18 Nov 2026 traces).
    MAX_DWELL_GAP_S = 0.6
    MAX_LID_GAP_S = 1.5
    # GazeOS buttons that still respond while paused (registry keys start with these).
    RESUME_TARGETS = ("action_bar:pause:", "menu:menu_pause:")
    # A repeatable GazeOS target (typing keys) fires again when the gaze stays on it this long
    # after it fired, plus a full dwell: 0.7 + 0.8 = 1.5 s. On 18 Nov the gaze left a key a
    # median 0.3 s after it fired, and stayed over 1.5 s once in 90 presses (on Del).
    REPEAT_AFTER_S = 0.7

    def __init__(
        self,
        settings: Settings,
        geometry: ScreenGeometry,
        backend: InputBackend,
        personalizer: Personalizer | None = None,
        registry: TargetRegistry | None = None,
        recalibrator: ImplicitRecalibrator | None = None,
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        self.s = settings
        self.geometry = geometry
        self.backend = backend
        self.personalizer = personalizer or NoPersonalization(geometry)
        self.registry = registry or TargetRegistry()
        self.recalibrator = recalibrator
        self.confirm = confirm or (lambda _name: True)
        self.drift = DriftCorrector()
        self.history = GazeHistory()
        self.px_per_deg = geometry.deg_to_px(1.0)
        self._dwell_target: GazeTarget | None = None
        self._last_valid_t: float | None = None  # time of the last valid gaze sample
        self._prev_t: float | None = None  # time of the previous frame of any kind
        self._lost_run = 0.0  # seconds without a sample because tracking was lost or the eyes closed
        self._lid_run = 0.0  # seconds without a sample because the lid gate dropped it
        self._repeat_key: str | None = None  # the GazeOS target that fired last, and when
        self._repeat_t = 0.0
        self._build(settings)
        self.action = Action.LEFT
        self.dragging = False
        self.paused = False
        self.auto_paused = False
        self.zoom: ZoomRegion | None = None
        self.cursor: FloatArray | None = None
        self._frozen: FloatArray | None = None
        self._last_scroll_t = -1e9
        self._zone_since: float | None = None
        self._corner_since: dict[str, float] = {}
        self._start_t: float | None = None
        self._last_rest_reminder = 0.0
        self._fine_anchor: tuple[FloatArray, FloatArray] | None = None
        self.error_px = settings.dwell.radius_deg * self.px_per_deg  # updated after validation
        self.event_log: list[tuple[float, str]] = []

    def _build(self, settings: Settings) -> None:
        """The parts that copy values out of the settings when they are made."""
        self.filter: PointFilter = make_filter(settings.filter, self.px_per_deg)
        self.fixation = FixationDetector(
            self.px_per_deg,
            saccade_jump_deg=settings.fixation.saccade_jump_deg,
            dispersion_deg=settings.fixation.dispersion_deg,
            min_fixation_ms=settings.fixation.min_fixation_ms,
            confirm_frames=settings.fixation.saccade_confirm_frames,
            hold_deadband_deg=settings.fixation.hold_deadband_deg,
            hold_ease=settings.fixation.hold_ease,
            hold_lock_after_s=settings.fixation.hold_lock_after_ms / 1000.0,
        )
        self.dwell = DwellClicker(
            settings.dwell.dwell_ms, settings.dwell.radius_deg * self.px_per_deg, settings.dwell.cooldown_ms
        )
        self.snapper = TargetSnapper(settings.snap.probability_threshold, settings.snap.radius_sigmas)

    # -- public controls -------------------------------------------------------
    def apply_settings(self, settings: Settings) -> None:
        """Saved settings take effect at once: the dwell clicker, filter, fixation detector
        and snapper are rebuilt from them (they keep their own copies of the values)."""
        self.s = settings
        self._build(settings)
        self._dwell_target = None

    def release_inputs(self) -> None:
        """Let go of everything the engine may be holding or about to do: a held mouse
        button, a running dwell, an open magnifier. Called on pause, before calibration,
        on exit and when the camera stops. A click by blink needs a fresh gaze sample after."""
        if self.dragging:
            self._end_drag()
        self.dwell.reset()
        self._dwell_target = None
        self.zoom = None
        self._frozen = None
        self._last_valid_t = None
        self._lost_run = self._lid_run = 0.0

    def set_action(self, action: Action) -> None:
        if self.dragging and action != Action.DRAG:
            self._end_drag()
        self.action = action

    def toggle_pause(self, reason: str = "user") -> None:
        self.set_paused(not self.paused, reason)

    def set_paused(self, paused: bool, reason: str = "user") -> None:
        self.release_inputs()
        self.paused = paused
        self.auto_paused = paused and reason == "face_lost"
        self._log(0.0, f"{'pause' if paused else 'resume'}:{reason}")

    def emergency_stop(self) -> None:
        """Ctrl+Alt+G: always stops all injected input immediately (PRD 18.3)."""
        self.set_paused(True, "emergency")

    def confirm_click(self) -> None:
        """Hybrid mode: a physical key or switch confirms the click at the cursor."""
        if self.cursor is not None and not self.paused:
            self._fire(self.cursor, self._t_last, source="switch")

    def run_shortcut(self, name: str) -> bool:
        if self.paused:
            return False
        if (
            name in DESTRUCTIVE_SHORTCUTS
            and self.s.system.confirm_destructive_shortcuts
            and not self.confirm(name)
        ):
            return False
        self.backend.hotkey(*shortcut_keys(name, sys.platform))
        return True

    def type_text(self, text: str) -> None:
        if not self.paused:
            self.backend.type_text(text)

    _t_last: float = 0.0

    # -- per-frame update ------------------------------------------------------
    def update(self, inp: EngineInput) -> EngineOutput:
        t = inp.t
        self._t_last = t
        if self._start_t is None:
            self._start_t = t
        events: list[str] = []
        q = inp.quality

        # Face lost for > 1 s -> auto-pause; resume when the face returns.
        if q.auto_pause and not self.paused:
            self.set_paused(True, "face_lost")
            events.append("auto_pause")
        elif self.auto_paused and q.status != TrackingStatus.NO_FACE and self.s.system.auto_resume_after_face_lost:
            self.set_paused(False, "face_back")
            events.append("auto_resume")

        # Blink-driven pause works even while paused.
        if inp.blink_event == BlinkEvent.PAUSE_TOGGLE:
            self.toggle_pause("eyes_closed")
            events.append("pause_toggle")

        sigma_px = self.error_px
        point: FloatArray | None = None
        if inp.obs is not None and q.ok and not inp.eyes_closed:
            point, sigma_px = self._map(inp.obs, t)
        if inp.eyes_closed or q.eye_closed:
            # Freeze the cursor at its pre-blink position so the blink does not move the click point.
            if self._frozen is None and self.cursor is not None:
                self._frozen = self.cursor.copy()
        else:
            self._frozen = None

        dt = 0.0 if self._prev_t is None else max(t - self._prev_t, 0.0)
        self._prev_t = t
        if point is not None:
            self._last_valid_t = t
            self._lost_run = self._lid_run = 0.0
            self.cursor = point
        elif q.ok and not (inp.eyes_closed or q.eye_closed):
            # Face tracked, eyes open, sample dropped by the lid gate: the time counts. A gap
            # this long means the gaze could be anywhere by now (looking down at the lap):
            # the dwell starts again.
            self._lid_run += dt
            if self._lid_run > self.MAX_LID_GAP_S:
                self.dwell.reset()
                self._dwell_target = None
        else:
            # Tracking lost or eyes closed: does not count; a long loss starts the dwell again.
            self.dwell.hold(dt)
            self._lost_run += dt
            if self._lost_run > self.MAX_DWELL_GAP_S:
                self.dwell.reset()
                self._dwell_target = None
        cursor = self._frozen if self._frozen is not None else self.cursor

        # Corners: top-right opens the menu, top-left toggles pause.
        if cursor is not None and point is not None:
            corner = self._corner_event(t, cursor)
            if corner == "pause":
                self.toggle_pause("corner")
                events.append("pause_toggle")
            elif corner == "menu":
                events.append("open_menu")

        state = ControlState.PAUSED if self.paused else ControlState.IDLE
        progress = 0.0
        if not self.paused and cursor is not None:
            if point is not None:
                self.backend.move(int(round(cursor[0])), int(round(cursor[1])))
                if self.dragging:
                    events.append("dragging")
            # Clicking from blinks and gestures, at the frozen point.
            fired_by = self._discrete_click(inp, cursor, t, self._trusted(q, t))
            if fired_by:
                events.append(fired_by)
            # Scrolling zones.
            if point is not None and self._scroll(t, cursor):
                events.append("scroll")
            # Dwell.
            dwell_enabled = self.s.dwell.enabled and self.s.mode != ControlMode.HYBRID
            if point is not None and dwell_enabled and not self._in_scroll_zone(cursor):
                dwell_point = self._dwell_point(cursor)
                tgt = self._dwell_target
                self.dwell.dwell_s = self.s.dwell.dwell_ms / 1000.0 * (tgt.dwell_factor if tgt is not None else 1.0)
                if (tgt is not None and tgt.repeat and tgt.key == self._repeat_key
                        and self.dwell.state in (ControlState.FIRED, ControlState.COOLDOWN)
                        and t - self._repeat_t >= self.REPEAT_AFTER_S):
                    self.dwell.restart()  # still on the key that just fired: type it again
                upd = self.dwell.update(t, dwell_point)
                state, progress = upd.state, upd.progress
                if upd.fired and upd.anchor is not None:
                    events.append(self._fire(upd.anchor, t, source="dwell"))
            elif point is None:
                state = self.dwell.state
        elif self.paused and cursor is not None and point is not None:
            # While paused only the Pause buttons respond, so a user without hands can resume
            # by looking at Pause where they paused (the top-left corner works too). Nothing
            # else is pressed, clicked or typed.
            target = self.registry.hit(float(cursor[0]), float(cursor[1]))
            if target is not None and target.key.startswith(self.RESUME_TARGETS):
                self.dwell.dwell_s = self.s.dwell.dwell_ms / 1000.0 * target.dwell_factor
                upd = self.dwell.update(t, np.asarray(target.rect.center, dtype=np.float64))
                progress = upd.progress
                if upd.fired:
                    self._invoke_target(target, t)
                    events.append(f"target:{target.key}")
            else:
                self.dwell.reset()
                self.dwell.forget_if_left(cursor)  # looked away from Pause: it may be pressed again
        # Rest reminder every N minutes (PRD 7.7).
        period = self.s.system.rest_reminder_min * 60.0
        if period > 0 and t - self._start_t - self._last_rest_reminder >= period:
            self._last_rest_reminder = t - self._start_t
            events.append("rest_reminder")

        status = TrackingStatus.PAUSED if self.paused and q.status == TrackingStatus.TRACKING else q.status
        out_cursor = None if cursor is None else (int(round(cursor[0])), int(round(cursor[1])))
        for e in events:
            self._log(t, e)
        return EngineOutput(
            cursor=out_cursor,
            state=state,
            status=status,
            dwell_progress=progress,
            sigma_px=sigma_px,
            action=self.action,
            zoom=self.zoom,
            events=events,
            paused=self.paused,
            dragging=self.dragging,
        )

    # -- internals -------------------------------------------------------------
    def _map(self, obs: GazeObservation, t: float) -> tuple[FloatArray, float]:
        self.history.add(t, obs)
        raw = self.drift.apply(self.personalizer.predict_px(obs))
        sigma_px = self.personalizer.predict_sigma_px(obs) or self.error_px
        if obs.sigma is not None:
            # Network uncertainty (radians) -> pixels, combined with the calibration error.
            net_px = float(np.mean(obs.sigma)) * self.geometry.viewing_distance_mm * (
                sum(self.geometry.px_per_mm) / 2.0
            )
            sigma_px = math.hypot(sigma_px, net_px)
        filtered = self.filter(t, raw, sigma_px)
        fix = self.fixation.update(t, raw, filtered)
        if fix.saccade_started:
            # Jump at once without trailing (PRD 12.4).
            self.filter.reset()
            self.filter(t, raw, sigma_px)
            self._fine_anchor = None
        point = fix.point
        if self.s.mode == ControlMode.HEAD_FINE_TUNE and fix.event == EyeEvent.FIXATION:
            point = self._head_fine_tune(point, obs)
        margin = self.px_per_deg  # within 1 degree of an edge, snap onto it
        point = magnetic_edges(point, self.geometry.width_px, self.geometry.height_px, margin)
        return point, sigma_px

    def _dwell_point(self, cursor: FloatArray) -> FloatArray:
        """Where the dwell clicker should see the cursor. On GazeOS's own targets (keys,
        action bar) dwell counts per target: while the cursor stays on one it sees the
        target's centre, so wandering inside a large key does not restart the ring; a
        margin keeps the target through a one-frame wobble across its border."""
        x, y = float(cursor[0]), float(cursor[1])
        cur = self._dwell_target
        if cur is not None:
            r = cur.rect
            m = min(self.s.dwell.target_margin_deg * self.px_per_deg, 0.25 * min(r.width, r.height))
            if r.left - m <= x < r.left + r.width + m and r.top - m <= y < r.top + r.height + m:
                return np.asarray(r.center, dtype=np.float64)
        self._dwell_target = self.registry.hit(x, y)
        if self._dwell_target is None:
            return cursor
        return np.asarray(self._dwell_target.rect.center, dtype=np.float64)

    def _head_fine_tune(self, point: FloatArray, obs: GazeObservation) -> FloatArray:
        head = np.asarray(obs.head_pyr[:2], dtype=np.float64)
        if self._fine_anchor is None:
            self._fine_anchor = (point.copy(), head.copy())
            return point
        anchor_pt, anchor_head = self._fine_anchor
        d_deg = np.degrees(head - anchor_head)
        gain = self.s.system.head_fine_tune_gain_px_per_deg
        # Pitch up moves the cursor up; yaw moves it horizontally (camera +x is screen -x).
        return np.asarray(anchor_pt + np.array([d_deg[1] * gain, -d_deg[0] * gain]), dtype=np.float64)

    def _corner_event(self, t: float, cursor: FloatArray) -> str | None:
        size = self.s.system.corner_size_px
        w = self.geometry.width_px
        corners = {
            "menu": (cursor[0] >= w - size and cursor[1] <= size, self.s.system.corner_menu_dwell_ms),
            "pause": (cursor[0] <= size and cursor[1] <= size, self.s.system.corner_pause_dwell_ms),
        }
        for name, (inside, dwell_ms) in corners.items():
            if not inside:
                self._corner_since.pop(name, None)
                continue
            since = self._corner_since.setdefault(name, t)
            if t - since >= dwell_ms / 1000.0:
                self._corner_since[name] = t + 3600.0  # once per visit
                return name
        return None

    def _in_scroll_zone(self, cursor: FloatArray) -> bool:
        if not (self.s.scroll.always_on or self.action == Action.SCROLL):
            return False
        zone = self.s.scroll.zone_fraction * self.geometry.height_px
        return bool(cursor[1] < zone or cursor[1] > self.geometry.height_px - zone)

    def _scroll(self, t: float, cursor: FloatArray) -> bool:
        if not self._in_scroll_zone(cursor):
            self._zone_since = None
            return False
        if self._zone_since is None:
            self._zone_since = t
        if t - self._zone_since < self.s.scroll.zone_dwell_ms / 1000.0:
            return False
        if t - self._last_scroll_t < 1.0 / self.s.scroll.rate_hz:
            return False
        self._last_scroll_t = t
        zone = self.s.scroll.zone_fraction * self.geometry.height_px
        direction = -1 if cursor[1] < zone else 1
        self.backend.scroll(0, direction * self.s.scroll.step)
        return True

    def _trusted(self, q: QualityReport, t: float) -> bool:
        """May a blink or gesture act at the cursor? Only with one face in view and a valid
        gaze sample recent enough to be from just before the blink."""
        if q.status in (TrackingStatus.NO_FACE, TrackingStatus.MULTIPLE_FACES):
            return False
        if q.status == TrackingStatus.MOVE_CLOSER and not q.ok:  # too far to track (far but tracked is fine)
            return False
        limit = self.s.blink.long_blink_max_ms / 1000.0 + 0.5
        return self._last_valid_t is not None and t - self._last_valid_t <= limit

    def _discrete_click(self, inp: EngineInput, cursor: FloatArray, t: float, trusted: bool = True) -> str | None:
        mapping = self.s.gestures.mapping
        name: str | None = None
        if inp.blink_event in (BlinkEvent.LONG_BLINK, BlinkEvent.LEFT_WINK, BlinkEvent.RIGHT_WINK):
            name = inp.blink_event.value
        elif inp.gesture_event is not None and (self.s.gestures.enabled or self.s.mode == ControlMode.EYES_FACE):
            name = inp.gesture_event
        if name is None:
            return None
        action = mapping.get(name, "none")
        if action == "none":
            return None
        if action == "pause":
            self.toggle_pause("gesture")
            return "pause_toggle"
        if action == "drag_toggle" and self.dragging:  # letting go is always allowed
            self._end_drag()
            return f"{name}:drag"
        if not trusted:
            return None
        if action == "drag_toggle":
            if self.dragging:
                self._end_drag()
            else:
                self._start_drag(cursor)
            return f"{name}:drag"
        if action in ("scroll_up", "scroll_down"):
            self.backend.scroll(0, (-1 if action == "scroll_up" else 1) * self.s.scroll.step)
            return f"{name}:{action}"
        if action in ("left", "right", "double"):
            return self._fire(cursor, t, source=name, override=Action(action))
        return None

    def _start_drag(self, point: FloatArray) -> None:
        self.backend.move(int(point[0]), int(point[1]))
        self.backend.press("left")
        self.dragging = True

    def _end_drag(self) -> None:
        self.backend.release("left")
        self.dragging = False

    def _fire(self, anchor: FloatArray, t: float, source: str, override: Action | None = None) -> str:
        """Perform the current action at `anchor`. Returns an event name."""
        x, y = float(anchor[0]), float(anchor[1])
        action = override or self.action

        # A dwell inside the magnifier selects the exact original screen point,
        # which is clicked directly (no second round of snapping or zooming).
        from_zoom = False
        if self.zoom is not None:
            if self.zoom.in_view(x, y):
                x, y = self.zoom.view_to_screen(x, y)
                self.zoom = None
                from_zoom = True
            else:
                self.zoom = None
                return "zoom_cancel"

        # GazeOS's own targets (keyboard keys, action bar) handle themselves.
        target = self.registry.hit(x, y)
        if target is not None:
            self._invoke_target(target, t)
            return f"target:{target.key}"

        # Snapping or zoom-click for OS targets (PRD 7.3, 12.6).
        if self.s.snap.enabled and action != Action.DRAG and source == "dwell" and not from_zoom:
            radius = int(self.s.snap.radius_sigmas * self.error_px)
            elements = self.backend.elements_near(int(x), int(y), radius)
            decision = self.snapper.decide(np.array([x, y]), self.error_px, elements, now=t)
            if decision.element is not None and self.snapper.should_click_directly(decision, self.error_px):
                assert decision.point is not None
                x, y = decision.point
                self.snapper.note_used(decision.element, t)
            elif elements:
                # No confident target, or one smaller than the tracker's error: magnify
                # instead of clicking where the cursor happens to be.
                self.zoom = make_zoom_region(
                    (x, y), self.geometry.width_px, self.geometry.height_px,
                    self.s.snap.zoom_region_px, self.s.snap.zoom_factor,
                )
                self.dwell.rearm()
                return "zoom_open"

        self.backend.move(int(round(x)), int(round(y)))
        if action == Action.DRAG:
            if self.dragging:
                self._end_drag()
                self.action = Action.LEFT
                return f"{source}:drag_end"
            self._start_drag(np.array([x, y]))
            return f"{source}:drag_start"
        if action == Action.RIGHT:
            self.backend.click("right", 1)
        elif action == Action.DOUBLE:
            self.backend.click("left", 2)
        else:
            self.backend.click("left", 1)
        if action in (Action.RIGHT, Action.DOUBLE):
            self.action = Action.LEFT  # one-shot actions revert to left click
        self._implicit_sample(x, y, t)
        return f"{source}:{action.value}"

    def _invoke_target(self, target: GazeTarget, t: float) -> None:
        self._repeat_key, self._repeat_t = target.key, t
        try:
            target.callback()
        except Exception:
            log.exception("gaze target %s failed", target.key)
        if self.recalibrator is not None and self.s.calibration.implicit_enabled:
            self.recalibrator.on_click(t, self.history, target.rect.center)

    def _implicit_sample(self, x: float, y: float, t: float) -> None:
        if self.recalibrator is None or not self.s.calibration.implicit_enabled:
            return
        element = self.backend.element_at(int(x), int(y))
        if element is not None and element.invokable and element.rect.area < (self.error_px * 6) ** 2:
            self.recalibrator.on_click(t, self.history, element.rect.center)

    def _log(self, t: float, event: str) -> None:
        self.event_log.append((t, event))
        if len(self.event_log) > 10000:
            del self.event_log[:5000]
