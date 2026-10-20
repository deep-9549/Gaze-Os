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
from enum import Enum

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


class Action(str, Enum):
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
        self.filter: PointFilter = make_filter(settings.filter)
        self.fixation = FixationDetector(
            self.px_per_deg,
            settings.fixation.saccade_velocity_deg_s,
            settings.fixation.dispersion_deg,
            settings.fixation.min_fixation_ms,
        )
        self.dwell = DwellClicker(
            settings.dwell.dwell_ms, settings.dwell.radius_deg * self.px_per_deg, settings.dwell.cooldown_ms
        )
        self.snapper = TargetSnapper(settings.snap.probability_threshold, settings.snap.radius_sigmas)
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

    # -- public controls -------------------------------------------------------
    def set_action(self, action: Action) -> None:
        if self.dragging and action != Action.DRAG:
            self._end_drag()
        self.action = action

    def toggle_pause(self, reason: str = "user") -> None:
        self.set_paused(not self.paused, reason)

    def set_paused(self, paused: bool, reason: str = "user") -> None:
        if paused and self.dragging:
            self._end_drag()
        self.paused = paused
        self.auto_paused = paused and reason == "face_lost"
        self.dwell.reset()
        self.zoom = None
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

        if point is not None:
            self.cursor = point
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
            fired_by = self._discrete_click(inp, cursor, t)
            if fired_by:
                events.append(fired_by)
            # Scrolling zones.
            if point is not None and self._scroll(t, cursor):
                events.append("scroll")
            # Dwell.
            dwell_enabled = self.s.dwell.enabled and self.s.mode != ControlMode.HYBRID
            if point is not None and dwell_enabled and not self._in_scroll_zone(cursor):
                upd = self.dwell.update(t, cursor)
                state, progress = upd.state, upd.progress
                if upd.fired and upd.anchor is not None:
                    events.append(self._fire(upd.anchor, t, source="dwell"))
            elif point is None:
                state = self.dwell.state
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

    def _head_fine_tune(self, point: FloatArray, obs: GazeObservation) -> FloatArray:
        head = np.asarray(obs.head_pyr[:2], dtype=np.float64)
        if self._fine_anchor is None:
            self._fine_anchor = (point.copy(), head.copy())
            return point
        anchor_pt, anchor_head = self._fine_anchor
        d_deg = np.degrees(head - anchor_head)
        gain = self.s.system.head_fine_tune_gain_px_per_deg
        # Pitch up moves the cursor up; yaw moves it horizontally (camera +x is screen -x).
        return anchor_pt + np.array([d_deg[1] * gain, -d_deg[0] * gain])

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

    def _discrete_click(self, inp: EngineInput, cursor: FloatArray, t: float) -> str | None:
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
            elif elements and (decision.element is None or decision.point is None):
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
