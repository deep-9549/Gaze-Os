"""The desktop application: wires capture, pipeline, engine and every window together."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Slot
from PySide6.QtGui import QAction, QCursor, QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.calibration.implicit import ImplicitRecalibrator
from gazeos.calibration.personalization import Personalizer, load_personalizer, make_personalizer
from gazeos.calibration.protocol import CalibrationSample
from gazeos.capture.camera import CaptureThread, OpenCVSource
from gazeos.control.engine import Action, ControlEngine, EngineOutput
from gazeos.control.targets import TargetRegistry
from gazeos.os_backends.base import InputBackend, get_backend
from gazeos.os_backends.keys import shortcut_keys
from gazeos.pipeline import FrameProcessor, ProcessedFrame, ProcessingThread
from gazeos.profiles.logs import ExperimentLogger, HeatmapRecorder
from gazeos.profiles.store import Profile, ProfileStore
from gazeos.settings import Settings
from gazeos.text_entry.layouts import get_layout
from gazeos.text_entry.phrases import PhraseBank
from gazeos.text_entry.predictor import LocalLLM, Predictor
from gazeos.text_entry.session import TypingSession
from gazeos.text_entry.tts import make_speaker
from gazeos.ui.action_bar import ActionBar
from gazeos.ui.calibration import CalibrationOutcome, CalibrationWindow
from gazeos.ui.common import MainThreadBridge, TargetSync, color
from gazeos.ui.debug_view import DebugView
from gazeos.ui.keyboard import KeyboardWindow
from gazeos.ui.onboarding import Onboarding
from gazeos.ui.overlay import Overlay
from gazeos.ui.panels import ButtonPanel, ConfirmPanel, PrivacyPanel
from gazeos.ui.settings_dialog import SettingsDialog
from gazeos.vision.head_pose import CameraIntrinsics

log = logging.getLogger(__name__)


def screen_geometry(settings: Settings) -> ScreenGeometry:
    """Physical screen size and resolution from the OS, falling back to settings."""
    screen = QGuiApplication.primaryScreen()
    s = settings.screen
    if screen is None:
        return ScreenGeometry.from_settings(s)
    ratio = screen.devicePixelRatio()
    g = screen.geometry()
    mm = screen.physicalSize()
    width_mm = mm.width() if mm.width() > 50 else s.width_mm
    height_mm = mm.height() if mm.height() > 30 else s.height_mm
    return ScreenGeometry(
        width_px=int(g.width() * ratio), height_px=int(g.height() * ratio),
        width_mm=width_mm, height_mm=height_mm,
        camera_offset_mm=s.camera_offset_mm, viewing_distance_mm=s.viewing_distance_mm,
    )


class SimulatedCapture:
    """--simulate: a synthetic face looks wherever the mouse is. No camera needed."""

    def __init__(self, processor_geometry: ScreenGeometry, intrinsics: CameraIntrinsics, fps: float = 30.0) -> None:
        from gazeos.sim import SimulatedUser, SyntheticFace

        self.user = SimulatedUser(processor_geometry, SyntheticFace(intrinsics, noise_px=0.25))
        self.period = 1.0 / fps
        self._next = time.monotonic()
        self.ratio = 1.0
        self.mouse = (processor_geometry.width_px / 2, processor_geometry.height_px / 2)
        self.eyes_closed_until = 0.0

    def latest(self, timeout: float = 0.5) -> object:
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + self.period, time.monotonic())
        openness = 0.0 if time.monotonic() < self.eyes_closed_until else 1.0
        lm = self.user.look_at(*self.mouse, open_left=openness, open_right=openness)

        class _F:
            pass

        f = _F()
        f.landmarks = lm  # type: ignore[attr-defined]
        f.t_capture = time.monotonic()  # type: ignore[attr-defined]
        f.image_rgb = np.full((480, 640, 3), 128, np.uint8)  # type: ignore[attr-defined]
        return f

    def stop(self) -> None:
        pass


class GazeOSApp(QObject):
    def __init__(self, args: object) -> None:
        super().__init__()
        self.args = args
        app = QApplication.instance()
        self.qapp: QApplication = app if isinstance(app, QApplication) else QApplication(sys.argv)
        self.qapp.setQuitOnLastWindowClosed(False)
        self.store = ProfileStore(getattr(args, "home", None))
        name = getattr(args, "profile", None) or self.store.last_used or "default"
        self.profile: Profile = self.store.get_or_create(name)
        self.store.last_used = name
        self.settings: Settings = self.profile.load_settings()
        if getattr(args, "camera", None) is not None:
            self.settings.camera.device_index = args.camera  # type: ignore[attr-defined]
        self.simulate = bool(getattr(args, "simulate", False))
        self.geometry = screen_geometry(self.settings)
        self.intrinsics = (
            CameraIntrinsics.load(self.settings.camera.intrinsics_file)
            if self.settings.camera.intrinsics_file and Path(self.settings.camera.intrinsics_file).exists()
            else CameraIntrinsics.default(self.settings.camera.width, self.settings.camera.height)
        )
        self.backend: InputBackend = get_backend(dry_run=self.simulate or bool(getattr(args, "dry_run", False)))
        self.registry = TargetRegistry()
        self.bridge = MainThreadBridge()
        self.bridge.frame.connect(self._on_frame)
        self.sync = TargetSync(self.registry, self.bridge)
        self.samples: list[CalibrationSample] = []
        self.personalizer = self._load_personalizer()
        self.recalibrator = ImplicitRecalibrator(
            self.personalizer, self.samples, self.settings.calibration.implicit_refit_every,
            self.settings.calibration.implicit_decay_min, on_refit=self._on_refit,
        )
        self.engine = ControlEngine(
            self.settings, self.geometry, self.backend, self.personalizer, self.registry,
            self.recalibrator, confirm=self._confirm_destructive,
        )
        self.processor = self._make_processor()
        self.calibration: CalibrationWindow | None = None
        self.keyboard: KeyboardWindow | None = None
        self.onboarding: Onboarding | None = None
        self.logger = ExperimentLogger(self.profile.logs_dir) if self.settings.privacy.experiment_logging else None
        self.heatmap = (HeatmapRecorder(self.geometry.width_px, self.geometry.height_px)
                        if self.settings.privacy.heatmaps else None)
        self._build_ui()
        self._start_threads()
        self._install_hotkeys()

    # -- construction ------------------------------------------------------------
    def _load_personalizer(self) -> Personalizer:
        if self.profile.has_calibration():
            try:
                p = load_personalizer(self.profile.calibration_file, self.geometry)
                return p
            except Exception:
                log.exception("could not load calibration; starting uncalibrated")
        return make_personalizer(self.settings.calibration.method, self.geometry)

    def _make_processor(self) -> FrameProcessor:
        gaze_model = gesture_model = None
        landmarker = None
        if not self.simulate:
            from gazeos.vision.landmarks import FaceLandmarker

            landmarker = FaceLandmarker()
            model_path = Path(self.settings.gaze_model)
            if model_path.exists():
                from gazeos.inference.gaze import OnnxGazeEstimator

                gaze_model = OnnxGazeEstimator(model_path)
                log.info("gaze model: %s", model_path)
            else:
                log.warning("gaze model %s not found; using the geometric M0 baseline", model_path)
        if self.settings.gesture_model and Path(self.settings.gesture_model).exists():
            from gazeos.inference.gesture import OnnxGestureClassifier

            gesture_model = OnnxGestureClassifier(self.settings.gesture_model)
        return FrameProcessor(self.settings, self.geometry, self.intrinsics, landmarker, gaze_model, gesture_model)

    def _build_ui(self) -> None:
        th = self.settings.theme
        self.overlay = Overlay(th)
        if self.settings.system.show_overlay:
            self.overlay.show()
        if self.settings.privacy.research_recording:
            self.overlay.banner = "Research recording is ON"
        self.action_bar = ActionBar(self._set_action, self.toggle_keyboard, self._show_shortcuts,
                                    self.toggle_pause, self._show_menu, th)
        self.action_bar.show()
        self.sync.track("action_bar", self.action_bar)
        self.debug = DebugView(self.intrinsics)
        self.menu = ButtonPanel("GazeOS", [
            ("Calibrate", "menu_calibrate", self.start_calibration),
            ("Quick fix (1 point)", "menu_quick", lambda: self.start_calibration(quick=True)),
            ("Keyboard", "menu_keyboard", self.toggle_keyboard),
            ("Settings", "menu_settings", self._show_settings),
            ("Privacy", "menu_privacy", self._show_privacy),
            ("Debug view", "menu_debug", self._toggle_debug),
            ("Pause / resume", "menu_pause", self.toggle_pause),
            ("Hide overlay", "menu_overlay", self._toggle_overlay),
            ("Quit GazeOS", "menu_quit", self.quit),
        ], th)
        self.shortcuts = ButtonPanel("Shortcuts", [
            (label, f"sc_{name}", self._shortcut_runner(name))
            for label, name in (("Copy", "copy"), ("Paste", "paste"), ("Cut", "cut"), ("Undo", "undo"),
                                ("Select all", "select_all"), ("Switch window", "switch_window"),
                                ("Close window", "close_window"), ("Start menu", "start_menu"),
                                ("New tab", "new_tab"), ("Close tab", "close_tab"))
        ], th, columns=4)
        for group, w in (("menu", self.menu), ("shortcuts", self.shortcuts)):
            self.sync.track(group, w)
        self._tray()
        if not self.profile.has_calibration() and not self.simulate:
            QTimer.singleShot(500, self.start_onboarding)

    def _tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        pm = QPixmap(32, 32)
        pm.fill(color(self.settings.theme, "accent"))
        self.tray = QSystemTrayIcon(QIcon(pm))
        menu = QMenu()
        for label, cb in (("Pause / resume", self.toggle_pause), ("Calibrate", self.start_calibration),
                          ("Settings", self._show_settings), ("Quit", self.quit)):
            act = QAction(label, menu)
            act.triggered.connect(cb)
            menu.addAction(act)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("GazeOS")
        self.tray.show()

    def _start_threads(self) -> None:
        if self.simulate:
            self.capture: SimulatedCapture | CaptureThread = SimulatedCapture(self.geometry, self.intrinsics)
            self._sim_timer = QTimer(self)
            self._sim_timer.timeout.connect(self._sim_mouse)
            self._sim_timer.start(15)
            self.worker: ProcessingThread = _SimThread(self.capture, self.processor, self.engine, self._frame_from_thread,
                                     self._observation_hook)
        else:
            c = self.settings.camera
            source = OpenCVSource(c.device_index, c.width, c.height, c.fps)
            capture = CaptureThread(source)
            capture.start()
            self.capture = capture
            if c.lock_exposure:
                QTimer.singleShot(3000, source.lock_exposure)  # after auto exposure settles
            self.worker = ProcessingThread(self.capture, self.processor, self.engine, self._frame_from_thread,
                                           self._observation_hook)
        self.worker.start()

    def _install_hotkeys(self) -> None:
        """Ctrl+Alt+G emergency stop always works (PRD 18.3); Ctrl+Alt+P pauses."""
        try:
            from pynput import keyboard

            def fmt(h: str) -> str:
                return "+".join(f"<{p}>" if len(p) > 1 else p for p in h.split("+"))

            self._hotkeys = keyboard.GlobalHotKeys({
                fmt(self.settings.system.emergency_hotkey): self.bridge.later(self.emergency_stop),
                fmt(self.settings.system.pause_hotkey): self.bridge.later(self.toggle_pause),
            })
            self._hotkeys.start()
        except Exception as exc:
            log.warning("global hotkeys unavailable: %s", exc)

    # -- frame flow --------------------------------------------------------------
    def _observation_hook(self, pf: ProcessedFrame) -> None:
        cal = self.calibration
        if cal is not None:
            cal.add_frame(pf.result.t_capture, pf.observation, pf.engine_input.eyes_closed)

    def _frame_from_thread(self, pf: ProcessedFrame, out: EngineOutput | None, image: np.ndarray) -> None:
        self.bridge.frame.emit(pf, out, image)

    @Slot(object, object, object)
    def _on_frame(self, pf: ProcessedFrame, out: EngineOutput | None, image: np.ndarray) -> None:
        if out is not None:
            self.overlay.set_output(out)
            self.action_bar.set_action(out.action)
            self.action_bar.set_paused(out.paused)
            if self.keyboard is not None:
                self.keyboard.last_gaze_px = out.cursor
            for e in out.events:
                self._handle_event(e)
            if self.heatmap is not None and out.cursor is not None and not out.paused:
                self.heatmap.add(*out.cursor)
            if self.logger is not None:
                self.logger.log_frame(pf.result.t_capture, out.cursor, out.state.value, out.status.value,
                                      pf.result.timings_ms, out.sigma_px)
        self.debug.fps = getattr(self.worker, "fps", 0.0)
        self.debug.update_frame(pf, image)
        if self.onboarding is not None:
            self.onboarding.update_frame(pf, image, self.processor.blinks.last_closure_ms)

    def _handle_event(self, e: str) -> None:
        if e == "open_menu":
            self._show_menu()
        elif e == "rest_reminder":
            self.overlay.show_toast("Time for a short rest: look away from the screen for 20 seconds.", 8)
        elif e == "auto_pause":
            self.overlay.show_toast("Face lost: tracking paused", 3)
        elif e == "auto_resume":
            self.overlay.show_toast("Face found: tracking resumed", 2)
        elif e == "zoom_open":
            self.overlay.show_toast("Zoom: look at your target again", 2)
        if self.recalibrator.drift_suspected():
            self.overlay.show_toast("Accuracy dropped. Open the menu and choose Quick fix.", 5)
            self.recalibrator.baseline_error_px = self.recalibrator.current_error_px()

    # -- actions -----------------------------------------------------------------
    def _set_action(self, action: Action) -> None:
        with self.worker.with_engine():
            self.engine.set_action(action)

    def toggle_pause(self) -> None:
        with self.worker.with_engine():
            self.engine.toggle_pause("user")

    def emergency_stop(self) -> None:
        with self.worker.with_engine():
            self.engine.emergency_stop()
        self.overlay.show_toast("Emergency stop: all input stopped. Ctrl+Alt+P resumes.", 6)

    def _confirm_destructive(self, name: str) -> bool:
        """Called from the engine thread: ask on the GUI thread, run the shortcut on Yes."""

        def run() -> None:
            if not self.engine.paused:
                self.backend.hotkey(*shortcut_keys(name, sys.platform))

        def ask() -> None:
            question = f"Really {name.replace('_', ' ')}? Unsaved work may be lost."
            self._confirm_panel = ConfirmPanel(question, run, self.settings.theme)
            self.sync.track("confirm", self._confirm_panel)
            self._confirm_panel.show()

        self.bridge.call.emit(ask)
        return False

    def _shortcut_runner(self, name: str) -> Callable[[], None]:
        def run() -> None:
            self.engine.run_shortcut(name)

        return run

    def _show_menu(self) -> None:
        self.menu.show()

    def _show_shortcuts(self) -> None:
        self.shortcuts.show()

    def _toggle_overlay(self) -> None:
        self.overlay.show_cursor = not self.overlay.show_cursor

    def _toggle_debug(self) -> None:
        self.debug.setVisible(not self.debug.isVisible())

    def toggle_keyboard(self) -> None:
        if self.keyboard is None:
            layout = get_layout(self.settings.typing.layout)
            predictor = Predictor.default(layout, self.settings.typing.num_predictions)
            if self.settings.typing.use_llm:
                predictor.llm = LocalLLM(self.settings.typing.llm_model)
            if self.profile.dictionary_file.exists():
                for line in self.profile.dictionary_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        predictor.trie.add(line.strip(), 5.0)
            session = TypingSession(layout, predictor, emit=self.engine.type_text,
                                    emit_key=lambda k: None if self.engine.paused else self.backend.hotkey(k))
            self.keyboard = KeyboardWindow(
                session, PhraseBank(self.profile.phrases_file),
                make_speaker(self.settings.typing.tts_engine, self.settings.typing.piper_voice),
                self.engine.error_px, self.toggle_keyboard, self.settings.theme, self.settings.typing.num_predictions,
            )
        if self.keyboard.isVisible():
            self.keyboard.hide()
            self.sync.untrack("keyboard")
        else:
            self.keyboard.show()
            self.sync.track("keyboard", self.keyboard)

    def _show_settings(self) -> None:
        self._settings_dialog = SettingsDialog(self.settings, self._save_settings, self.settings.theme)
        self.sync.track("settings", self._settings_dialog)
        self._settings_dialog.show()

    def _save_settings(self, settings: Settings) -> None:
        self.profile.save_settings(settings)
        if settings.system.start_with_os != self.settings.system.start_with_os:
            try:
                from gazeos.autostart import set_enabled

                set_enabled(settings.system.start_with_os)
            except OSError as exc:
                log.warning("could not change start-with-OS: %s", exc)
        self.overlay.show_toast("Settings saved. Restart GazeOS to apply camera and model changes.", 4)
        self.settings = settings
        with self.worker.with_engine():
            self.engine.s = settings

    def _show_privacy(self) -> None:
        self._privacy = PrivacyPanel(self.profile, self._data_deleted, self.settings.theme)
        self.sync.track("privacy", self._privacy)
        self._privacy.show()

    def _data_deleted(self) -> None:
        self.overlay.show_toast("All data for this profile was deleted.", 4)
        self.profile = self.store.get_or_create(self.profile.name)

    # -- calibration ---------------------------------------------------------------
    def start_onboarding(self) -> None:
        self.onboarding = Onboarding(self.start_calibration, self._onboarding_done, self.settings.theme)
        self.sync.track("onboarding", self.onboarding)
        self.onboarding.show()

    def _onboarding_done(self, threshold_ms: float | None) -> None:
        if threshold_ms is not None:
            self.settings.blink.long_blink_min_ms = threshold_ms
            self.profile.save_settings(self.settings)
        self.sync.untrack("onboarding")
        self.onboarding = None

    def start_calibration(self, quick: bool = False) -> None:
        with self.worker.with_engine():
            self.worker.engine_enabled = False  # no clicks while calibrating
        p = self.personalizer if quick else make_personalizer(self.settings.calibration.method, self.geometry)
        self.calibration = CalibrationWindow(
            self.geometry, p, self.settings.calibration, self._calibration_done, self.settings.theme,
            quick=quick,
            current_predict=(lambda obs: self.engine.drift.apply(self.personalizer.predict_px(obs))) if quick else None,
        )
        self.sync.track("calibration", self.calibration)
        self.calibration.showFullScreen()

    def _calibration_done(self, outcome: CalibrationOutcome | None) -> None:
        self.sync.untrack("calibration")
        self.calibration = None
        with self.worker.with_engine():
            self.worker.engine_enabled = True
            if outcome is not None:
                if outcome.quick_offset is not None:
                    self.engine.drift.offset = self.engine.drift.offset + outcome.quick_offset
                else:
                    self._apply_calibration(outcome)
        if self.onboarding is not None:
            v = outcome.validation if outcome is not None and outcome.quick_offset is None else None
            self.onboarding.calibration_done(v.score if v else None, v.mean_deg if v else None)
        if outcome is not None and outcome.quick_offset is None:
            self.overlay.show_toast(f"Calibration score {outcome.validation.score}/100", 4)

    def _apply_calibration(self, outcome: CalibrationOutcome) -> None:
        self.personalizer = outcome.personalizer
        self.samples = outcome.samples
        self.engine.personalizer = self.personalizer
        self.engine.drift.reset()
        self.engine.error_px = max(outcome.validation.mean_px, 20.0)
        self.recalibrator.personalizer = self.personalizer
        self.recalibrator.base_samples = list(outcome.samples)
        self.recalibrator.baseline_error_px = outcome.validation.mean_px
        self.personalizer.save(self.profile.calibration_file)
        self.profile.save_calibration_features(
            pitchyaw=np.stack([s.obs.pitchyaw for s in outcome.samples]),
            origin=np.stack([s.obs.origin for s in outcome.samples]),
            screen_px=np.stack([s.obs.screen_px for s in outcome.samples]),
            target_px=np.stack([s.target_px for s in outcome.samples]),
        )
        if self.keyboard is not None:
            self.keyboard.build(self.keyboard.session.layout, self.engine.error_px)

    def _on_refit(self, personalizer: Personalizer) -> None:
        self.bridge.call.emit(lambda: personalizer.save(self.profile.calibration_file))

    # -- simulate ----------------------------------------------------------------
    def _sim_mouse(self) -> None:
        pos = QCursor.pos()
        screen = QGuiApplication.primaryScreen()
        r = screen.devicePixelRatio() if screen else 1.0
        if isinstance(self.capture, SimulatedCapture):
            self.capture.mouse = (pos.x() * r, pos.y() * r)

    # -- lifecycle -----------------------------------------------------------------
    def quit(self) -> None:
        self.worker.stop()
        self.capture.stop()
        if self.logger is not None:
            self.logger.close()
        if self.heatmap is not None:
            self.heatmap.save(self.profile.heatmaps_dir / time.strftime("heatmap-%Y%m%d-%H%M%S"))
        if hasattr(self, "_hotkeys"):
            self._hotkeys.stop()
        self.qapp.quit()

    def run(self) -> int:
        return int(self.qapp.exec())


class _SimThread(ProcessingThread):
    """Processing thread for --simulate: frames carry landmarks instead of images."""

    def run(self) -> None:
        last, frames = time.monotonic(), 0
        while not self._stop_evt.is_set():
            frame = self.capture.latest()  # type: ignore[attr-defined]
            try:
                pf = self.processor.process_landmarks(frame.landmarks, frame.t_capture)
                if self.observation_hook is not None:
                    self.observation_hook(pf)
                out = None
                with self._lock:
                    if self.engine is not None and self.engine_enabled:
                        out = self.engine.update(pf.engine_input)
                if out is not None:
                    pf.result.screen_xy = out.cursor
                    pf.result.state = out.state.value
                if self.on_frame is not None:
                    self.on_frame(pf, out, frame.image_rgb)
            except Exception:
                log.exception("simulated frame failed")
            frames += 1
            now = time.monotonic()
            if now - last >= 1.0:
                self.fps, frames, last = frames / (now - last), 0, now

