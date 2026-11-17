"""The per-frame pipeline (PRD 8): capture -> landmarks -> head pose and
normalization -> gaze and gesture networks -> control engine.

`FrameProcessor` is pure (no threads) so it can be driven by the live camera,
by recorded sessions in replay tests (PRD 23), or by the benchmark tools.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.calibration.protocol import GazeObservation
from gazeos.control.engine import ControlEngine, EngineInput, EngineOutput
from gazeos.control.filters import OneEuroFilter
from gazeos.gestures.blink import BlinkDetector, BlinkEvent, LidGate, eye_state
from gazeos.gestures.events import FeatureWindow, GestureEventDetector, scale_by_gesture_size
from gazeos.inference.gaze import GazeEstimate, GeometricGazeEstimator
from gazeos.settings import Settings
from gazeos.types import FrameResult
from gazeos.vision import features as feat
from gazeos.vision.enhance import mean_brightness
from gazeos.vision.geometry import head_direction_pitchyaw, rotation_to_euler
from gazeos.vision.head_pose import CameraIntrinsics, HeadPose, HeadPoseEstimator
from gazeos.vision.landmarks import LandmarkResult
from gazeos.vision.normalization import NormalizationResult, normalize, warp_matrix
from gazeos.vision.quality import QualityGate, QualityReport

log = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


class LandmarkSource(Protocol):
    def process(self, image_rgb: NDArray[np.uint8], t_seconds: float) -> LandmarkResult: ...


class NetworkGaze(Protocol):
    def estimate(self, norm: NormalizationResult) -> GazeEstimate: ...


class GestureModel(Protocol):
    def predict(self, window: NDArray[np.float32]) -> dict[str, float]: ...


@dataclass
class ProcessedFrame:
    result: FrameResult
    engine_input: EngineInput
    norm: NormalizationResult | None = None
    pose: HeadPose | None = None
    estimate: GazeEstimate | None = None
    observation: GazeObservation | None = None


class FrameProcessor:
    def __init__(
        self,
        settings: Settings,
        geometry: ScreenGeometry,
        intrinsics: CameraIntrinsics,
        landmarker: LandmarkSource | None,
        gaze_model: NetworkGaze | None = None,
        gesture_model: GestureModel | None = None,
    ) -> None:
        self.s = settings
        self.geometry = geometry
        self.intrinsics = intrinsics
        self.landmarker = landmarker
        self.gaze_model = gaze_model
        self.geometric = GeometricGazeEstimator()
        self.gesture_model = gesture_model
        self.head = HeadPoseEstimator(intrinsics, settings.quality.head_pose_disagreement_deg)
        self.head_filter = OneEuroFilter(1.0, 0.01, 1.0)
        self.quality = QualityGate(settings.quality)
        self.blinks = BlinkDetector(settings.blink)
        self.window = FeatureWindow()
        self.gestures = GestureEventDetector(settings.gestures.probability_threshold, settings.gestures.refractory_ms)
        self._prev_head: tuple[float, float, float] | None = None
        q = settings.quality
        self.lids = LidGate(q.lid_hidden_level, q.lid_speed, q.lid_hold_frames)

    def apply_settings(self, settings: Settings) -> None:
        """Saved settings take effect at once (the gates and detectors copy their values)."""
        self.s = settings
        self.quality = QualityGate(settings.quality)
        self.blinks = BlinkDetector(settings.blink)
        self.gestures = GestureEventDetector(settings.gestures.probability_threshold, settings.gestures.refractory_ms)
        q = settings.quality
        self.lids = LidGate(q.lid_hidden_level, q.lid_speed, q.lid_hold_frames)

    def process_image(self, image_rgb: NDArray[np.uint8], t: float) -> ProcessedFrame:
        if self.landmarker is None:
            raise RuntimeError("no landmarker configured")
        t0 = time.perf_counter()
        lm = self.landmarker.process(image_rgb, t)
        t1 = time.perf_counter()
        out = self.process_landmarks(lm, t, image_rgb)
        out.result.timings_ms["landmarks"] = (t1 - t0) * 1000.0
        return out

    def process_landmarks(
        self, lm: LandmarkResult, t: float, image_rgb: NDArray[np.uint8] | None = None,
        crops: tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]] | None = None,
    ) -> ProcessedFrame:
        """Everything after face landmarks; image is needed for the CNN and brightness.

        `crops` (normalized face, right eye, left eye) replays a research recording through
        the CNN without the camera image: the crops are the saved ones, the normalization
        rotation is recomputed from the landmarks."""
        timings: dict[str, float] = {}
        res = FrameResult(t_capture=t, num_faces=lm.num_faces)
        if image_rgb is not None:
            res.image_size = (image_rgb.shape[1], image_rgb.shape[0])
        if lm.num_faces == 0 or lm.landmarks_px is None:
            q = self.quality.check(t, 0)
            self.head.reset()
            self.blinks.reset()
            self.lids.reset()
            res.status = q.status.value
            res.quality_ok = False
            return ProcessedFrame(res, EngineInput(t, None, q))

        landmarks = lm.landmarks_px
        res.face_found = True
        res.landmarks = landmarks
        res.blendshapes = lm.blendshapes

        ta = time.perf_counter()
        pose = self.head.estimate(landmarks, lm.transform)
        if pose is None:
            q = QualityReport(ok=False, status=self.quality.check(t, lm.num_faces).status, hold_cursor=True)
            return ProcessedFrame(res, EngineInput(t, None, q))
        # Smooth head pose (PRD 9.3) through its Euler angles.
        pyr = rotation_to_euler(pose.R)
        res.head_R, res.head_t = pose.R, pose.t
        head_dir = head_direction_pitchyaw(pose.R)

        # Eyes and blinks.
        ear_l, ear_r = feat.ear_pair(landmarks)
        state = eye_state(lm.blendshapes, ear_l, ear_r, self.s.blink.ear_threshold, self.s.blink.closed_score)
        blink_score = max(lm.blendshapes.get("eyeBlinkLeft", 0.0), lm.blendshapes.get("eyeBlinkRight", 0.0)) \
            if lm.blendshapes else (1.0 if state.both_closed else 0.0)
        both_score = min(lm.blendshapes.get("eyeBlinkLeft", 0.0), lm.blendshapes.get("eyeBlinkRight", 0.0)) \
            if lm.blendshapes else None
        blink_event = self.blinks.update(t, state, both_score)
        lid_down = not self.lids.update(blink_score)

        brightness = 255.0
        if image_rgb is not None:
            brightness = mean_brightness(image_rgb, feat.face_bbox(landmarks))
        q = self.quality.check(
            t, lm.num_faces, feat.inter_eye_distance_px(landmarks), head_dir[0], head_dir[1],
            blink_score if state.both_closed else 0.0, brightness,
        )
        if pose.low_confidence:
            q.lower_confidence = True
        timings["head_pose"] = (time.perf_counter() - ta) * 1000.0

        # Gestures: 17-d features over a 30-frame window.
        gesture_event = None
        delta = (0.0, 0.0, 0.0) if self._prev_head is None else tuple(
            float(a - b) for a, b in zip(pyr, self._prev_head, strict=True))
        self._prev_head = pyr
        self.window.push(feat.gesture_features(landmarks, lm.blendshapes, delta))
        if self.gesture_model is not None and self.s.gestures.enabled:
            tg = time.perf_counter()
            probs = self.gesture_model.predict(self.window.array())
            probs = scale_by_gesture_size(probs, self.s.gestures.gesture_size)
            res.gesture_probs = probs
            gesture_event = self.gestures.update(t, probs)
            timings["gesture"] = (time.perf_counter() - tg) * 1000.0
            # The learned classifier owns blink events when enabled.
            if gesture_event in ("long_blink", "left_wink", "right_wink"):
                blink_event = BlinkEvent(gesture_event)
                gesture_event = None

        # Gaze.
        norm = estimate = obs = None
        if q.ok:
            tg = time.perf_counter()
            if self.gaze_model is not None and (image_rgb is not None or crops is not None):
                if image_rgb is not None:
                    norm = normalize(image_rgb, landmarks, pose, self.intrinsics)
                else:
                    assert crops is not None
                    W, R_n, S = warp_matrix(pose.face_center, pose.R, self.intrinsics)
                    norm = NormalizationResult(crops[0], crops[1], crops[2], R_n, S, W,
                                               rotation_to_euler(R_n @ pose.R))
                timings["normalize"] = (time.perf_counter() - tg) * 1000.0
                tn = time.perf_counter()
                estimate = self.gaze_model.estimate(norm)
                timings["gaze_net"] = (time.perf_counter() - tn) * 1000.0
            else:
                estimate = self.geometric.estimate(landmarks, pose)
                timings["gaze_geometric"] = (time.perf_counter() - tg) * 1000.0
            origin = pose.gaze_origin
            from gazeos.vision.geometry import pitchyaw_to_vector

            px = self.geometry.gaze_to_px(origin, pitchyaw_to_vector(estimate.pitchyaw))
            if px is not None:
                obs = GazeObservation(
                    pitchyaw=estimate.pitchyaw,
                    origin=origin,
                    screen_px=np.asarray(px, dtype=np.float64),
                    head_pyr=np.asarray(head_dir + (pyr[2],), dtype=np.float64),
                    features=estimate.features,
                    sigma=estimate.sigma,
                )
                res.gaze_pitch_yaw = (float(estimate.pitchyaw[0]), float(estimate.pitchyaw[1]))
                res.gaze_sigma = (float(estimate.sigma[0]), float(estimate.sigma[1]))
                res.features = estimate.features
        res.status = q.status.value
        res.quality_ok = q.ok
        res.eye_closed = state.both_closed
        res.timings_ms.update(timings)
        res.lid_down = lid_down
        res.blink_score = blink_score
        if lid_down:  # recorded in res for analysis, but neither cursor nor calibration use it
            obs = None
        inp = EngineInput(t, obs, q, blink_event=blink_event, gesture_event=gesture_event,
                          eyes_closed=state.both_closed)
        return ProcessedFrame(res, inp, norm, pose, estimate, obs)


class ProcessingThread(threading.Thread):
    """Runs capture -> processing -> engine; never blocks on the UI (PRD 8.1)."""

    def __init__(
        self,
        capture: object,
        processor: FrameProcessor,
        engine: ControlEngine | None,
        on_frame: Callable[[ProcessedFrame, EngineOutput | None, NDArray[np.uint8]], None] | None = None,
        observation_hook: Callable[[ProcessedFrame], None] | None = None,
    ) -> None:
        super().__init__(name="gazeos-processing", daemon=True)
        self.capture = capture
        self.processor = processor
        self.engine = engine
        self.on_frame = on_frame
        # Calibration and data collection hook in here instead of the engine.
        self.observation_hook = observation_hook
        self.engine_enabled = True
        self._stop_evt = threading.Event()
        self.fps = 0.0
        self._lock = threading.Lock()
        self.stalled = False
        # GAZEOS_TRACE=path.csv: one line per frame with every stage of the cursor, for
        # finding where a screen area becomes unreachable (fable_opinion.md 4.7).
        import os

        self._trace = None
        if os.environ.get("GAZEOS_TRACE"):
            self._trace = open(os.environ["GAZEOS_TRACE"], "w", buffering=1)  # noqa: SIM115
            self._trace.write("t,status,quality_ok,lid_down,blink,raw_x,raw_y,cal_x,cal_y,cursor_x,cursor_y,state,dwell_key,paused\n")
            self._trace_targets_t = 0.0

    def _write_trace(self, pf: ProcessedFrame, out: EngineOutput | None) -> None:
        r, o, eng = pf.result, pf.engine_input.obs, self.engine
        raw = cal = (float("nan"), float("nan"))
        if o is not None and eng is not None:
            raw = (float(o.screen_px[0]), float(o.screen_px[1]))
            c = eng.drift.apply(eng.personalizer.predict_px(o))
            cal = (float(c[0]), float(c[1]))
        cur = out.cursor if out is not None and out.cursor is not None else (float("nan"), float("nan"))
        key = eng._dwell_target.key if eng is not None and eng._dwell_target is not None else ""
        assert self._trace is not None
        self._trace.write(f"{r.t_capture:.4f},{r.status},{int(r.quality_ok)},{int(getattr(r, 'lid_down', False))},"
                          f"{getattr(r, 'blink_score', 0.0):.3f},{raw[0]:.1f},{raw[1]:.1f},{cal[0]:.1f},{cal[1]:.1f},"
                          f"{cur[0]},{cur[1]},{out.state.value if out else ''},{key},{int(out.paused) if out else ''}\n")
        if eng is not None and r.t_capture - self._trace_targets_t > 2.0:  # where GazeOS's own buttons are
            self._trace_targets_t = r.t_capture
            import json

            with open(self._trace.name + ".targets.json", "w") as fh:
                json.dump({t.key: [t.rect.left, t.rect.top, t.rect.width, t.rect.height] for t in eng.registry.all()}, fh)

    # No frame for this long, or this many frames failing in a row: stop acting.
    STALL_S = 1.0
    STALL_FAILURES = 15

    def _stall(self) -> None:
        """The camera stopped or processing keeps failing. The engine only sees frames, so
        without this a held drag or a running dwell would stay as it was. Pauses like a
        lost face, so control resumes by itself when frames return (if that is enabled)."""
        self.stalled = True
        with self._lock:
            if self.engine is not None and not self.engine.paused:
                self.engine.set_paused(True, "face_lost")
        log.warning("no usable camera frames; input paused")

    def run(self) -> None:
        last = time.monotonic()
        frames = 0
        last_frame = time.monotonic()
        failures = 0
        while not self._stop_evt.is_set():
            frame = self.capture.latest(timeout=0.5)  # type: ignore[attr-defined]
            if frame is None:
                if not self.stalled and time.monotonic() - last_frame > self.STALL_S:
                    self._stall()
                continue
            last_frame = time.monotonic()
            t0 = time.perf_counter()
            try:
                pf = self.processor.process_image(frame.image_rgb, frame.t_capture)
                if self.observation_hook is not None:
                    self.observation_hook(pf)
                out = None
                with self._lock:
                    if self.engine is not None and self.engine_enabled:
                        out = self.engine.update(pf.engine_input)
                pf.result.timings_ms["total"] = (time.perf_counter() - t0) * 1000.0
                pf.result.timings_ms["latency"] = (time.monotonic() - frame.t_capture) * 1000.0
                if out is not None:
                    pf.result.screen_xy = out.cursor
                    pf.result.state = out.state.value
                if self._trace is not None:
                    self._write_trace(pf, out)
                if self.on_frame is not None:
                    self.on_frame(pf, out, frame.image_rgb)
                failures = 0
                self.stalled = False
            except Exception:
                log.exception("frame processing failed")
                failures += 1
                if failures == self.STALL_FAILURES:
                    self._stall()
            frames += 1
            now = time.monotonic()
            if now - last >= 1.0:
                self.fps = frames / (now - last)
                frames, last = 0, now

    def with_engine(self) -> threading.Lock:
        """Lock to hold while changing the engine from another thread."""
        return self._lock

    def stop(self) -> None:
        self._stop_evt.set()
