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
from gazeos.gestures.blink import BlinkDetector, BlinkEvent, eye_state
from gazeos.gestures.events import FeatureWindow, GestureEventDetector, scale_by_gesture_size
from gazeos.inference.gaze import GazeEstimate, GeometricGazeEstimator
from gazeos.settings import Settings
from gazeos.types import FrameResult
from gazeos.vision import features as feat
from gazeos.vision.enhance import mean_brightness
from gazeos.vision.geometry import head_direction_pitchyaw, rotation_to_euler
from gazeos.vision.head_pose import CameraIntrinsics, HeadPose, HeadPoseEstimator
from gazeos.vision.landmarks import LandmarkResult
from gazeos.vision.normalization import NormalizationResult, normalize
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
        self, lm: LandmarkResult, t: float, image_rgb: NDArray[np.uint8] | None = None
    ) -> ProcessedFrame:
        """Everything after face landmarks; image is needed for the CNN and brightness."""
        timings: dict[str, float] = {}
        res = FrameResult(t_capture=t, num_faces=lm.num_faces)
        if image_rgb is not None:
            res.image_size = (image_rgb.shape[1], image_rgb.shape[0])
        if lm.num_faces == 0 or lm.landmarks_px is None:
            q = self.quality.check(t, 0)
            self.head.reset()
            self.blinks.reset()
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
        state = eye_state(lm.blendshapes, ear_l, ear_r, self.s.blink.ear_threshold,
                          self.s.quality.blink_score_threshold)
        blink_score = max(lm.blendshapes.get("eyeBlinkLeft", 0.0), lm.blendshapes.get("eyeBlinkRight", 0.0)) \
            if lm.blendshapes else (1.0 if state.both_closed else 0.0)
        blink_event = self.blinks.update(t, state)

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
            if self.gaze_model is not None and image_rgb is not None:
                norm = normalize(image_rgb, landmarks, pose, self.intrinsics)
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

    def run(self) -> None:
        last = time.monotonic()
        frames = 0
        while not self._stop_evt.is_set():
            frame = self.capture.latest(timeout=0.5)  # type: ignore[attr-defined]
            if frame is None:
                continue
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
                if self.on_frame is not None:
                    self.on_frame(pf, out, frame.image_rgb)
            except Exception:
                log.exception("frame processing failed")
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
