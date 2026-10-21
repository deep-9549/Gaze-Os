"""MediaPipe Face Landmarker wrapper (PRD 9.1, FR-04).

Runs in VIDEO mode: BlazeFace detection once, then tracking between frames.
Asks for two faces so FR-03 can warn when more than one face is visible; only
the first face drives the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class LandmarkResult:
    num_faces: int
    landmarks_px: FloatArray | None = None  # (478, 3): x, y in pixels, z in pixel-ish units
    blendshapes: dict[str, float] = field(default_factory=dict)
    transform: FloatArray | None = None  # (4, 4) facial transformation matrix


class FaceLandmarker:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m tools.download_models` first."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, image_rgb: NDArray[np.uint8], t_seconds: float) -> LandmarkResult:
        h, w = image_rgb.shape[:2]
        ts_ms = int(t_seconds * 1000)
        if ts_ms <= self._last_ts_ms:  # VIDEO mode needs strictly increasing timestamps
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        n = len(result.face_landmarks)
        if n == 0:
            return LandmarkResult(num_faces=0)
        lm = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]])
        shapes: dict[str, float] = {}
        if result.face_blendshapes:
            shapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        transform = None
        if result.facial_transformation_matrixes:
            transform = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
        return LandmarkResult(num_faces=n, landmarks_px=lm, blendshapes=shapes, transform=transform)

    def close(self) -> None:
        self._landmarker.close()
