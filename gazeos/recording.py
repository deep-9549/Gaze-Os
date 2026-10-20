"""Session recording and replay (PRD 23, FR-38).

A recording stores landmarks, blendshapes, the MediaPipe transform and the
on-screen target for every frame, never full camera frames. With research
recording on (opt-in, PRD 21.1) it also stores the normalized face and eye
crops, which lets the CNN path be replayed too.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gazeos.vision.landmarks import LandmarkResult
from gazeos.vision.normalization import NormalizationResult


@dataclass
class RecordedFrame:
    t: float
    landmarks: LandmarkResult
    target_px: tuple[float, float] | None = None
    face: NDArray[np.uint8] | None = None
    right_eye: NDArray[np.uint8] | None = None
    left_eye: NDArray[np.uint8] | None = None


@dataclass
class SessionMeta:
    screen: dict[str, float]
    camera: dict[str, float]
    participant: str = "anonymous"
    condition: dict[str, str] = field(default_factory=dict)


class SessionRecorder:
    def __init__(self, path: str | Path, meta: SessionMeta, keep_crops: bool = False) -> None:
        self.path = Path(path)
        self.meta = meta
        self.keep_crops = keep_crops
        self._t: list[float] = []
        self._lm: list[NDArray[np.float64]] = []
        self._valid: list[bool] = []
        self._bs: list[dict[str, float]] = []
        self._tf: list[NDArray[np.float64]] = []
        self._targets: list[tuple[float, float]] = []
        self._faces: list[NDArray[np.uint8]] = []
        self._reyes: list[NDArray[np.uint8]] = []
        self._leyes: list[NDArray[np.uint8]] = []

    def add(
        self,
        t: float,
        lm: LandmarkResult,
        target_px: tuple[float, float] | None = None,
        norm: NormalizationResult | None = None,
    ) -> None:
        self._t.append(t)
        ok = lm.landmarks_px is not None
        self._valid.append(ok)
        self._lm.append(lm.landmarks_px if ok else np.zeros((478, 3)))  # type: ignore[arg-type]
        self._bs.append(dict(lm.blendshapes))
        self._tf.append(lm.transform if lm.transform is not None else np.eye(4))
        self._targets.append(target_px if target_px is not None else (np.nan, np.nan))
        if self.keep_crops:
            if norm is not None:
                self._faces.append(norm.face)
                self._reyes.append(norm.right_eye)
                self._leyes.append(norm.left_eye)
            else:
                self._faces.append(np.zeros((224, 224, 3), np.uint8))
                self._reyes.append(np.zeros((36, 60), np.uint8))
                self._leyes.append(np.zeros((36, 60), np.uint8))

    def __len__(self) -> int:
        return len(self._t)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        names = sorted({k for b in self._bs for k in b})
        arrays: dict[str, NDArray[np.generic]] = {
            "t": np.asarray(self._t),
            "landmarks": np.asarray(self._lm, dtype=np.float32),
            "valid": np.asarray(self._valid),
            "blendshapes": np.asarray([[b.get(n, 0.0) for n in names] for b in self._bs], dtype=np.float32),
            "transform": np.asarray(self._tf, dtype=np.float32),
            "targets": np.asarray(self._targets, dtype=np.float64),
        }
        if self.keep_crops and self._faces:
            arrays["faces"] = np.asarray(self._faces)
            arrays["right_eyes"] = np.asarray(self._reyes)
            arrays["left_eyes"] = np.asarray(self._leyes)
        meta = {"blendshape_names": names, "screen": self.meta.screen, "camera": self.meta.camera,
                "participant": self.meta.participant, "condition": self.meta.condition}
        np.savez_compressed(self.path, meta=np.asarray(json.dumps(meta)), **arrays)  # type: ignore[arg-type]
        return self.path


class RecordedSession:
    def __init__(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(data["meta"]))
        self.t = data["t"]
        self.landmarks = data["landmarks"].astype(np.float64)
        self.valid = data["valid"]
        self.blendshapes = data["blendshapes"]
        self.transform = data["transform"].astype(np.float64)
        self.targets = data["targets"]
        self.faces = data["faces"] if "faces" in data.files else None
        self.right_eyes = data["right_eyes"] if "right_eyes" in data.files else None
        self.left_eyes = data["left_eyes"] if "left_eyes" in data.files else None
        self.names: list[str] = self.meta["blendshape_names"]

    def __len__(self) -> int:
        return len(self.t)

    def frames(self) -> Iterator[RecordedFrame]:
        for i in range(len(self.t)):
            if self.valid[i]:
                lm = LandmarkResult(
                    num_faces=1,
                    landmarks_px=self.landmarks[i],
                    blendshapes={n: float(v) for n, v in zip(self.names, self.blendshapes[i], strict=True)},
                    transform=self.transform[i],
                )
            else:
                lm = LandmarkResult(num_faces=0)
            target = None if np.isnan(self.targets[i, 0]) else (float(self.targets[i, 0]), float(self.targets[i, 1]))
            yield RecordedFrame(
                float(self.t[i]), lm, target,
                None if self.faces is None else self.faces[i],
                None if self.right_eyes is None else self.right_eyes[i],
                None if self.left_eyes is None else self.left_eyes[i],
            )
