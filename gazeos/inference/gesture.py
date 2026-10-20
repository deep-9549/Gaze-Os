"""Gesture classifier inference (B1 temporal CNN / B2 GRU exported to ONNX; PRD 13.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gazeos.gestures.events import GESTURE_CLASSES
from gazeos.inference.gaze import softmax


class OnnxGestureClassifier:
    """Input `window` (1, 30, 17) float32 -> output `logits` (1, 9)."""

    def __init__(self, path: str | Path, threads: int = 1) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.classes = tuple(meta["classes"].split(",")) if "classes" in meta else GESTURE_CLASSES
        mean = meta.get("feature_mean")
        std = meta.get("feature_std")
        self.mean = np.asarray([float(v) for v in mean.split(",")], dtype=np.float32) if mean else None
        self.std = np.asarray([float(v) for v in std.split(",")], dtype=np.float32) if std else None

    def predict(self, window: NDArray[np.float32]) -> dict[str, float]:
        x = window.astype(np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        logits = self.session.run(None, {self.input_name: x})[0][0]
        probs = softmax(logits)
        return {c: float(p) for c, p in zip(self.classes, probs, strict=True)}


class RuleGestureClassifier:
    """B0-style thresholds on blendshapes, used when no trained model is available."""

    def __init__(self, brow: float = 0.55, jaw: float = 0.5, smile: float = 0.6) -> None:
        self.brow, self.jaw, self.smile = brow, jaw, smile

    def predict_frame(self, blendshapes: dict[str, float]) -> dict[str, float]:
        probs = dict.fromkeys(GESTURE_CLASSES, 0.0)
        brow = max(blendshapes.get("browInnerUp", 0.0),
                   (blendshapes.get("browOuterUpLeft", 0.0) + blendshapes.get("browOuterUpRight", 0.0)) / 2)
        jaw = blendshapes.get("jawOpen", 0.0)
        smile = (blendshapes.get("mouthSmileLeft", 0.0) + blendshapes.get("mouthSmileRight", 0.0)) / 2
        if jaw > self.jaw:
            probs["mouth_open"] = 0.9
        elif brow > self.brow:
            probs["eyebrow_raise"] = 0.9
        elif smile > self.smile:
            probs["smile"] = 0.9
        else:
            probs["none"] = 0.9
        rest = 0.1 / (len(probs) - 1)
        return {k: (v if v else rest) for k, v in probs.items()}
