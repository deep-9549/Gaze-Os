"""Small transformer language model (L2, PRD 14.1) exported to ONNX.

Word-level vocabulary for simple, fast next-word scoring on CPU: input
`tokens` (1, T) int64 -> output `logits` (1, T, V). The vocabulary is stored
next to the model as <model>.vocab.txt.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from gazeos.inference.gaze import softmax


class OnnxWordLM:
    def __init__(self, path: str | Path, context: int = 16) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        vocab_path = Path(str(path).removesuffix(".onnx") + ".vocab.txt")
        self.itos = vocab_path.read_text(encoding="utf-8").splitlines()
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.unk = self.stoi.get("<unk>", 0)
        self.bos = self.stoi.get("<s>", 0)
        self.context = context
        self._cache_key: tuple[str, ...] | None = None
        self._cache: np.ndarray | None = None

    def _dist(self, context: Sequence[str]) -> np.ndarray:
        key = tuple(context[-self.context :])
        if key != self._cache_key or self._cache is None:
            ids = [self.bos] + [self.stoi.get(w, self.unk) for w in key]
            logits = self.session.run(None, {"tokens": np.asarray([ids], dtype=np.int64)})[0][0, -1]
            self._cache_key, self._cache = key, softmax(logits)
        return self._cache

    def prob(self, w: str, context: Sequence[str]) -> float:
        return float(self._dist(context)[self.stoi.get(w, self.unk)])

    def next_words(self, context: Sequence[str], k: int = 5, prefix: str = "") -> list[tuple[str, float]]:
        p = self._dist(context)
        order = np.argsort(-p)
        out = []
        for i in order:
            w = self.itos[i]
            if w.startswith("<") or (prefix and not w.startswith(prefix)):
                continue
            out.append((w, float(p[i])))
            if len(out) >= k:
                break
        return out
