"""Train and export the gesture classifiers B1 / B2 (PRD 13, experiment E10).

Data: labelled sessions from tools/label_gestures.py, each an .npz with
`features` (N, 17) per frame at 30 FPS and `labels` (N,) class index per frame.
Windows of 30 frames are labelled by their last frame's class.

    python -m training.train_gesture "data/gestures/*.npz" models/gesture_tcn.onnx --model tcn
    python -m training.train_gesture --synthetic models/gesture_tcn.onnx       # pipeline check only
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gazeos.gestures.events import GESTURE_CLASSES
from training.models.gesture import GRUClassifier, TemporalCNN, count_parameters

log = logging.getLogger(__name__)
WINDOW = 30


def windows(features: np.ndarray, labels: np.ndarray, stride: int = 2) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for end in range(WINDOW, len(features) + 1, stride):
        xs.append(features[end - WINDOW:end])
        ys.append(labels[end - 1])
    return np.asarray(xs, np.float32), np.asarray(ys, np.int64)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Time warp 0.8-1.2x, feature noise, random feature dropout (PRD 13.3)."""
    out = np.empty_like(x)
    for i, w in enumerate(x):
        speed = rng.uniform(0.8, 1.2)
        src = np.clip(np.arange(WINDOW) * speed + (WINDOW - WINDOW * speed), 0, WINDOW - 1)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, WINDOW - 1)
        frac = (src - lo)[:, None]
        w2 = w[lo] * (1 - frac) + w[hi] * frac
        w2 = w2 + rng.normal(0, 0.02, w2.shape)
        drop = rng.random(w2.shape[1]) < 0.05
        w2[:, drop] = 0.0
        out[i] = w2
    return out


def synthetic_sessions(n: int = 6, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Feature sequences with stylised gestures, only for checking the training path."""
    rng = np.random.default_rng(seed)
    sessions = []
    for _ in range(n):
        T = 3000
        f = np.zeros((T, 17), np.float32)
        f[:, 0:2] = 0.3 + rng.normal(0, 0.01, (T, 2))
        lab = np.zeros(T, np.int64)
        t = 30
        while t < T - 60:
            c = int(rng.integers(0, len(GESTURE_CLASSES)))
            dur = {1: 6, 2: 18, 3: 14, 4: 14}.get(c, 15)
            seg = slice(t, t + dur)
            if c in (1, 2):
                f[seg, 0:2] = 0.05
                f[seg, 2:4] = 0.9
            elif c == 3:
                f[seg, 0] = 0.05
                f[seg, 2] = 0.9
            elif c == 4:
                f[seg, 1] = 0.05
                f[seg, 3] = 0.9
            elif c == 5:
                f[seg, 6:9] = 0.8
            elif c == 6:
                f[seg, 9] = 0.8
            elif c == 7:
                f[seg, 10:12] = 0.8
            elif c == 8:
                f[seg, 9] = rng.uniform(0.1, 0.6, dur)
            lab[seg] = c
            t += dur + int(rng.integers(20, 60))
        sessions.append((f, lab))
    return sessions


def load_sessions(pattern: str) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for p in sorted(glob.glob(pattern)):
        d = np.load(p)
        out.append((d["features"].astype(np.float32), d["labels"].astype(np.int64)))
    return out


def train(sessions: list[tuple[np.ndarray, np.ndarray]], model_name: str = "tcn", epochs: int = 30,
          seed: int = 0) -> tuple[nn.Module, dict[str, object], np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_val = max(1, len(sessions) // 5)
    train_s, val_s = sessions[n_val:], sessions[:n_val]  # subject-independent split by session
    Xtr, ytr = map(np.concatenate, zip(*(windows(f, l) for f, l in train_s), strict=True))
    Xva, yva = map(np.concatenate, zip(*(windows(f, l) for f, l in val_s), strict=True))
    mean, std = Xtr.reshape(-1, Xtr.shape[2]).mean(0), Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-6
    model: nn.Module = TemporalCNN() if model_name == "tcn" else GRUClassifier()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    counts = np.bincount(ytr, minlength=len(GESTURE_CLASSES)).astype(np.float32)
    weights = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights / weights.mean())
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            xb = torch.tensor((augment(Xtr[idx], rng) - mean) / std)
            loss = loss_fn(model(xb), torch.tensor(ytr[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor((Xva - mean) / std)).argmax(1).numpy()
    report = {"val_window_accuracy": float((pred == yva).mean()), "params": count_parameters(model),
              "classes": list(GESTURE_CLASSES)}
    return model, report, mean, std


def export(model: nn.Module, out: str | Path, mean: np.ndarray, std: np.ndarray) -> Path:
    import onnx

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model.eval(), torch.zeros(1, WINDOW, 17), str(out), input_names=["window"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    proto = onnx.load(str(out))
    for k, v in {"classes": ",".join(GESTURE_CLASSES), "feature_mean": ",".join(f"{x:.6g}" for x in mean),
                 "feature_std": ",".join(f"{x:.6g}" for x in std)}.items():
        e = proto.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(proto, str(out))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=None)
    ap.add_argument("out")
    ap.add_argument("--model", choices=["tcn", "gru"], default="tcn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    sessions = synthetic_sessions() if a.synthetic else load_sessions(a.data)
    model, report, mean, std = train(sessions, a.model, a.epochs)
    export(model, a.out, mean, std)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
