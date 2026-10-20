"""Stage D (optional): meta-learned personalization P5 (PRD 11.4, inspired by FAZE).

First-order MAML over subjects. Each episode samples one subject, adapts an
additive linear correction on the frozen network's 256-d features from k
"calibration" samples, and evaluates on held-out samples of the same subject.
The meta-learned initialization (W, b, feature normalization) is exported as
an .npz that `MetaLearnedAdaptation.from_file` loads at runtime, where the
same adaptation runs in numpy from 9 calibration points.

    python -m training.meta_train outputs/m4/best.pt "data/shards/gazeos_webcam/*.h5" models/meta_init.npz
"""

from __future__ import annotations

import argparse
import glob
import logging
import random

import numpy as np
import torch

from training.data.shards import GazeShardDataset
from training.export_onnx import load_checkpoint

log = logging.getLogger(__name__)


@torch.no_grad()
def extract(model, ds: GazeShardDataset, batch: int = 64) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:  # type: ignore[no-untyped-def]
    """Per subject: (features, predicted angles rad, true angles rad)."""
    from training.models.gazenet import bin_centers_deg

    centers = bin_centers_deg(model.num_bins)
    feats, preds, trues, subs = [], [], [], []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch)
    for b in loader:
        out = model(b["face"], b["right_eye"], b["left_eye"], b["head"])
        feats.append(out.features.numpy())
        preds.append(np.radians(out.angles_deg(centers).numpy()))
        trues.append(np.radians(b["gaze_deg"].numpy()))
        subs.append(np.asarray(b["subject"]))
    F, P, Y, S = map(np.concatenate, (feats, preds, trues, subs))
    return {int(s): (F[S == s], P[S == s], Y[S == s]) for s in np.unique(S)}


def meta_train(
    per_subject: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    k_shot: int = 9,
    inner_steps: int = 20,
    inner_lr: float = 0.05,
    meta_lr: float = 0.01,
    episodes: int = 2000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = random.Random(seed)
    allF = np.concatenate([v[0] for v in per_subject.values()])
    mu, sd = allF.mean(0), allF.std(0) + 1e-6
    d = allF.shape[1]
    W = torch.zeros(d, 2, requires_grad=False)
    b = torch.zeros(2)
    subjects = [s for s, v in per_subject.items() if len(v[0]) > 2 * k_shot]
    if not subjects:
        raise ValueError("need subjects with more than 2k samples")
    for ep in range(episodes):
        s = rng.choice(subjects)
        F, P, Y = per_subject[s]
        idx = rng.sample(range(len(F)), 2 * k_shot)
        sup, qry = idx[:k_shot], idx[k_shot:]
        Fs = torch.tensor((F[sup] - mu) / sd, dtype=torch.float32)
        Ts = torch.tensor(Y[sup] - P[sup], dtype=torch.float32)
        Fq = torch.tensor((F[qry] - mu) / sd, dtype=torch.float32)
        Tq = torch.tensor(Y[qry] - P[qry], dtype=torch.float32)
        w, bb = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
        for _ in range(inner_steps):  # inner loop: plain SGD, as at runtime
            loss = ((Fs @ w + bb - Ts) ** 2).mean() + 1e-2 * (w**2).sum()
            gw, gb = torch.autograd.grad(loss, (w, bb))
            w = (w - inner_lr * gw).detach().requires_grad_(True)
            bb = (bb - inner_lr * gb).detach().requires_grad_(True)
        q_loss = ((Fq @ w + bb - Tq) ** 2).mean()
        gw, gb = torch.autograd.grad(q_loss, (w, bb))
        # First-order MAML: apply the query gradient at the adapted weights to the init.
        W -= meta_lr * gw
        b -= meta_lr * gb
        if ep % 200 == 0:
            log.info("episode %d query loss %.6f", ep, float(q_loss))
    return {"W": W.numpy(), "b": b.numpy(), "mu": mu, "sd": sd}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("shards")
    ap.add_argument("out")
    ap.add_argument("--episodes", type=int, default=2000)
    a = ap.parse_args()
    model, _ = load_checkpoint(a.checkpoint)
    data = extract(model, GazeShardDataset(sorted(glob.glob(a.shards))))
    init = meta_train(data, episodes=a.episodes)
    np.savez(a.out, **init)
    log.info("saved meta-learned init to %s", a.out)


if __name__ == "__main__":
    main()
