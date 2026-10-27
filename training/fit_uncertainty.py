"""Fit the uncertainty head of a trained gaze model after the fact (REVIEW.md item 11, E12).

Training switches the NLL loss on only after a warm-up, and early stopping often keeps a
checkpoint from before it, so `var_head` was effectively untrained (Spearman rho about 0
in all 45 E1 runs). The angles do not depend on `var_head`, so it can be fitted on its
own: run the frozen network once, cache the 256-d features and the per-angle errors, fit
the linear head with the Gaussian NLL on the training subjects, then rescale it on the
validation subjects so that sigma is calibrated (mean squared z-score of 1 per angle).

    python -m training.fit_uncertainty D:/GazeOS-cache/outputs/m4/best.pt D:/GazeOS-cache/outputs/m4/best_unc.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from training.data.shards import GazeShardDataset
from training.export_onnx import load_checkpoint
from training.losses.gaze import angular_error_deg
from training.models.gazenet import GazeNetGOS, bin_centers_deg
from training.train import split_subjects


@torch.no_grad()
def collect(model: GazeNetGOS, paths: list[str], subjects: list[int], device: torch.device,
            batch: int = 256, workers: int = 4) -> dict[str, np.ndarray]:
    """Features, per-angle error (deg) and angular error (deg) of every sample of `subjects`."""
    model.eval().to(device)
    centers = bin_centers_deg(model.num_bins).to(device)
    dl = DataLoader(GazeShardDataset(paths, subjects=subjects), batch_size=batch, num_workers=workers)
    feats, errs, ang, subj = [], [], [], []
    for b in dl:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(b["face"].to(device), b["right_eye"].to(device), b["left_eye"].to(device), b["head"].to(device))
        pred = out.angles_deg(centers).float()
        target = b["gaze_deg"].to(device).float()
        feats.append(out.features.float().cpu())
        errs.append((pred - target).cpu())
        ang.append(angular_error_deg(pred, target).cpu())
        subj.append(torch.as_tensor(b["subject"]))
    return {"features": torch.cat(feats).numpy(), "err_deg": torch.cat(errs).numpy(),
            "angular_deg": torch.cat(ang).numpy(), "subject": torch.cat(subj).numpy()}


# Log-variance in radians^2, kept between sigma 0.5 deg and 57 deg. The linear head
# extrapolates: with the old -20 floor a few samples got sigma ~0.003 deg, their huge
# z-scores dominated the variance rescaling and pushed every sample to the ceiling
# (E1 fold 2: median sigma 56 deg, Spearman 0).
LOG_VAR_RANGE = (-9.5, 0.0)


def fit_head(features: np.ndarray, err_deg: np.ndarray, steps: int = 400, lr: float = 0.05,
             weight_decay: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Linear log-variance head (radians^2) by full-batch Gaussian NLL; returns (weight, bias)."""
    x = torch.as_tensor(features, dtype=torch.float32)
    e2 = torch.deg2rad(torch.as_tensor(err_deg, dtype=torch.float32)).pow(2)
    # Fit on standardized features, fold back after. The floor keeps near-constant features
    # (dead units) from becoming huge on new samples: unfloored, one fold's head saturated
    # at the clamp, where the gradient is zero (median sigma 56 deg on its test subject).
    mu, sd = x.mean(0), x.std(0)
    sd = sd.clamp_min(0.05 * float(sd.median()) + 1e-6)
    xs = (x - mu) / sd
    w = torch.zeros(x.shape[1], 2, requires_grad=True)
    b = torch.log(e2.mean(0)).clone().requires_grad_(True)  # start at the constant-variance fit
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        lv = (xs @ w + b).clamp(*LOG_VAR_RANGE)
        loss = (e2 / (2 * lv.exp()) + 0.5 * lv).mean() + weight_decay * w.pow(2).sum()
        loss.backward()
        opt.step()
    w_raw = (w / sd[:, None]).detach()
    b_raw = (b - (mu[:, None] * w_raw).sum(0)).detach()
    return w_raw.T.numpy(), b_raw.numpy()  # nn.Linear layout: weight (2, 256), bias (2,)


def log_var(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.clip(features @ weight.T + bias, *LOG_VAR_RANGE)


def calibration_shift(lv: np.ndarray, err_deg: np.ndarray) -> np.ndarray:
    """Per-angle log-variance shift that makes the mean squared z-score 1 (variance scaling)."""
    z2 = np.deg2rad(err_deg) ** 2 / np.exp(lv)
    return np.log(z2.mean(0))


def spearman(lv: np.ndarray, angular_deg: np.ndarray) -> float:
    from scipy.stats import spearmanr

    sigma = np.sqrt(np.exp(lv)).mean(1)
    return float(spearmanr(sigma, angular_deg).correlation)


def fit(checkpoint: str | Path, out: str | Path, shards: list[str] | None = None, workers: int = 4) -> dict[str, Any]:
    model, state = load_checkpoint(checkpoint)
    cfg = OmegaConf.create(state["config"])
    patterns = shards or list(cfg.data.shards)
    paths = sorted({p for pattern in patterns for p in glob.glob(str(pattern))})
    probe = GazeShardDataset(paths)
    excluded = set(cfg.data.get("exclude_subjects") or [])
    train_s, val_s, test_s = split_subjects([s for s in probe.subjects() if s not in excluded], cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    before = state["model"]["var_head.weight"].numpy(), state["model"]["var_head.bias"].numpy()
    data = {name: collect(model, paths, subs, device, workers=workers)
            for name, subs in (("train", train_s), ("val", val_s), ("test", test_s)) if subs}
    w, b = fit_head(data["train"]["features"], data["train"]["err_deg"])
    calib = "val" if "val" in data else "train"
    b = b + calibration_shift(log_var(data[calib]["features"], w, b), data[calib]["err_deg"])
    report: dict[str, Any] = {"checkpoint": str(checkpoint), "split": {"train": train_s, "val": val_s, "test": test_s},
                              "calibrated_on": calib, "log_var_range": list(LOG_VAR_RANGE)}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, d in data.items():
        lv_old, lv_new = log_var(d["features"], *before), log_var(d["features"], w, b)
        z2 = np.deg2rad(d["err_deg"]) ** 2 / np.exp(lv_new)
        if name == "test":  # per sample, for the E12 reliability diagram
            np.savez(out.with_suffix(".test.npz"), log_var=lv_new, err_deg=d["err_deg"], angular_deg=d["angular_deg"])
        report[name] = {
            "n": int(len(d["angular_deg"])),
            "mean_error_deg": float(d["angular_deg"].mean()),
            "spearman_before": spearman(lv_old, d["angular_deg"]),
            "spearman_after": spearman(lv_new, d["angular_deg"]),
            "median_sigma_deg_after": float(np.median(np.degrees(np.sqrt(np.exp(lv_new))).mean(1))),
            "mean_z2_after": [float(v) for v in z2.mean(0)],
        }
    new_state = dict(state)
    new_state["model"] = dict(state["model"])
    new_state["model"]["var_head.weight"] = torch.as_tensor(w, dtype=state["model"]["var_head.weight"].dtype)
    new_state["model"]["var_head.bias"] = torch.as_tensor(b, dtype=state["model"]["var_head.bias"].dtype)
    new_state["uncertainty_fit"] = {k: v for k, v in report.items() if k != "checkpoint"}
    torch.save(new_state, out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--shards", nargs="+", default=None, help="shard globs (default: from the checkpoint config)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    r = fit(a.checkpoint, a.out, a.shards, a.workers)
    for name in ("train", "val", "test"):
        if name in r:
            v = r[name]
            print(f"{name}: n {v['n']}, error {v['mean_error_deg']:.2f} deg, Spearman {v['spearman_before']:+.2f} -> "
                  f"{v['spearman_after']:+.2f}, median sigma {v['median_sigma_deg_after']:.2f} deg, "
                  f"z^2 {[round(x, 2) for x in v['mean_z2_after']]}")
    if not math.isfinite(r.get("val", r["train"])["spearman_after"]):
        raise SystemExit("fit failed")


if __name__ == "__main__":
    main()
