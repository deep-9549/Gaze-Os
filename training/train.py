"""Train GazeNet-GOS models (PRD 10.5, 16.1).

    python -m training.train                                   # configs/train.yaml defaults
    python -m training.train experiment=stage_a_ethxgaze        # Stage A pre-training
    python -m training.train experiment=stage_b_mpii fold=3      # Stage B, LOSO fold 3
    python -m training.train experiment=e2_streams model.use_eyes=false seed=1

Every run is defined by a committed Hydra config; the resolved config and the
Git commit are saved with the checkpoint (PRD 16.3).
"""

from __future__ import annotations

import json
import logging
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.data.augment import GazeAugment
from training.data.shards import GazeShardDataset
from training.losses.gaze import angular_error_deg, total_loss
from training.models.gazenet import GazeNetGOS, bin_centers_deg

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def build_model(cfg: Any) -> GazeNetGOS:
    m = cfg.model
    return GazeNetGOS(
        backbone=m.backbone, pretrained=m.pretrained, use_eyes=m.use_eyes, use_head=m.use_head,
        use_face=m.get("use_face", True), fusion=m.fusion, num_bins=m.num_bins, dropout=m.dropout,
    )


def param_groups(model: GazeNetGOS, lr_backbone: float, lr_new: float, weight_decay: float) -> list[dict[str, Any]]:
    backbone = list(model.face.parameters()) if model.use_face else []
    ids = {id(p) for p in backbone}
    new = [p for p in model.parameters() if id(p) not in ids]
    groups = [{"params": new, "lr": lr_new, "weight_decay": weight_decay}]
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay})
    return groups


def lr_lambda(warmup_steps: int, total_steps: int):  # type: ignore[no-untyped-def]
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return f


def split_subjects(all_subjects: list[int], cfg: Any) -> tuple[list[int], list[int], list[int]]:
    """Leave-one-subject-out fold, or held-out validation subjects (PRD 15.3)."""
    subjects = sorted(all_subjects)
    test: list[int] = []
    if cfg.data.get("fold") is not None and cfg.data.fold >= 0:
        test = [subjects[cfg.data.fold % len(subjects)]]
    rest = [s for s in subjects if s not in test]
    rng = random.Random(cfg.seed)
    n_val = max(1, int(round(len(rest) * cfg.data.val_fraction))) if len(rest) > 1 else 0
    val = sorted(rng.sample(rest, n_val)) if n_val else []
    train = [s for s in rest if s not in val]
    return train, val, test


def make_loaders(cfg: Any) -> tuple[DataLoader, DataLoader | None, DataLoader | None, dict[str, list[int]]]:
    paths = sorted(str(p) for pattern in cfg.data.shards for p in Path().glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no shards match {list(cfg.data.shards)}")
    probe = GazeShardDataset(paths)
    train_s, val_s, test_s = split_subjects(probe.subjects(), cfg)
    aug = GazeAugment(seed=cfg.seed) if cfg.data.augment else None
    train_ds = GazeShardDataset(paths, subjects=train_s, transform=aug, stride=cfg.data.stride)
    kw = {"batch_size": cfg.train.batch_size, "num_workers": cfg.data.workers, "pin_memory": torch.cuda.is_available()}
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > cfg.train.batch_size, **kw)
    val_dl = DataLoader(GazeShardDataset(paths, subjects=val_s), **kw) if val_s else None
    test_dl = DataLoader(GazeShardDataset(paths, subjects=test_s), **kw) if test_s else None
    return train_dl, val_dl, test_dl, {"train": train_s, "val": val_s, "test": test_s}


@torch.no_grad()
def evaluate(model: GazeNetGOS, loader: DataLoader, device: torch.device, centers: torch.Tensor) -> dict[str, Any]:
    model.eval()
    errs, sigmas, subjects = [], [], []
    for b in loader:
        out = model(b["face"].to(device), b["right_eye"].to(device), b["left_eye"].to(device), b["head"].to(device))
        pred = out.angles_deg(centers)
        errs.append(angular_error_deg(pred, b["gaze_deg"].to(device)).cpu())
        sigmas.append(torch.rad2deg(out.log_var.exp().sqrt().mean(-1)).cpu())
        subjects.append(torch.as_tensor(b["subject"]))
    e = torch.cat(errs).numpy()
    s = torch.cat(sigmas).numpy()
    subj = torch.cat(subjects).numpy()
    per_subject = {int(k): float(e[subj == k].mean()) for k in np.unique(subj)}
    from scipy.stats import spearmanr

    rho = float(spearmanr(s, e).correlation) if len(e) > 2 else float("nan")
    return {"mean_error_deg": float(e.mean()), "median_error_deg": float(np.median(e)),
            "per_subject": per_subject, "uncertainty_spearman": rho, "n": int(len(e))}


def train(cfg: Any) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device != "cpu" else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dl, val_dl, test_dl, split = make_loaders(cfg)
    model = build_model(cfg).to(device)
    if cfg.get("init_checkpoint"):
        state = torch.load(cfg.init_checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        log.info("initialized from %s (missing %d, unexpected %d)", cfg.init_checkpoint, len(missing), len(unexpected))
    centers = bin_centers_deg(cfg.model.num_bins).to(device)
    t = cfg.train
    opt = torch.optim.AdamW(param_groups(model, t.lr_backbone, t.lr_new, t.weight_decay))
    steps_per_epoch = max(len(train_dl), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps_per_epoch * t.warmup_epochs, steps_per_epoch * t.epochs))
    use_amp = device.type == "cuda" and t.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = _writer(cfg, out_dir)
    best, best_epoch, history = math.inf, -1, []
    for epoch in range(t.epochs):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for b in train_dl:
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(b["face"].to(device), b["right_eye"].to(device), b["left_eye"].to(device), b["head"].to(device))
                loss, _ = total_loss(out, b["gaze_deg"].to(device), centers, epoch, t.beta, t.nll_weight,
                                     t.nll_warmup_epochs, t.loss_mode)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss.detach()) * len(b["subject"])
            n += len(b["subject"])
        metrics = {"epoch": epoch, "train_loss": running / max(n, 1), "time_s": time.time() - t0}
        if val_dl is not None:
            metrics["val"] = evaluate(model, val_dl, device, centers)
            score = metrics["val"]["mean_error_deg"]
        else:
            score = metrics["train_loss"]
        history.append(metrics)
        log.info("epoch %d: %s", epoch, {k: v if not isinstance(v, dict) else v.get("mean_error_deg") for k, v in metrics.items()})
        if writer is not None:
            writer.add_scalar("train/loss", metrics["train_loss"], epoch)
            if "val" in metrics:
                writer.add_scalar("val/mean_error_deg", score, epoch)
        if score < best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "config": _plain(cfg), "epoch": epoch, "commit": git_commit()},
                       out_dir / "best.pt")
        elif epoch - best_epoch >= t.patience:  # early stopping on validation error
            log.info("early stop at epoch %d (best %d)", epoch, best_epoch)
            break
    state = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    result: dict[str, Any] = {"best_epoch": best_epoch, "best_val": best, "split": split, "history": history,
                              "commit": git_commit(), "config": _plain(cfg)}
    if test_dl is not None:
        result["test"] = evaluate(model, test_dl, device, centers)
        log.info("test mean angular error: %.2f deg", result["test"]["mean_error_deg"])
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    if writer is not None:
        writer.close()
    return result


def _plain(cfg: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return dict(cfg)


def _writer(cfg: Any, out_dir: Path):  # type: ignore[no-untyped-def]
    if cfg.logging.backend == "tensorboard":
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(out_dir / "tb")
        except ImportError:
            return None
    if cfg.logging.backend == "wandb":
        import wandb  # type: ignore[import-not-found]

        wandb.init(project=cfg.logging.project, config=_plain(cfg), dir=str(out_dir))
    return None


def main() -> None:
    import hydra

    @hydra.main(config_path="../configs", config_name="train", version_base="1.3")
    def _run(cfg: Any) -> None:
        train(cfg)

    _run()


if __name__ == "__main__":
    main()
