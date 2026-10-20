"""Stage C: distil GazeNet-GOS (M3, teacher) into GazeNet-GOS-Lite (M4, student) (PRD 10.7).

    python -m training.distill teacher=outputs/m3/best.pt   # plus any train.yaml override

Loss = 0.5 x task loss + 0.5 x KL(teacher bins || student bins) x T^2, T = 4.
Goal: M4 within 0.3 degrees of M3 while at least 3x faster on CPU.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch

from training.export_onnx import load_checkpoint
from training.losses.gaze import distillation_loss, total_loss
from training.models.gazenet import GazeNetGOS, bin_centers_deg
from training.train import (
    _plain,
    evaluate,
    git_commit,
    lr_lambda,
    make_loaders,
    param_groups,
    set_seed,
)

log = logging.getLogger(__name__)


def distill(cfg: Any) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device != "cpu" else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher, _ = load_checkpoint(cfg.teacher)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = GazeNetGOS("mobilenet_v3_small", cfg.model.pretrained, num_bins=teacher.num_bins).to(device)
    train_dl, val_dl, test_dl, split = make_loaders(cfg)
    centers = bin_centers_deg(teacher.num_bins).to(device)
    t = cfg.train
    opt = torch.optim.AdamW(param_groups(student, t.lr_backbone, t.lr_new, t.weight_decay))
    spe = max(len(train_dl), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(spe * t.warmup_epochs, spe * t.epochs))
    best, best_epoch = math.inf, -1
    T = float(cfg.get("temperature", 4.0))
    for epoch in range(t.epochs):
        student.train()
        for b in train_dl:
            inputs = [b[k].to(device) for k in ("face", "right_eye", "left_eye", "head")]
            with torch.no_grad():
                t_out = teacher(*inputs)
            s_out = student(*inputs)
            task, _ = total_loss(s_out, b["gaze_deg"].to(device), centers, epoch, t.beta, t.nll_weight, t.nll_warmup_epochs)
            loss = distillation_loss(s_out, t_out, task, T, 0.5)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        score = evaluate(student, val_dl, device, centers)["mean_error_deg"] if val_dl is not None else float(loss.detach())
        log.info("epoch %d val %.3f", epoch, score)
        if score < best:
            best, best_epoch = score, epoch
            torch.save({"model": student.state_dict(), "config": {**_plain(cfg), "model": {
                "backbone": "mobilenet_v3_small", "use_eyes": True, "use_head": True, "use_face": True,
                "fusion": "transformer", "num_bins": teacher.num_bins, "dropout": 0.3}},
                "epoch": epoch, "commit": git_commit()}, out_dir / "best.pt")
        elif epoch - best_epoch >= t.patience:
            break
    result: dict[str, Any] = {"best_val": best, "best_epoch": best_epoch, "split": split}
    if test_dl is not None:
        student.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
        result["test_student"] = evaluate(student, test_dl, device, centers)
        result["test_teacher"] = evaluate(teacher, test_dl, device, centers)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    import hydra

    @hydra.main(config_path="../configs", config_name="distill", version_base="1.3")
    def _run(cfg: Any) -> None:
        distill(cfg)

    _run()


if __name__ == "__main__":
    main()
