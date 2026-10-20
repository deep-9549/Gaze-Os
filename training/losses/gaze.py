"""Losses: L2CS combined loss, Gaussian NLL for the uncertainty head, distillation (PRD 10.4, 10.7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from training.models.gazenet import BIN_MIN_DEG, BIN_WIDTH_DEG, GazeOutput


def angle_to_bin(angle_deg: torch.Tensor, num_bins: int, width: float = BIN_WIDTH_DEG, lo: float = BIN_MIN_DEG) -> torch.Tensor:
    return ((angle_deg - lo) / width).floor().clamp(0, num_bins - 1).long()


def l2cs_loss(
    out: GazeOutput,
    target_deg: torch.Tensor,
    centers: torch.Tensor,
    beta: float = 1.0,
    width: float = BIN_WIDTH_DEG,
    mode: str = "combined",
) -> tuple[torch.Tensor, torch.Tensor]:
    """sum over pitch, yaw of CE(z, k) + beta * (theta_hat - theta)^2.

    `mode` supports the E3 ablation: "combined", "classification", "regression".
    Returns (loss, predicted angles in degrees). Angles are compared in degrees
    scaled by 1/bin width, so beta = 1 matches the L2CS formulation's units.
    """
    k = centers.numel()
    pred = out.angles_deg(centers)
    loss = torch.zeros((), device=target_deg.device)
    for i, logits in enumerate((out.pitch_logits, out.yaw_logits)):
        if mode in ("combined", "classification"):
            loss = loss + F.cross_entropy(logits, angle_to_bin(target_deg[:, i], k, width, float(centers[0]) - width / 2))
        if mode in ("combined", "regression"):
            loss = loss + beta * F.mse_loss(pred[:, i] / width, target_deg[:, i] / width)
    return loss, pred


def gaussian_nll(pred_deg: torch.Tensor, target_deg: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """sum over angles of (theta_hat - theta)^2 / 2 sigma^2 + 1/2 log sigma^2, with sigma in radians."""
    err = torch.deg2rad(pred_deg - target_deg)
    log_var = log_var.clamp(-20, 5)
    return (err.pow(2) / (2 * log_var.exp()) + 0.5 * log_var).sum(-1).mean()


def total_loss(
    out: GazeOutput,
    target_deg: torch.Tensor,
    centers: torch.Tensor,
    epoch: int,
    beta: float = 1.0,
    nll_weight: float = 0.1,
    nll_warmup_epochs: int = 5,
    mode: str = "combined",
) -> tuple[torch.Tensor, torch.Tensor]:
    loss, pred = l2cs_loss(out, target_deg, centers, beta, mode=mode)
    if epoch >= nll_warmup_epochs and nll_weight > 0:
        # Detach the prediction so the NLL trains the uncertainty head, not the angle.
        loss = loss + nll_weight * gaussian_nll(pred.detach(), target_deg, out.log_var)
    return loss, pred


def distillation_loss(
    student: GazeOutput, teacher: GazeOutput, task_loss: torch.Tensor, temperature: float = 4.0, alpha: float = 0.5
) -> torch.Tensor:
    """0.5 x task loss + 0.5 x KL(teacher || student) on bin distributions, scaled by T^2."""
    kl = torch.zeros((), device=task_loss.device)
    for s, t in ((student.pitch_logits, teacher.pitch_logits), (student.yaw_logits, teacher.yaw_logits)):
        kl = kl + F.kl_div(
            F.log_softmax(s / temperature, -1), F.softmax(t / temperature, -1), reduction="batchmean"
        ) * temperature**2
    return alpha * task_loss + (1 - alpha) * kl


def angular_error_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    """Per-sample 3D angle between gaze vectors given as pitch/yaw in degrees."""

    def vec(py: torch.Tensor) -> torch.Tensor:
        p, y = torch.deg2rad(py[:, 0]), torch.deg2rad(py[:, 1])
        return torch.stack([-p.cos() * y.sin(), -p.sin(), -p.cos() * y.cos()], -1)

    cos = (vec(pred_deg) * vec(target_deg)).sum(-1).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cos))
