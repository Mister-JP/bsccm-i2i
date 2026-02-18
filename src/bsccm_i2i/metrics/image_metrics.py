"""Per-channel PSNR and SSIM utilities."""

from __future__ import annotations

import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def _validate_inputs(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must have identical shapes, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.ndim != 4:
        raise ValueError(f"pred and target must be rank-4 [B, C, H, W], got ndim={pred.ndim}")


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute per-channel PSNR for `[B, C, H, W]` tensors and return shape `[C]`."""
    _validate_inputs(pred, target)
    pred_clamped = pred.clamp(0.0, 1.0)
    target_clamped = target.clamp(0.0, 1.0)

    per_channel: list[torch.Tensor] = []
    for channel_idx in range(pred.shape[1]):
        score = peak_signal_noise_ratio(
            pred_clamped[:, channel_idx : channel_idx + 1],
            target_clamped[:, channel_idx : channel_idx + 1],
            data_range=data_range,
        )
        per_channel.append(score)
    return torch.stack(per_channel, dim=0)


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute per-channel SSIM for `[B, C, H, W]` tensors and return shape `[C]`."""
    _validate_inputs(pred, target)
    pred_clamped = pred.clamp(0.0, 1.0)
    target_clamped = target.clamp(0.0, 1.0)

    per_channel: list[torch.Tensor] = []
    for channel_idx in range(pred.shape[1]):
        score = structural_similarity_index_measure(
            pred_clamped[:, channel_idx : channel_idx + 1],
            target_clamped[:, channel_idx : channel_idx + 1],
            data_range=data_range,
        )
        per_channel.append(score)
    return torch.stack(per_channel, dim=0)
