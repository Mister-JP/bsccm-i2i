"""Metric collections for fluorescence prediction."""

from __future__ import annotations

import torch

from bsccm_i2i.metrics.image_metrics import psnr, ssim


class Fluor6Metrics:
    """Compute per-channel and aggregate metrics for `[B, 6, H, W]` predictions."""

    NUM_CHANNELS = 6

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        assert pred.shape == target.shape, (
            f"pred and target shapes must match, got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
        assert pred.ndim == 4, f"pred and target must be rank-4 [B, C, H, W], got ndim={pred.ndim}"
        assert pred.shape[1] == self.NUM_CHANNELS, (
            f"expected {self.NUM_CHANNELS} channels, got {pred.shape[1]}"
        )

        l1_per_channel = (pred - target).abs().mean(dim=(0, 2, 3))
        l2_per_channel = (pred - target).pow(2).mean(dim=(0, 2, 3))
        psnr_per_channel = psnr(pred, target, data_range=1.0)
        ssim_per_channel = ssim(pred, target, data_range=1.0)

        metrics: dict[str, torch.Tensor] = {}
        for channel_idx in range(self.NUM_CHANNELS):
            metrics[f"l1_ch{channel_idx}"] = l1_per_channel[channel_idx]
            metrics[f"l2_ch{channel_idx}"] = l2_per_channel[channel_idx]
            metrics[f"psnr_ch{channel_idx}"] = psnr_per_channel[channel_idx]
            metrics[f"ssim_ch{channel_idx}"] = ssim_per_channel[channel_idx]

        metrics["l1_mean"] = l1_per_channel.mean()
        metrics["l2_mean"] = l2_per_channel.mean()
        metrics["psnr_mean"] = psnr_per_channel.mean()
        metrics["ssim_mean"] = ssim_per_channel.mean()
        return metrics
