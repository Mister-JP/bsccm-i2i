from __future__ import annotations

import torch

from bsccm_i2i.metrics.collections import Fluor6Metrics


def test_fluor6_metrics_cpu_outputs_expected_finite_scalars() -> None:
    torch.manual_seed(0)
    pred = torch.rand(2, 6, 32, 32)
    target = torch.rand(2, 6, 32, 32)

    metrics = Fluor6Metrics().compute(pred, target)

    expected_keys = {"l1_mean", "l2_mean", "psnr_mean", "ssim_mean"}
    for metric_name in ("l1", "l2", "psnr", "ssim"):
        for channel_idx in range(6):
            expected_keys.add(f"{metric_name}_ch{channel_idx}")

    assert expected_keys.issubset(metrics.keys())
    assert "l1_mean" in metrics
    assert "ssim_mean" in metrics
    assert "psnr_mean" in metrics
    assert "l1_ch0" in metrics
    assert "ssim_ch5" in metrics

    for key, value in metrics.items():
        assert isinstance(value, torch.Tensor), f"{key} must be a tensor"
        assert bool(torch.isfinite(value).all().item()), f"{key} must be finite"
        if "_ch" in key:
            assert value.ndim == 0, f"{key} must be a scalar tensor"
