from __future__ import annotations

import torch

from bsccm_i2i.config.schema import ModelConfig
from bsccm_i2i.models.registry import build_model


def test_unet_cnn_shape_and_training_loss_finite() -> None:
    torch.manual_seed(0)
    model_cfg = ModelConfig(
        name="unet_cnn",
        in_channels=23,
        out_channels=6,
        base_channels=32,
        lr=1e-3,
        weight_decay=0.0,
    )
    module = build_model(model_cfg)

    x = torch.randn(2, 23, 128, 128)
    y = torch.randn(2, 6, 128, 128)

    y_hat = module(x)
    assert y_hat.shape == (2, 6, 128, 128)

    loss = module.training_step((x, y), 0)
    assert bool(torch.isfinite(loss).item())
