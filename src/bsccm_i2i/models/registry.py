"""Model registry for runner-facing model construction."""

from __future__ import annotations

import pytorch_lightning as pl

from bsccm_i2i.config.schema import ModelConfig
from bsccm_i2i.models.unet_cnn import UNetCNNModule

ALLOWED_MODEL_NAMES = ("unet_cnn",)


def build_model(model_cfg: ModelConfig) -> pl.LightningModule:
    """Construct the configured model module from validated model config."""
    if model_cfg.name == "unet_cnn":
        return UNetCNNModule(
            in_channels=model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            base_channels=model_cfg.base_channels,
            lr=model_cfg.lr,
            weight_decay=model_cfg.weight_decay,
        )

    allowed_names = ", ".join(ALLOWED_MODEL_NAMES)
    raise ValueError(
        f"Unsupported model name {model_cfg.name!r}. Allowed model names: {allowed_names}."
    )


def load_model_from_checkpoint(model_cfg: ModelConfig, checkpoint_path: str) -> pl.LightningModule:
    """Load a model module from checkpoint using module-stored hyperparameters."""
    if model_cfg.name == "unet_cnn":
        return UNetCNNModule.load_from_checkpoint(checkpoint_path)

    allowed_names = ", ".join(ALLOWED_MODEL_NAMES)
    raise ValueError(
        f"Unsupported model name {model_cfg.name!r}. Allowed model names: {allowed_names}."
    )
