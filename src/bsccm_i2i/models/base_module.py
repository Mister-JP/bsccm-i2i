"""Shared Lightning base module helpers for distributed-safe metric logging."""

from __future__ import annotations

from collections.abc import Mapping

import pytorch_lightning as pl
import torch


class BaseI2IModule(pl.LightningModule):
    """Project-level Lightning base class with DDP-safe logging helpers."""

    def log_epoch_metric(
        self,
        name: str,
        value: torch.Tensor | float | int,
        *,
        batch_size: int,
        on_step: bool = False,
        prog_bar: bool = False,
    ) -> None:
        """Log one metric with consistent epoch semantics across devices."""
        self.log(
            name,
            value,
            prog_bar=prog_bar,
            on_step=on_step,
            on_epoch=True,
            batch_size=int(batch_size),
            sync_dist=True,
        )

    def log_epoch_metrics(
        self,
        metrics: Mapping[str, torch.Tensor | float | int],
        *,
        batch_size: int,
        prefix: str = "",
        on_step: bool = False,
        prog_bar: bool = False,
    ) -> None:
        """Log multiple metrics with a shared optional name prefix."""
        normalized_prefix = prefix.strip("/")
        for key, value in metrics.items():
            name = f"{normalized_prefix}/{key}" if normalized_prefix else str(key)
            self.log_epoch_metric(
                name,
                value,
                batch_size=batch_size,
                on_step=on_step,
                prog_bar=prog_bar,
            )
