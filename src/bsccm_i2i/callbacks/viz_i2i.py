"""TensorBoard visualization callback for image-to-image validation outputs."""

from __future__ import annotations

import pytorch_lightning as pl
import torch


class I2IVizCallback(pl.Callback):
    """Log target/pred/error channel grids from `pl_module._viz_cache` at validation epoch end."""

    def __init__(self, num_viz_samples: int = 4) -> None:
        super().__init__()
        self.num_viz_samples = max(1, int(num_viz_samples))

    @staticmethod
    def _build_grid(images: torch.Tensor, num_samples: int) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected rank-4 images [B, C, H, W], got ndim={images.ndim}")

        sample_count = min(num_samples, int(images.shape[0]))
        if sample_count <= 0:
            raise ValueError("Cannot build visualization grid from an empty batch.")

        clipped = images[:sample_count].detach().float().clamp(0.0, 1.0).cpu()
        sample_n, channel_n, height, width = clipped.shape
        return (
            clipped.permute(0, 2, 1, 3)
            .contiguous()
            .view(sample_n * height, channel_n * width)
            .unsqueeze(0)
        )

    @staticmethod
    def _resolve_tb_experiment(trainer: pl.Trainer) -> object | None:
        loggers = []
        if getattr(trainer, "loggers", None) is not None:
            loggers.extend(trainer.loggers)
        elif getattr(trainer, "logger", None) is not None:
            loggers.append(trainer.logger)

        for logger in loggers:
            experiment = getattr(logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "add_image"):
                return experiment
        return None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not bool(getattr(trainer, "is_global_zero", True)):
            return

        experiment = self._resolve_tb_experiment(trainer)
        if experiment is None:
            return

        cache = getattr(pl_module, "_viz_cache", None)
        if not isinstance(cache, dict):
            return
        if not {"x", "y", "y_hat"}.issubset(cache):
            return

        target = cache.get("y")
        prediction = cache.get("y_hat")
        if not isinstance(target, torch.Tensor) or not isinstance(prediction, torch.Tensor):
            return
        if target.shape != prediction.shape or target.ndim != 4:
            return
        if int(target.shape[0]) <= 0:
            return

        target_grid = self._build_grid(target, self.num_viz_samples)
        pred_grid = self._build_grid(prediction, self.num_viz_samples)
        error_grid = self._build_grid((target - prediction).abs(), self.num_viz_samples)

        global_step = int(getattr(trainer, "global_step", 0))
        experiment.add_image("viz/target_fluor_grid", target_grid, global_step=global_step)
        experiment.add_image("viz/pred_fluor_grid", pred_grid, global_step=global_step)
        experiment.add_image("viz/error_abs_grid", error_grid, global_step=global_step)
