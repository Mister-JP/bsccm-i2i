"""TensorBoard visualization callback for antibody-grouped validation outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
import torch


@dataclass
class _PanelGroup:
    tag_component: str
    x: torch.Tensor
    y: torch.Tensor


class I2IVizCallback(pl.Callback):
    """Log deterministic antibody-grouped target/pred/error grids at validation epoch end."""

    def __init__(
        self,
        *,
        viz_antibodies: list[str] | None = None,
        viz_samples_per_antibody: int = 2,
        image_log_every_n_steps: int = 200,
        viz_log_target_once: bool = True,
        viz_log_error: bool = True,
    ) -> None:
        super().__init__()
        self.viz_antibodies = list(viz_antibodies or [])
        self.viz_samples_per_antibody = max(1, int(viz_samples_per_antibody))
        self.image_log_every_n_steps = max(1, int(image_log_every_n_steps))
        self.viz_log_target_once = bool(viz_log_target_once)
        self.viz_log_error = bool(viz_log_error)
        self._next_log_step = 0
        self._target_logged_tags: set[str] = set()
        self._panel_groups: list[_PanelGroup] | None = None
        self._all_inputs_cpu: torch.Tensor | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "next_log_step": int(self._next_log_step),
            "target_logged_tags": sorted(self._target_logged_tags),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        next_step = int(state_dict.get("next_log_step", 0))
        self._next_log_step = max(0, next_step)
        restored = state_dict.get("target_logged_tags", [])
        if isinstance(restored, list):
            self._target_logged_tags = {str(value) for value in restored if str(value)}

    def _should_log_for_step(self, global_step: int) -> bool:
        if global_step < self._next_log_step:
            return False
        self._next_log_step = global_step + self.image_log_every_n_steps
        return True

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

    @staticmethod
    def _sanitize_tag_component(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
        normalized = normalized.strip("_")
        return (normalized or "unknown").lower()

    def _build_panel_groups(self, trainer: pl.Trainer) -> list[_PanelGroup]:
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return []

        build_panel = getattr(datamodule, "build_antibody_viz_panel", None)
        if not callable(build_panel):
            return []

        panel = build_panel(
            antibodies=list(self.viz_antibodies),
            samples_per_antibody=self.viz_samples_per_antibody,
        )
        if not isinstance(panel, list):
            return []

        groups: list[_PanelGroup] = []
        for entry in panel:
            if not isinstance(entry, dict):
                continue
            antibody = str(entry.get("antibody", "")).strip()
            x_value = entry.get("x")
            y_value = entry.get("y")
            if not antibody:
                continue
            if not isinstance(x_value, torch.Tensor) or not isinstance(y_value, torch.Tensor):
                continue
            if x_value.ndim != 4 or y_value.ndim != 4:
                continue
            groups.append(
                _PanelGroup(
                    tag_component=self._sanitize_tag_component(antibody),
                    x=x_value,
                    y=y_value,
                )
            )
        return groups

    @staticmethod
    def _is_valid_logging_context(trainer: pl.Trainer) -> bool:
        if not bool(getattr(trainer, "is_global_zero", True)):
            return False
        if bool(getattr(trainer, "sanity_checking", False)):
            return False
        return True

    def _ensure_panel_cache(self, trainer: pl.Trainer) -> bool:
        if self._panel_groups is None:
            self._panel_groups = self._build_panel_groups(trainer)
            if self._panel_groups:
                self._all_inputs_cpu = torch.cat([group.x for group in self._panel_groups], dim=0)
        return bool(self._panel_groups) and self._all_inputs_cpu is not None

    def _predict_panel_outputs(self, pl_module: pl.LightningModule) -> torch.Tensor | None:
        if self._all_inputs_cpu is None:
            return None
        with torch.inference_mode():
            predictions = pl_module(self._all_inputs_cpu.to(pl_module.device))
        if not isinstance(predictions, torch.Tensor) or predictions.ndim != 4:
            return None
        return predictions.detach().float().clamp(0.0, 1.0).cpu()

    def _log_group_images(
        self,
        *,
        experiment: object,
        group: _PanelGroup,
        prediction: torch.Tensor,
        global_step: int,
    ) -> None:
        target = group.y
        if prediction.shape != target.shape:
            return

        pred_grid = self._build_grid(prediction, int(group.x.shape[0]))
        target_tag = f"{group.tag_component}/target"
        pred_tag = f"{group.tag_component}/pred"
        error_tag = f"{group.tag_component}/error_abs"

        should_log_target = (not self.viz_log_target_once) or (
            target_tag not in self._target_logged_tags
        )
        if should_log_target:
            target_grid = self._build_grid(target, int(group.x.shape[0]))
            experiment.add_image(target_tag, target_grid, global_step=global_step)
            self._target_logged_tags.add(target_tag)
        experiment.add_image(pred_tag, pred_grid, global_step=global_step)
        if self.viz_log_error:
            error_grid = self._build_grid((target - prediction).abs(), int(group.x.shape[0]))
            experiment.add_image(error_tag, error_grid, global_step=global_step)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._is_valid_logging_context(trainer):
            return

        experiment = self._resolve_tb_experiment(trainer)
        if experiment is None:
            return

        global_step = int(getattr(trainer, "global_step", 0))
        if not self._should_log_for_step(global_step):
            return

        if not self._ensure_panel_cache(trainer):
            return

        predictions = self._predict_panel_outputs(pl_module)
        if predictions is None:
            return

        offset = 0
        panel_groups = self._panel_groups or []
        for group in panel_groups:
            sample_count = int(group.x.shape[0])
            prediction = predictions[offset : offset + sample_count]
            offset += sample_count
            self._log_group_images(
                experiment=experiment,
                group=group,
                prediction=prediction,
                global_step=global_step,
            )
