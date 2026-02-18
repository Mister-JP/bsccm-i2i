"""Shared runner helpers for train/eval runtime behavior."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch

from bsccm_i2i.config.schema import EvalConfig, TrainConfig
from bsccm_i2i.datamodules.bsccm_datamodule import BSCCM23to6DataModule

REQUIRED_SPLIT_ID_PLACEHOLDER = "REQUIRED_SPLIT_ID"


def require_explicit_split_id(split_id: str) -> str:
    """Validate that split id is explicitly set and not left as placeholder."""
    normalized = split_id.strip()
    if not normalized or normalized == REQUIRED_SPLIT_ID_PLACEHOLDER:
        raise ValueError(
            "train requires an explicit split artifact id via split.name=<SPLIT_ID>. "
            "Run `bsccm-i2i split` first, then rerun train with the printed SPLIT_ID. "
            "Automatic split creation during train is intentionally disabled."
        )
    return normalized


def configure_torch_determinism(seed: int) -> None:
    """Configure torch and related RNGs for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_precision(raw_precision: str) -> str:
    """Map user precision strings to Lightning precision modes."""
    value = raw_precision.strip().lower()
    if value == "32":
        return "32-true"
    if value == "16":
        return "16-mixed"
    if value == "bf16":
        return "bf16-mixed"
    return raw_precision


def resolve_accelerator(device: str) -> tuple[str, int | str]:
    """Resolve trainer accelerator/devices pair from user device config."""
    normalized = device.strip().lower()
    if normalized in {"gpu", "cuda"}:
        return "gpu", "auto"
    if normalized in {"mps"}:
        return "mps", 1
    if normalized in {"auto"}:
        if torch.cuda.is_available():
            return "gpu", "auto"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and mps_backend.is_available():
            return "mps", 1
        return "cpu", 1
    return "cpu", 1


def extract_scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Convert tensor/scalar callback metrics into plain float payloads."""
    scalar_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            scalar_metrics[key] = float(value.detach().cpu().item())
            continue
        if isinstance(value, (int, float)):
            scalar_metrics[key] = float(value)
    return scalar_metrics


def build_datamodule_from_train_config(
    train_cfg: TrainConfig,
    *,
    indices_csv: str,
) -> BSCCM23to6DataModule:
    """Construct a datamodule using the data/split settings from train config."""
    return BSCCM23to6DataModule(
        root_dir=train_cfg.data.root_dir,
        dataset_variant=train_cfg.data.dataset_variant,
        batch_size=train_cfg.data.batch_size,
        num_workers=train_cfg.data.num_workers,
        pin_memory=train_cfg.data.pin_memory,
        seed=train_cfg.split.seed,
        train_frac=train_cfg.split.train_frac,
        val_frac=train_cfg.split.val_frac,
        test_frac=train_cfg.split.test_frac,
        indices_csv=indices_csv,
        log_progress=train_cfg.logging.data_progress,
    )


def make_train_trainer(
    *,
    run_dir: Path,
    train_cfg: TrainConfig,
    callbacks: list[pl.Callback],
    logger: Any,
) -> pl.Trainer:
    """Build a Lightning trainer for training runs with standardized defaults."""
    accelerator, devices = resolve_accelerator(train_cfg.trainer.device)
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=normalize_precision(train_cfg.trainer.precision),
        max_epochs=train_cfg.trainer.max_epochs,
        max_steps=train_cfg.trainer.max_steps if train_cfg.trainer.max_steps > 0 else -1,
        overfit_batches=train_cfg.trainer.overfit_n if train_cfg.trainer.overfit_n > 0 else 0.0,
        deterministic=train_cfg.trainer.deterministic,
        limit_val_batches=train_cfg.trainer.limit_val_batches,
        log_every_n_steps=train_cfg.logging.log_every_n_steps,
        enable_checkpointing=train_cfg.trainer.enable_checkpointing,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(run_dir),
    )


def make_eval_trainer(
    *,
    run_dir: Path,
    eval_cfg: EvalConfig,
    deterministic: bool,
) -> pl.Trainer:
    """Build a Lightning trainer for eval-only test execution."""
    accelerator, devices = resolve_accelerator(eval_cfg.device)
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=normalize_precision(eval_cfg.precision),
        deterministic=deterministic,
        limit_test_batches=eval_cfg.limit_test_batches,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=str(run_dir),
    )
