"""Lightning training runner and standardized run-artifact orchestration."""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from bsccm_i2i.callbacks import I2IVizCallback
from bsccm_i2i.config.schema import TrainConfig
from bsccm_i2i.datamodules.bsccm_datamodule import BSCCM23to6DataModule
from bsccm_i2i.models.registry import build_model
from bsccm_i2i.runners.paths import create_run_dir, write_env_snapshot, write_json, write_yaml
from bsccm_i2i.splits.registry import (
    load_split_indices,
    load_split_metadata,
    resolve_split_dir,
    validate_split_matches_config,
)

REQUIRED_SPLIT_ID_PLACEHOLDER = "REQUIRED_SPLIT_ID"
LOGGER = logging.getLogger(__name__)


def _require_explicit_split_id(split_id: str) -> str:
    normalized = split_id.strip()
    if not normalized or normalized == REQUIRED_SPLIT_ID_PLACEHOLDER:
        raise ValueError(
            "train requires an explicit split artifact id via split.name=<SPLIT_ID>. "
            "Run `bsccm-i2i split` first, then rerun train with the printed SPLIT_ID. "
            "Automatic split creation during train is intentionally disabled."
        )
    return normalized


def _configure_torch_determinism(seed: int) -> None:
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


def _normalize_precision(raw_precision: str) -> str:
    value = raw_precision.strip().lower()
    if value == "32":
        return "32-true"
    if value == "16":
        return "16-mixed"
    if value == "bf16":
        return "bf16-mixed"
    return raw_precision


def _resolve_accelerator(device: str) -> tuple[str, int]:
    normalized = device.strip().lower()
    if normalized in {"gpu", "cuda"}:
        return "gpu", 1
    if normalized in {"mps"}:
        return "mps", 1
    if normalized in {"auto"}:
        return "auto", 1
    return "cpu", 1


def _extract_scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
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


def _write_epoch_metrics_csv(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_keys = ["epoch", "global_step"]
    for key in sorted(metrics):
        if key not in ordered_keys:
            ordered_keys.append(key)

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=ordered_keys)
        writer.writeheader()
        writer.writerow({key: metrics.get(key, "") for key in ordered_keys})


def _read_git_commit(run_dir: Path) -> str | None:
    commit_path = run_dir / "env" / "git_commit.txt"
    if not commit_path.is_file():
        return None
    value = commit_path.read_text(encoding="utf-8").strip()
    return value or None


def run_train(train_cfg: TrainConfig) -> Path:
    """Execute model training and emit standardized run artifacts."""
    run_dir = create_run_dir(train_cfg.run.run_name)
    write_yaml(run_dir / "config_resolved.yaml", train_cfg.model_dump(mode="json"))
    write_env_snapshot(run_dir)

    split_id = _require_explicit_split_id(train_cfg.split.name)
    split_dir = resolve_split_dir(split_id)
    split_metadata = load_split_metadata(split_id)
    validate_split_matches_config(split_metadata=split_metadata, train_cfg=train_cfg)
    split_indices = load_split_indices(split_id)
    indices_csv = str(split_dir / "indices.csv")

    write_yaml(
        run_dir / "split_ref.yaml",
        {
            "split_id": split_id,
            "split_dir": str(split_dir),
            "indices_csv": indices_csv,
            "fingerprint": split_metadata["fingerprint"],
            "counts": {
                "train": len(split_indices["train"]),
                "val": len(split_indices["val"]),
                "test": len(split_indices["test"]),
            },
        },
    )

    if train_cfg.trainer.deterministic:
        _configure_torch_determinism(seed=train_cfg.trainer.seed)

    datamodule = BSCCM23to6DataModule(
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
    model = build_model(train_cfg.model)

    callbacks: list[pl.Callback] = [
        I2IVizCallback(num_viz_samples=train_cfg.logging.num_viz_samples)
    ]
    checkpoint_callback: ModelCheckpoint | None = None
    if train_cfg.trainer.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="best",
            monitor="loss/val",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks.insert(0, checkpoint_callback)

    logger: TensorBoardLogger | bool
    if train_cfg.trainer.logger and train_cfg.logging.tensorboard:
        try:
            logger = TensorBoardLogger(
                save_dir=str(run_dir / "tensorboard"),
                name="",
                version="",
                default_hp_metric=False,
            )
        except ModuleNotFoundError:
            LOGGER.warning(
                "TensorBoard logger disabled because tensorboard dependencies are unavailable."
            )
            logger = False
    else:
        logger = False

    accelerator, devices = _resolve_accelerator(train_cfg.trainer.device)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=_normalize_precision(train_cfg.trainer.precision),
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
    trainer.fit(model=model, datamodule=datamodule)

    final_metrics = _extract_scalar_metrics(dict(trainer.callback_metrics))
    final_metrics["epoch"] = float(getattr(trainer, "current_epoch", 0))
    final_metrics["global_step"] = float(getattr(trainer, "global_step", 0))
    _write_epoch_metrics_csv(run_dir / "metrics" / "epoch_metrics.csv", final_metrics)

    best_metric_value: float | None = None
    best_checkpoint_path: str | None = None
    if checkpoint_callback is not None:
        best_score = checkpoint_callback.best_model_score
        if isinstance(best_score, torch.Tensor) and best_score.numel() == 1:
            best_metric_value = float(best_score.detach().cpu().item())
        if checkpoint_callback.best_model_path:
            best_checkpoint_path = checkpoint_callback.best_model_path

    write_json(
        run_dir / "report.json",
        {
            "split_id": split_id,
            "best_metric": {
                "name": "loss/val",
                "value": best_metric_value,
            },
            "best_checkpoint_path": best_checkpoint_path,
            "git_commit": _read_git_commit(run_dir),
        },
    )
    return run_dir
