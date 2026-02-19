"""Lightning training runner and standardized run-artifact orchestration."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from bsccm_i2i.callbacks import I2IVizCallback
from bsccm_i2i.config.schema import SplitRefCounts, TrainConfig
from bsccm_i2i.models.registry import build_model
from bsccm_i2i.runners.artifacts import write_split_ref
from bsccm_i2i.runners.common import (
    build_datamodule_from_train_config,
    configure_torch_determinism,
    extract_scalar_metrics,
    make_train_trainer,
    require_explicit_split_id,
)
from bsccm_i2i.runners.paths import create_run_dir, write_env_snapshot, write_json, write_yaml
from bsccm_i2i.splits.registry import (
    load_split_indices,
    load_split_metadata,
    resolve_split_dir,
    validate_split_matches_config,
)

LOGGER = logging.getLogger(__name__)


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

    split_id = require_explicit_split_id(train_cfg.split.name)
    split_dir = resolve_split_dir(split_id)
    split_metadata = load_split_metadata(split_id)
    validate_split_matches_config(split_metadata=split_metadata, train_cfg=train_cfg)
    split_indices = load_split_indices(split_id)
    indices_csv = str(split_dir / "indices.csv")

    write_split_ref(
        run_dir=run_dir,
        split_id=split_id,
        split_dir=split_dir,
        indices_csv=indices_csv,
        fingerprint=split_metadata["fingerprint"],
        counts=SplitRefCounts(
            train=len(split_indices["train"]),
            val=len(split_indices["val"]),
            test=len(split_indices["test"]),
        ),
    )

    if train_cfg.trainer.deterministic:
        configure_torch_determinism(seed=train_cfg.trainer.seed)

    datamodule = build_datamodule_from_train_config(
        train_cfg,
        indices_csv=indices_csv,
    )
    model = build_model(train_cfg.model)

    callbacks: list[pl.Callback] = [
        I2IVizCallback(
            num_viz_samples=train_cfg.logging.num_viz_samples,
            image_log_every_n_steps=train_cfg.logging.image_log_every_n_steps,
        )
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

    trainer = make_train_trainer(
        run_dir=run_dir,
        train_cfg=train_cfg,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model=model, datamodule=datamodule)

    final_metrics = extract_scalar_metrics(dict(trainer.callback_metrics))
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
