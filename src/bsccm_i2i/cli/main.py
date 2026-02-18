"""Typer CLI entrypoint with stable command surface for Story 2."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from dotenv import load_dotenv

from bsccm_i2i import __version__
from bsccm_i2i.config.loader import load_config, to_resolved_dict
from bsccm_i2i.config.schema import SplitTaskConfig, TrainConfig
from bsccm_i2i.datamodules.bsccm_datamodule import BSCCM23to6DataModule
from bsccm_i2i.runners.paths import create_run_dir, write_json
from bsccm_i2i.splits.builder import build_split_artifact
from bsccm_i2i.splits.registry import (
    load_split_indices,
    load_split_metadata,
    resolve_split_dir,
    validate_split_matches_config,
)

DATAMODULE_LOGGER_NAME = "bsccm_i2i.datamodules.bsccm_datamodule"
SPLITS_BUILDER_LOGGER_NAME = "bsccm_i2i.splits.builder"
SPLITS_REGISTRY_LOGGER_NAME = "bsccm_i2i.splits.registry"
LOGGER = logging.getLogger(__name__)
REQUIRED_SPLIT_ID_PLACEHOLDER = "REQUIRED_SPLIT_ID"

app = typer.Typer(
    name="bsccm-i2i",
    help="BSCCM image-to-image experiment CLI.",
    add_completion=False,
    no_args_is_help=True,
)

ConfigPath = Annotated[
    Path | None,
    typer.Option(
        "--config",
        exists=False,
        dir_okay=False,
        help="Path to a config file (alternative to key=value overrides).",
    ),
]
Overrides = Annotated[
    list[str] | None,
    typer.Argument(
        metavar="KEY=VALUE",
        help="Hydra-style key=value override(s).",
    ),
]


def _validate_config_or_overrides(config: Path | None, overrides: list[str]) -> None:
    if config is not None and overrides:
        raise typer.BadParameter("Use either --config or key=value overrides, not both.")


def _require_explicit_split_id(split_id: str) -> str:
    normalized = split_id.strip()
    if not normalized or normalized == REQUIRED_SPLIT_ID_PLACEHOLDER:
        raise typer.BadParameter(
            "train requires an explicit split artifact id via split.name=<SPLIT_ID>. "
            "Run `bsccm-i2i split` first, then rerun train with the printed SPLIT_ID. "
            "Automatic split creation during train is intentionally disabled."
        )
    return normalized


def _run_split(config: ConfigPath, overrides: Overrides) -> None:
    _validate_config_or_overrides(config=config, overrides=overrides)
    configure_split_logging(enabled=True)
    LOGGER.info("Starting split command with overrides=%s", overrides or [])
    cfg = load_config(config_path=config, config_name="task/split", overrides=overrides or [])
    task_cfg = SplitTaskConfig.model_validate(to_resolved_dict(cfg))
    summary = build_split_artifact(task_cfg)
    counts = summary["counts"]
    typer.echo(f"SPLIT_ID {summary['split_id']}")
    typer.echo(f"SPLIT_DIR {summary['artifact_dir']}")
    typer.echo(
        f"SPLIT_COUNTS train={counts['train']} val={counts['val']} test={counts['test']}"
    )


def _run_train(config: ConfigPath, overrides: Overrides) -> None:
    _validate_config_or_overrides(config=config, overrides=overrides)
    LOGGER.info("Starting train command with overrides=%s", overrides or [])
    hydra_cfg = load_config(config_path=config, config_name="task/train", overrides=overrides or [])
    train_input_config = to_resolved_dict(hydra_cfg)
    train_config = TrainConfig.model_validate(train_input_config)
    configure_split_logging(enabled=train_config.logging.data_progress)
    split_artifact_id = _require_explicit_split_id(train_config.split.name)
    try:
        split_artifact_dir = resolve_split_dir(split_artifact_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"split artifact id {split_artifact_id!r} was not found under artifacts/splits. "
            "Run `bsccm-i2i split` and rerun train with split.name=<SPLIT_ID>."
        ) from exc
    split_indices_csv_path = str(split_artifact_dir / "indices.csv")
    split_artifact_metadata = load_split_metadata(split_artifact_id)
    validate_split_matches_config(split_metadata=split_artifact_metadata, train_cfg=train_config)
    split_index_groups = load_split_indices(split_artifact_id)
    LOGGER.info(
        "Using split artifact %s (indices_csv=%s)",
        split_artifact_id,
        split_indices_csv_path,
    )
    if train_config.trainer.deterministic:
        configure_torch_determinism(seed=train_config.trainer.seed)

    train_run_dir = create_run_dir(train_config.run.run_name)
    write_json(train_run_dir / "input_train_config.json", train_input_config)
    write_json(
        train_run_dir / "split_ref.json",
        {
            "split_id": split_artifact_id,
            "split_dir": str(split_artifact_dir),
            "fingerprint": split_artifact_metadata["fingerprint"],
            "counts": {
                "train": len(split_index_groups["train"]),
                "val": len(split_index_groups["val"]),
                "test": len(split_index_groups["test"]),
            },
        },
    )
    configure_datamodule_logging(enabled=train_config.logging.data_progress)

    smoke_max_steps = train_config.trainer.max_steps if train_config.trainer.max_steps > 0 else 0
    if train_config.trainer.smoke and smoke_max_steps == 0:
        smoke_max_steps = 2
    if smoke_max_steps > 0:
        run_train_smoke_loader(
            train_cfg=train_config,
            max_steps=smoke_max_steps,
            indices_csv=split_indices_csv_path,
        )

    typer.echo(f"RUN_DIR {train_run_dir}")
    typer.echo("CALLED train")


def run_train_smoke_loader(train_cfg: TrainConfig, max_steps: int, indices_csv: str) -> None:
    datamodule = BSCCM23to6DataModule(
        root_dir=train_cfg.data.root_dir,
        variant=train_cfg.data.variant,
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

    loader = datamodule.train_dataloader()
    iterator = iter(loader)
    for step in range(max_steps):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"train dataloader exhausted after {step} steps; expected at least {max_steps}"
            ) from exc
        x, y = batch
        typer.echo(f"SMOKE_BATCH step={step + 1} x_shape={tuple(x.shape)} y_shape={tuple(y.shape)}")


def configure_torch_determinism(seed: int) -> None:
    """Configure python/numpy/torch RNG and deterministic torch backend behavior."""
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


def configure_datamodule_logging(enabled: bool) -> None:
    """Enable/disable human-readable datamodule progress logs."""
    configure_prefixed_logging(
        logger_name=DATAMODULE_LOGGER_NAME,
        handler_marker="_bsccm_datamodule_handler",
        prefix="[bsccm-datamodule] %(message)s",
        enabled=enabled,
    )


def configure_split_logging(enabled: bool) -> None:
    """Enable/disable split subsystem progress logs."""
    configure_prefixed_logging(
        logger_name=SPLITS_BUILDER_LOGGER_NAME,
        handler_marker="_bsccm_splits_builder_handler",
        prefix="[bsccm-splits] %(message)s",
        enabled=enabled,
    )
    configure_prefixed_logging(
        logger_name=SPLITS_REGISTRY_LOGGER_NAME,
        handler_marker="_bsccm_splits_registry_handler",
        prefix="[bsccm-splits] %(message)s",
        enabled=enabled,
    )


def configure_prefixed_logging(
    *,
    logger_name: str,
    handler_marker: str,
    prefix: str,
    enabled: bool,
) -> None:
    """Set up one prefixed stream logger with on/off control."""
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    handler = next((h for h in logger.handlers if getattr(h, handler_marker, False)), None)
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(prefix))
        setattr(handler, handler_marker, True)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO if enabled else logging.WARNING)


def _run_eval(config: ConfigPath, overrides: Overrides) -> None:
    _validate_config_or_overrides(config=config, overrides=overrides)
    typer.echo("CALLED eval")


def _run_report(config: ConfigPath, overrides: Overrides) -> None:
    _validate_config_or_overrides(config=config, overrides=overrides)
    typer.echo("CALLED report")


VersionFlag = Annotated[bool, typer.Option("--version", help="Show version and exit.")]


@app.callback()
def root_callback(version: VersionFlag = False) -> None:
    load_dotenv()
    if version:
        typer.echo(f"bsccm-i2i {__version__}")
        raise typer.Exit()


@app.command("split")
def split(
    config: ConfigPath = None,
    overrides: Overrides = None,
) -> None:
    _run_split(config=config, overrides=overrides or [])


@app.command("train")
def train(
    config: ConfigPath = None,
    overrides: Overrides = None,
) -> None:
    _run_train(config=config, overrides=overrides or [])


@app.command("eval")
def eval_cmd(
    config: ConfigPath = None,
    overrides: Overrides = None,
) -> None:
    _run_eval(config=config, overrides=overrides or [])


@app.command("report")
def report(
    config: ConfigPath = None,
    overrides: Overrides = None,
) -> None:
    _run_report(config=config, overrides=overrides or [])


def main(argv: Sequence[str] | None = None) -> int:
    app(args=list(argv) if argv is not None else None, prog_name="bsccm-i2i")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
