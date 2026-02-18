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
from bsccm_i2i.runners.paths import create_run_dir, write_yaml

DATAMODULE_LOGGER_NAME = "bsccm_i2i.datamodules.bsccm_datamodule"

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


def _run_split(config: ConfigPath, overrides: Overrides) -> None:
    _validate_config_or_overrides(config=config, overrides=overrides)
    cfg = load_config(config_path=config, config_name="task/i2i_23to6", overrides=overrides or [])
    SplitTaskConfig.model_validate(to_resolved_dict(cfg))
    typer.echo("CALLED split")


def _run_train(config: ConfigPath, overrides: Overrides) -> None:
    _validate_config_or_overrides(config=config, overrides=overrides)
    cfg = load_config(config_path=config, config_name="task/train", overrides=overrides or [])
    resolved_cfg = to_resolved_dict(cfg)
    train_cfg = TrainConfig.model_validate(resolved_cfg)
    if train_cfg.trainer.deterministic:
        configure_torch_determinism(seed=train_cfg.trainer.seed)

    run_dir = create_run_dir(train_cfg.run.run_name)
    write_yaml(run_dir / "config_resolved.yaml", resolved_cfg)
    configure_datamodule_logging(enabled=train_cfg.logging.data_progress)

    smoke_steps = train_cfg.trainer.max_steps if train_cfg.trainer.max_steps > 0 else 0
    if train_cfg.trainer.smoke and smoke_steps == 0:
        smoke_steps = 2
    if smoke_steps > 0:
        run_train_smoke_loader(train_cfg=train_cfg, max_steps=smoke_steps)

    typer.echo(f"RUN_DIR {run_dir}")
    typer.echo("CALLED train")


def run_train_smoke_loader(train_cfg: TrainConfig, max_steps: int) -> None:
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
        indices_csv=train_cfg.data.indices_csv,
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
    logger = logging.getLogger(DATAMODULE_LOGGER_NAME)
    logger.propagate = False

    handler = next(
        (h for h in logger.handlers if getattr(h, "_bsccm_datamodule_handler", False)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[bsccm-datamodule] %(message)s"))
        handler._bsccm_datamodule_handler = True
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
