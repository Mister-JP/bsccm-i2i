"""Typer CLI entrypoint with stable command surface for Story 2."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from bsccm_i2i import __version__
from bsccm_i2i.config.loader import load_config, to_resolved_dict
from bsccm_i2i.config.schema import SplitTaskConfig, TrainConfig
from bsccm_i2i.runners.train import run_train
from bsccm_i2i.splits.builder import build_split_artifact

SPLITS_BUILDER_LOGGER_NAME = "bsccm_i2i.splits.builder"
SPLITS_REGISTRY_LOGGER_NAME = "bsccm_i2i.splits.registry"
LOGGER = logging.getLogger(__name__)

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
    train_config = TrainConfig.model_validate(to_resolved_dict(hydra_cfg))
    try:
        train_run_dir = run_train(train_config)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"RUN_DIR {train_run_dir}")
    typer.echo("CALLED train")


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
