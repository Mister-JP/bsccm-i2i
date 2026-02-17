"""Typer CLI entrypoint with stable command surface for Story 2."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from bsccm_i2i import __version__

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
    typer.echo("CALLED split")


def _run_train(config: ConfigPath, overrides: Overrides) -> None:
    _validate_config_or_overrides(config=config, overrides=overrides)
    typer.echo("CALLED train")


def _run_eval(config: ConfigPath, overrides: Overrides) -> None:
    _validate_config_or_overrides(config=config, overrides=overrides)
    typer.echo("CALLED eval")


def _run_report(config: ConfigPath, overrides: Overrides) -> None:
    _validate_config_or_overrides(config=config, overrides=overrides)
    typer.echo("CALLED report")


VersionFlag = Annotated[bool, typer.Option("--version", help="Show version and exit.")]


@app.callback()
def root_callback(version: VersionFlag = False) -> None:
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
