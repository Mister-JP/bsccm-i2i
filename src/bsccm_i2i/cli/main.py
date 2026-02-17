"""Minimal CLI entrypoint for Story 1 bootstrap."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from bsccm_i2i import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsccm-i2i",
        description="BSCCM image-to-image experiment CLI (bootstrap).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path placeholder for future stories.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
