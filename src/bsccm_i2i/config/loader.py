"""Hydra config composition and resolved serialization helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def _configs_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"


def load_config(
    config_path: Path | None,
    config_name: str,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load config from file path or compose from the repository Hydra tree."""
    if config_path is not None:
        loaded = OmegaConf.load(config_path)
        if not isinstance(loaded, DictConfig):
            loaded = OmegaConf.create(loaded)
        return loaded

    with initialize_config_dir(version_base=None, config_dir=str(_configs_dir())):
        return compose(config_name=config_name, overrides=list(overrides or []))


def to_resolved_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert an OmegaConf config to a plain resolved dictionary."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("Resolved config must be a mapping.")
    return resolved
