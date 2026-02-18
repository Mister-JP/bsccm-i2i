"""Split artifact registry helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bsccm_i2i.config.schema import TrainConfig
from bsccm_i2i.splits.io import read_indices_csv

LOGGER = logging.getLogger(__name__)


def _split_artifacts_root() -> Path:
    return Path("artifacts") / "splits"


def resolve_split_dir(split_id: str) -> Path:
    """Resolve `artifacts/splits/<split_id>` and require it to exist."""
    normalized = split_id.strip()
    if not normalized:
        raise ValueError("split.name must be a non-empty split artifact id")

    split_dir = _split_artifacts_root() / normalized
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split artifact directory not found: {split_dir}")
    LOGGER.info("Resolved split artifact directory: %s", split_dir)
    return split_dir


def load_split_indices(split_id: str) -> dict[str, list[int]]:
    """Load train/val/test indices for a split artifact id."""
    split_dir = resolve_split_dir(split_id)
    indices_path = split_dir / "indices.csv"
    if not indices_path.is_file():
        raise FileNotFoundError(f"split artifact is missing indices.csv: {split_dir}")
    rows = read_indices_csv(indices_path)
    LOGGER.info(
        "Loaded split indices for %s: train=%d val=%d test=%d",
        split_id,
        len(rows["train"]),
        len(rows["val"]),
        len(rows["test"]),
    )
    return rows


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing split artifact file: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected mapping object in {path}")
    return loaded


def load_split_metadata(split_id: str) -> dict[str, Any]:
    """Read split metadata and fingerprint from a split artifact id."""
    split_dir = resolve_split_dir(split_id)
    split_json = _load_mapping(split_dir / "split.json")
    fingerprint = _load_mapping(split_dir / "dataset_fingerprint.json")
    LOGGER.info("Loaded split metadata and dataset fingerprint for %s", split_id)
    return {
        "split_id": split_id,
        "split_dir": str(split_dir),
        "split": split_json,
        "fingerprint": fingerprint,
    }


def validate_split_matches_config(split_metadata: dict[str, Any], train_cfg: TrainConfig) -> None:
    """Validate split artifact metadata against train config and fail on mismatch."""
    split_payload = split_metadata.get("split")
    if not isinstance(split_payload, dict):
        raise ValueError("split metadata is missing split.json content")

    mismatches: list[str] = []
    if split_payload.get("variant") != train_cfg.data.variant:
        mismatches.append(
            f"variant split={split_payload.get('variant')!r} config={train_cfg.data.variant!r}"
        )
    if split_payload.get("strategy") != train_cfg.split.strategy:
        mismatches.append(
            f"strategy split={split_payload.get('strategy')!r} config={train_cfg.split.strategy!r}"
        )
    if int(split_payload.get("seed", -1)) != int(train_cfg.split.seed):
        mismatches.append(f"seed split={split_payload.get('seed')} config={train_cfg.split.seed}")

    tolerance = 1e-6
    fraction_keys = ("train_frac", "val_frac", "test_frac")
    for key in fraction_keys:
        split_value = float(split_payload.get(key, float("nan")))
        config_value = float(getattr(train_cfg.split, key))
        if abs(split_value - config_value) > tolerance:
            mismatches.append(f"{key} split={split_value} config={config_value}")

    if mismatches:
        LOGGER.error("Split config validation failed: %s", "; ".join(mismatches))
        raise ValueError(
            "This run config doesn't match the split artifact; regenerate split or update config. "
            + "; ".join(mismatches)
        )
    LOGGER.info(
        "Split config validation succeeded for split_id=%s",
        split_metadata.get("split_id"),
    )
