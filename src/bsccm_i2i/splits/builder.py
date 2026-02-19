"""Split artifact builder."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from pathlib import Path
from typing import Any

import bsccm

from bsccm_i2i.config.schema import SplitTaskConfig
from bsccm_i2i.datamodules.bsccm_datamodule import resolve_dataset_root
from bsccm_i2i.runners.paths import write_json
from bsccm_i2i.splits.io import write_indices_csv
from bsccm_i2i.splits.strategies import (
    random_fraction_split,
    stratified_antibodies_fraction_split,
)
from bsccm_i2i.utils.antibodies import normalize_antibody_label

LOGGER = logging.getLogger(__name__)


def _extract_antibody_labels(bsccm_client: Any, indices: list[int]) -> list[str]:
    index_dataframe = getattr(bsccm_client, "index_dataframe", None)
    if index_dataframe is None:
        raise ValueError(
            "split strategy 'stratified_antibodies' requires bsccm client index_dataframe"
        )
    columns = getattr(index_dataframe, "columns", None)
    if columns is None or "antibodies" not in columns:
        raise ValueError(
            "split strategy 'stratified_antibodies' requires an 'antibodies' column "
            "in BSCCM_index.csv"
        )

    labels: list[str] = []
    for index_value in indices:
        try:
            raw_value = index_dataframe.loc[index_value, "antibodies"]
        except Exception as exc:  # pragma: no cover - defensive for external dependency.
            raise ValueError(
                "failed to read antibody label for global_index="
                f"{index_value} from bsccm index dataframe"
            ) from exc
        if hasattr(raw_value, "iloc"):
            raw_value = raw_value.iloc[0]
        labels.append(normalize_antibody_label(raw_value))
    return labels


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_artifacts_root() -> Path:
    return Path("artifacts") / "splits"


def _build_split_id(split_task_config: SplitTaskConfig, now: dt.datetime) -> str:
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return (
        "bsccm_23to6__"
        f"{split_task_config.data.dataset_variant}__"
        f"{split_task_config.split.strategy}__"
        f"seed{split_task_config.split.seed}__{timestamp}"
    )


def build_split_artifact(split_task_config: SplitTaskConfig) -> dict[str, Any]:
    """Build a split artifact directory and return a concise summary."""
    created_at = dt.datetime.now()
    split_id = _build_split_id(split_task_config, created_at)
    artifact_dir = _split_artifacts_root() / split_id
    if artifact_dir.exists():
        raise RuntimeError(
            f"split artifact already exists: {artifact_dir}. "
            "Wait a second and retry or use different split settings."
        )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    LOGGER.info("Creating split artifact: split_id=%s dir=%s", split_id, artifact_dir)

    dataset_root = resolve_dataset_root(
        split_task_config.data.root_dir, split_task_config.data.dataset_variant
    )
    LOGGER.info("Resolved dataset root for split build: %s", dataset_root)
    bsccm_client = bsccm.BSCCM(str(dataset_root))
    all_indices = [int(value) for value in bsccm_client.get_indices(shuffle=False)]

    strategy = split_task_config.split.strategy.strip().lower()
    if strategy == "random":
        train_indices, val_indices, test_indices = random_fraction_split(
            indices=all_indices,
            train_frac=split_task_config.split.train_frac,
            val_frac=split_task_config.split.val_frac,
            seed=split_task_config.split.seed,
        )
    elif strategy == "stratified_antibodies":
        antibody_labels = _extract_antibody_labels(bsccm_client, all_indices)
        unique_antibodies = len(set(antibody_labels))
        LOGGER.info(
            "Applying antibody-stratified split across %d antibody groups",
            unique_antibodies,
        )
        train_indices, val_indices, test_indices = stratified_antibodies_fraction_split(
            indices=all_indices,
            antibodies=antibody_labels,
            train_frac=split_task_config.split.train_frac,
            val_frac=split_task_config.split.val_frac,
            seed=split_task_config.split.seed,
        )
    else:
        raise ValueError(f"unsupported split strategy: {split_task_config.split.strategy!r}")
    LOGGER.info(
        "Computed split indices: train=%d val=%d test=%d",
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    indices_path = artifact_dir / "indices.csv"
    write_indices_csv(indices_path, train_indices, val_indices, test_indices)

    split_definition = {
        "split_id": split_id,
        "strategy": strategy,
        "seed": split_task_config.split.seed,
        "train_frac": split_task_config.split.train_frac,
        "val_frac": split_task_config.split.val_frac,
        "test_frac": split_task_config.split.test_frac,
        "dataset_variant": split_task_config.data.dataset_variant,
        "created_at": created_at.isoformat(timespec="seconds"),
    }
    write_json(artifact_dir / "split.json", split_definition)

    # Fingerprint captures immutable dataset identity so future runs can verify
    # they reference the same underlying BSCCM files, not just the same split id.
    fingerprint = {
        "dataset_variant": split_task_config.data.dataset_variant,
        "bsccm_package_version": getattr(bsccm, "__version__", None),
        "bsccm_index_csv_sha256": _compute_sha256(dataset_root / "BSCCM_index.csv"),
        "bsccm_global_metadata_sha256": _compute_sha256(
            dataset_root / "BSCCM_global_metadata.json"
        ),
    }
    write_json(artifact_dir / "dataset_fingerprint.json", fingerprint)
    write_json(artifact_dir / "input_split_config.json", split_task_config.model_dump())
    LOGGER.info("Finished writing split artifact files at %s", artifact_dir)

    return {
        "split_id": split_id,
        "artifact_dir": str(artifact_dir),
        "counts": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
        },
    }
