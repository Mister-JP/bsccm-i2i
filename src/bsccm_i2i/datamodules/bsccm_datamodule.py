"""BSCCM datamodule implementation for image-to-image tasks."""

from __future__ import annotations

import csv
import logging
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import bsccm
import torch
from dotenv import load_dotenv

from bsccm_i2i.datasets.bsccm_dataset import BSCCM23to6Dataset

LOGGER = logging.getLogger(__name__)


def is_dataset_root(path: Path) -> bool:
    """Check whether a directory looks like an extracted BSCCM dataset root."""
    return (
        path.is_dir()
        and (path / "BSCCM_global_metadata.json").is_file()
        and (path / "BSCCM_index.csv").is_file()
        and (path / "BSCCM_images.zarr").exists()
    )


def find_existing_dataset_root(path: Path) -> Path | None:
    """Return an existing dataset root at `path` or one level below it."""
    if is_dataset_root(path):
        return path
    if not path.is_dir():
        return None
    for child in path.iterdir():
        if child.is_dir() and is_dataset_root(child):
            return child
    return None


def get_dryad_token() -> str | None:
    """Load `.env` (if present) and return `BSCCM_DRYAD_TOKEN` when configured."""
    load_dotenv()
    token = os.getenv("BSCCM_DRYAD_TOKEN", "").strip()
    return token or None


def resolve_dataset_root(
    root_dir: str, variant: str, log_fn: Callable[[str], None] | None = None
) -> Path:
    """
    Resolve a usable BSCCM dataset directory.

    Uses an existing local root when present; otherwise triggers `bsccm.download_dataset`
    and validates that the resulting directory has required BSCCM files.
    """
    path = Path(root_dir)
    existing = find_existing_dataset_root(path)
    if existing is not None:
        if log_fn is not None:
            log_fn(f"Using existing dataset root at {existing}")
        return existing

    tiny_variant = variant.strip().lower() == "tiny"
    dryad_token = get_dryad_token()
    if log_fn is not None:
        token_state = "present" if dryad_token is not None else "missing"
        log_fn(
            f"No existing dataset found at {path}; downloading variant={variant!r} "
            f"(token={token_state})"
        )
    download_kwargs: dict[str, object] = {
        "location": str(path),
        "tiny": tiny_variant,
        "coherent": False,
    }
    if dryad_token is not None:
        download_kwargs["token"] = dryad_token
    try:
        downloaded = bsccm.download_dataset(**download_kwargs)
    except Exception as exc:  # pragma: no cover - depends on external service behavior.
        token_help = (
            " Set BSCCM_DRYAD_TOKEN in your environment or project .env file."
            if dryad_token is None
            else ""
        )
        raise RuntimeError(
            "Failed to download BSCCM dataset via bsccm.download_dataset. "
            "Set data.root_dir to an existing dataset or provide access credentials "
            f"required by the upstream source.{token_help}"
        ) from exc
    resolved = Path(downloaded)
    if log_fn is not None:
        log_fn(f"Download completed, validating dataset root at {resolved}")

    resolved_existing = find_existing_dataset_root(resolved)
    if resolved_existing is not None:
        if log_fn is not None:
            log_fn(f"Resolved dataset root: {resolved_existing}")
        return resolved_existing
    existing = find_existing_dataset_root(path)
    if existing is not None:
        if log_fn is not None:
            log_fn(f"Resolved dataset root: {existing}")
        return existing
    raise RuntimeError(
        f"Downloaded dataset path is invalid. Expected BSCCM root at {resolved} or {path}."
    )


def load_indices_csv(path: Path) -> dict[str, list[int]]:
    """
    Load sample indices from CSV.

    Supports common index column names and optional `split` labels (`train|val|test`).
    Returns a dictionary with `all`, `train`, `val`, and `test` index lists.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        field_names = [name.strip() for name in (reader.fieldnames or []) if name is not None]
        if not field_names:
            raise ValueError(f"indices CSV must include a header row: {path}")

        index_column = next(
            (name for name in ("global_index", "index", "idx", "id") if name in field_names),
            field_names[0],
        )
        split_column = "split" if "split" in field_names else None

        rows: dict[str, list[int]] = {"all": [], "train": [], "val": [], "test": []}
        for row_number, row in enumerate(reader, start=2):
            raw_index = row.get(index_column, "").strip()
            if not raw_index:
                raise ValueError(f"missing index value in {path}:{row_number}")
            try:
                index_value = int(raw_index)
            except ValueError as exc:
                raise ValueError(
                    f"invalid integer index in {path}:{row_number}: {raw_index}"
                ) from exc
            rows["all"].append(index_value)

            if split_column is None:
                continue

            split_name = row.get(split_column, "").strip().lower()
            if not split_name:
                continue
            if split_name not in {"train", "val", "test"}:
                raise ValueError(
                    f"invalid split value in {path}:{row_number}: {split_name!r} "
                    "(expected train|val|test)"
                )
            rows[split_name].append(index_value)

    if not rows["all"]:
        raise ValueError(f"indices CSV is empty: {path}")
    return rows


def split_indices(
    indices: list[int],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split a flat index list into train/val/test subsets with deterministic shuffling.

    Ensures train is non-empty and leaves the remainder for test after train and val cuts.
    """
    if not indices:
        raise ValueError("no dataset indices available to split")

    shuffled = list(indices)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_frac)
    val_count = int(total * val_frac)
    if train_count <= 0:
        raise ValueError("train split is empty; adjust split fractions or indices CSV")
    if train_count + val_count >= total:
        val_count = max(0, total - train_count - 1)

    train_indices = shuffled[:train_count]
    val_indices = shuffled[train_count : train_count + val_count]
    test_indices = shuffled[train_count + val_count :]

    if not train_indices:
        raise ValueError("train split is empty after split computation")
    return train_indices, val_indices, test_indices


class BSCCM23to6DataModule:
    """Simple datamodule that creates train/val/test dataloaders for BSCCM."""

    def __init__(
        self,
        *,
        root_dir: str,
        variant: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int,
        train_frac: float,
        val_frac: float,
        test_frac: float,
        indices_csv: str | None = None,
        log_progress: bool = False,
    ) -> None:
        """Store dataloader/split settings and prepare lazy dataset initialization."""
        self.root_dir = root_dir
        self.variant = variant
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.seed = int(seed)
        self.train_frac = float(train_frac)
        self.val_frac = float(val_frac)
        self.test_frac = float(test_frac)
        self.indices_csv = indices_csv
        self.log_progress = bool(log_progress)

        self._backend: Any | None = None
        self._train_dataset: BSCCM23to6Dataset | None = None
        self._val_dataset: BSCCM23to6Dataset | None = None
        self._test_dataset: BSCCM23to6Dataset | None = None

    def _log(self, message: str) -> None:
        """Emit datamodule progress logs when enabled by config."""
        if self.log_progress:
            LOGGER.info(message)

    def _build_backend(self) -> Any:
        """Instantiate and memoize a `bsccm.BSCCM` backend bound to resolved dataset root."""
        if self._backend is not None:
            return self._backend

        self._log(f"Resolving dataset root from {self.root_dir!r} (variant={self.variant!r})")
        dataset_root = resolve_dataset_root(
            root_dir=self.root_dir, variant=self.variant, log_fn=self._log
        )
        self._log(f"Opening BSCCM backend at {dataset_root}")
        self._backend = bsccm.BSCCM(str(dataset_root))
        self._log("BSCCM backend ready")
        return self._backend

    def setup(self) -> None:
        """
        Build train/val/test `BSCCM23to6Dataset` objects once.

        Uses CSV-provided indices when configured; otherwise uses backend indices and
        deterministic fraction-based splitting.
        """
        if self._train_dataset is not None:
            return

        self._log("Starting datamodule setup")
        backend = self._build_backend()

        if self.indices_csv:
            self._log(f"Using indices CSV: {self.indices_csv}")
            csv_indices = load_indices_csv(Path(self.indices_csv))
            has_explicit_splits = any(csv_indices[key] for key in ("train", "val", "test"))
            if has_explicit_splits:
                if not csv_indices["train"]:
                    raise ValueError("indices CSV split column provided, but train split is empty")
                train_indices = csv_indices["train"]
                val_indices = csv_indices["val"]
                test_indices = csv_indices["test"]
                self._log("Using explicit train/val/test splits from indices CSV")
            else:
                self._log("No explicit split labels in CSV; applying seeded fraction split")
                train_indices, val_indices, test_indices = split_indices(
                    indices=csv_indices["all"],
                    train_frac=self.train_frac,
                    val_frac=self.val_frac,
                    seed=self.seed,
                )
        else:
            self._log("Using backend indices with seeded fraction split")
            all_indices = [int(value) for value in backend.get_indices(shuffle=False)]
            train_indices, val_indices, test_indices = split_indices(
                indices=all_indices,
                train_frac=self.train_frac,
                val_frac=self.val_frac,
                seed=self.seed,
            )

        self._log(
            "Split sizes: "
            f"train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}"
        )
        self._train_dataset = BSCCM23to6Dataset(backend=backend, indices=train_indices)
        self._val_dataset = BSCCM23to6Dataset(
            backend=backend, indices=val_indices or train_indices[:1]
        )
        self._test_dataset = BSCCM23to6Dataset(
            backend=backend, indices=test_indices or train_indices[:1]
        )
        self._log("Datasets initialized")

    def _make_dataloader(self, dataset: BSCCM23to6Dataset, *, shuffle: bool) -> Any:
        """
        Create a torch `DataLoader` from a prepared dataset using configured options.

        When `shuffle=True`, a fixed-seed torch generator is provided so batch order is
        reproducible across runs with the same seed.
        """
        data_loader_cls = torch.utils.data.DataLoader
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        try:
            dataset_len = len(dataset)
        except TypeError:
            dataset_len = "unknown"
        self._log(
            "Creating DataLoader "
            f"(shuffle={shuffle}, batch_size={self.batch_size}, "
            f"num_workers={self.num_workers}, dataset_len={dataset_len})"
        )
        return data_loader_cls(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=generator,
            worker_init_fn=self._seed_worker,
            persistent_workers=self.num_workers > 0,
        )

    def _seed_worker(self, worker_id: int) -> None:
        """Seed worker-local RNGs for deterministic multi-worker data loading."""
        worker_seed = self.seed + int(worker_id)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        try:
            import numpy as np
        except ModuleNotFoundError:
            return
        np.random.seed(worker_seed % (2**32))

    def train_dataloader(self) -> Any:
        """Return the training dataloader, initializing datasets on first call."""
        self.setup()
        if self._train_dataset is None:
            raise RuntimeError("train dataset is not initialized")
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> Any:
        """Return the validation dataloader, initializing datasets on first call."""
        self.setup()
        if self._val_dataset is None:
            raise RuntimeError("val dataset is not initialized")
        return self._make_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> Any:
        """Return the test dataloader, initializing datasets on first call."""
        self.setup()
        if self._test_dataset is None:
            raise RuntimeError("test dataset is not initialized")
        return self._make_dataloader(self._test_dataset, shuffle=False)
