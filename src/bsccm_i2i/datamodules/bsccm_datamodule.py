"""BSCCM datamodule implementation for image-to-image tasks."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import bsccm
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from bsccm_i2i.datasets.bsccm_dataset import BSCCM23to6Dataset
from bsccm_i2i.splits.io import read_indices_csv
from bsccm_i2i.splits.strategies import random_fraction_split
from bsccm_i2i.utils.antibodies import normalize_antibody_label
from bsccm_i2i.utils.quiet import run_quietly
_STAGE_TO_SPLITS: dict[str | None, tuple[str, ...]] = {
    None: ("train", "val", "test"),
    "fit": ("train", "val"),
    "validate": ("val",),
    "test": ("test",),
    "predict": ("test",),
}


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    """Seed worker-local RNGs for deterministic multi-worker data loading."""
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    try:
        import numpy as np
    except ModuleNotFoundError:
        return
    np.random.seed(worker_seed % (2**32))


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
    root_dir: str, dataset_variant: str, log_fn: Callable[[str], None] | None = None
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

    tiny_variant = dataset_variant.strip().lower() == "tiny"
    dryad_token = get_dryad_token()
    if log_fn is not None:
        token_state = "present" if dryad_token is not None else "missing"
        log_fn(
            f"No existing dataset found at {path}; downloading dataset_variant={dataset_variant!r} "
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


class BSCCM23to6DataModule(pl.LightningDataModule):
    """Simple datamodule that creates train/val/test dataloaders for BSCCM."""

    def __init__(
        self,
        *,
        root_dir: str,
        dataset_variant: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int | None = None,
        seed: int,
        train_frac: float,
        val_frac: float,
        test_frac: float,
        indices_csv: str | None = None,
        log_progress: bool = False,
    ) -> None:
        """Store dataloader/split settings and prepare lazy dataset initialization."""
        super().__init__()
        self.root_dir = root_dir
        self.dataset_variant = dataset_variant
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        self.seed = int(seed)
        self.train_frac = float(train_frac)
        self.val_frac = float(val_frac)
        self.test_frac = float(test_frac)
        self.indices_csv = indices_csv
        self.log_progress = bool(log_progress)

        self._bsccm_client: Any | None = None
        self._dataset_root: Path | None = None
        self._split_indices: dict[str, list[int]] | None = None
        self._datasets: dict[str, BSCCM23to6Dataset | None] = {
            "train": None,
            "val": None,
            "test": None,
        }

    def _log(self, message: str) -> None:
        """Emit datamodule progress logs when enabled by config."""
        if self.log_progress:
            rank_zero_info(message)

    def _build_bsccm_client(self) -> Any:
        """Instantiate and memoize a `bsccm.BSCCM` client bound to resolved dataset root."""
        if self._bsccm_client is not None:
            return self._bsccm_client

        self._log(
            "Resolving dataset root from "
            f"{self.root_dir!r} (dataset_variant={self.dataset_variant!r})"
        )
        dataset_root = resolve_dataset_root(
            root_dir=self.root_dir,
            dataset_variant=self.dataset_variant,
            log_fn=self._log,
        )
        self._dataset_root = dataset_root
        self._log(f"Opening BSCCM client at {dataset_root}")
        self._bsccm_client = run_quietly(lambda: bsccm.BSCCM(str(dataset_root)))
        self._log("BSCCM client ready")
        return self._bsccm_client

    def _required_splits(self, stage: str | None) -> tuple[str, ...]:
        normalized = None if stage is None else stage.strip().lower()
        try:
            return _STAGE_TO_SPLITS[normalized]
        except KeyError as exc:
            allowed = ", ".join(repr(name) for name in _STAGE_TO_SPLITS)
            raise ValueError(f"Unsupported stage={stage!r}. Expected one of: {allowed}") from exc

    def _resolve_split_indices(self) -> dict[str, list[int]]:
        if self._split_indices is not None:
            return self._split_indices

        if self.indices_csv:
            self._log(f"Using indices CSV: {self.indices_csv}")
            csv_indices = read_indices_csv(Path(self.indices_csv))
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
                train_indices, val_indices, test_indices = random_fraction_split(
                    indices=csv_indices["all"],
                    train_frac=self.train_frac,
                    val_frac=self.val_frac,
                    seed=self.seed,
                )
        else:
            self._log("Using BSCCM client indices with seeded fraction split")
            bsccm_client = self._build_bsccm_client()
            all_indices = [int(value) for value in bsccm_client.get_indices(shuffle=False)]
            train_indices, val_indices, test_indices = random_fraction_split(
                indices=all_indices,
                train_frac=self.train_frac,
                val_frac=self.val_frac,
                seed=self.seed,
            )

        self._log(
            "Split sizes: "
            f"train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}"
        )
        self._split_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }
        return self._split_indices

    def setup(self, stage: str | None = None) -> None:
        """
        Build only the datasets needed for the current stage.

        Split indices are computed once and cached. Individual datasets are created
        lazily, so validate/test-only workflows do not force train dataset creation.
        """
        self._log(f"Starting datamodule setup (stage={stage!r})")
        required_splits = self._required_splits(stage)
        split_indices = self._resolve_split_indices()
        bsccm_client: Any | None = None
        for split_name in required_splits:
            if self._datasets[split_name] is not None:
                continue
            selected_indices = split_indices.get(split_name, [])
            if not selected_indices:
                raise ValueError(f"{split_name} split is empty")
            if bsccm_client is None:
                bsccm_client = self._build_bsccm_client()
            dataset = BSCCM23to6Dataset(
                bsccm_client=bsccm_client,
                indices=selected_indices,
            )
            if self._dataset_root is not None and hasattr(dataset, "set_dataset_root"):
                dataset.set_dataset_root(str(self._dataset_root))
            self._datasets[split_name] = dataset
        self._log("Requested datasets initialized")

    def _resolve_val_dataset_for_viz(self) -> Any | None:
        self.setup("validate")
        dataset = self._datasets.get("val")
        if dataset is None:
            return None
        indices = getattr(dataset, "indices", None)
        if not isinstance(indices, list) or not indices:
            return None
        return dataset

    @staticmethod
    def _resolve_index_dataframe_for_viz(dataset: Any) -> Any | None:
        get_client = getattr(dataset, "_get_bsccm_client", None)
        if callable(get_client):
            bsccm_client = get_client()
        else:
            bsccm_client = getattr(dataset, "bsccm_client", None)

        index_dataframe = getattr(bsccm_client, "index_dataframe", None)
        columns = getattr(index_dataframe, "columns", None)
        if index_dataframe is None or columns is None or "antibodies" not in columns:
            return None
        return index_dataframe

    @staticmethod
    def _group_positions_by_antibody(
        *,
        indices: list[int],
        index_dataframe: Any,
    ) -> dict[str, list[int]]:
        grouped_positions: dict[str, list[int]] = defaultdict(list)
        for position, global_index in enumerate(indices):
            raw_label = index_dataframe.loc[int(global_index), "antibodies"]
            if hasattr(raw_label, "iloc"):
                raw_label = raw_label.iloc[0]
            label = normalize_antibody_label(raw_label)
            grouped_positions[label].append(position)
        return grouped_positions

    @staticmethod
    def _select_antibody_labels(
        *,
        grouped_positions: dict[str, list[int]],
        antibodies: list[str] | None,
    ) -> list[str]:
        if not antibodies:
            return sorted(grouped_positions)

        selected_labels: list[str] = []
        lower_to_label = {label.lower(): label for label in grouped_positions}
        for raw_value in antibodies:
            normalized = normalize_antibody_label(raw_value)
            if normalized in grouped_positions:
                selected_labels.append(normalized)
                continue
            resolved = lower_to_label.get(normalized.lower())
            if resolved is not None:
                selected_labels.append(resolved)
        return selected_labels

    @staticmethod
    def _collect_xy_samples_for_positions(
        *,
        dataset: Any,
        positions: list[int],
        sample_cap: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        x_samples: list[torch.Tensor] = []
        y_samples: list[torch.Tensor] = []
        for position in positions[:sample_cap]:
            sample = dataset[position]
            if not isinstance(sample, tuple) or len(sample) != 2:
                continue
            x_value, y_value = sample
            if not isinstance(x_value, torch.Tensor) or not isinstance(y_value, torch.Tensor):
                continue
            if x_value.ndim != 3 or y_value.ndim != 3:
                continue
            x_samples.append(x_value.detach().float().clamp(0.0, 1.0).cpu())
            y_samples.append(y_value.detach().float().clamp(0.0, 1.0).cpu())

        if not x_samples or not y_samples:
            return None
        return torch.stack(x_samples, dim=0), torch.stack(y_samples, dim=0)

    def build_antibody_viz_panel(
        self,
        *,
        antibodies: list[str] | None,
        samples_per_antibody: int,
    ) -> list[dict[str, Any]]:
        """
        Build a deterministic validation panel grouped by antibody labels.

        Returns entries shaped as:
        - `{"antibody": str, "x": Tensor[B, C_in, H, W], "y": Tensor[B, C_out, H, W]}`
        """
        dataset = self._resolve_val_dataset_for_viz()
        if dataset is None:
            return []
        indices = getattr(dataset, "indices")

        index_dataframe = self._resolve_index_dataframe_for_viz(dataset)
        if index_dataframe is None:
            return []
        grouped_positions = self._group_positions_by_antibody(
            indices=indices,
            index_dataframe=index_dataframe,
        )
        selected_labels = self._select_antibody_labels(
            grouped_positions=grouped_positions,
            antibodies=antibodies,
        )

        panel: list[dict[str, Any]] = []
        sample_cap = max(1, int(samples_per_antibody))
        for label in selected_labels:
            positions = grouped_positions.get(label, [])
            if not positions:
                continue

            sample_tensors = self._collect_xy_samples_for_positions(
                dataset=dataset,
                positions=positions,
                sample_cap=sample_cap,
            )
            if sample_tensors is None:
                continue
            x_batch, y_batch = sample_tensors

            panel.append(
                {
                    "antibody": label,
                    "x": x_batch,
                    "y": y_batch,
                }
            )

        return panel

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
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        return data_loader_cls(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=prefetch_factor,
            generator=generator,
            worker_init_fn=partial(_seed_worker, base_seed=self.seed),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> Any:
        """Return the training dataloader, initializing datasets on first call."""
        self.setup("fit")
        dataset = self._datasets["train"]
        if dataset is None:
            raise RuntimeError("train dataset is not initialized")
        return self._make_dataloader(dataset, shuffle=True)

    def val_dataloader(self) -> Any:
        """Return the validation dataloader, initializing datasets on first call."""
        self.setup("validate")
        dataset = self._datasets["val"]
        if dataset is None:
            raise RuntimeError("val dataset is not initialized")
        return self._make_dataloader(dataset, shuffle=False)

    def test_dataloader(self) -> Any:
        """Return the test dataloader, initializing datasets on first call."""
        self.setup("test")
        dataset = self._datasets["test"]
        if dataset is None:
            raise RuntimeError("test dataset is not initialized")
        return self._make_dataloader(dataset, shuffle=False)
