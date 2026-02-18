"""Split CSV I/O helpers."""

from __future__ import annotations

import csv
from pathlib import Path


def read_indices_csv(path: Path) -> dict[str, list[int]]:
    """
    Load sample indices from CSV.

    Supports a required index column (`global_index`) and optional `split`
    labels (`train|val|test`). Returns `all/train/val/test`.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        field_names = [name.strip() for name in (reader.fieldnames or []) if name is not None]
        if not field_names:
            raise ValueError(f"indices CSV must include a header row: {path}")

        if "global_index" not in field_names:
            raise ValueError(f"indices CSV must include global_index column: {path}")
        split_column = "split" if "split" in field_names else None

        rows: dict[str, list[int]] = {"all": [], "train": [], "val": [], "test": []}
        for row_number, row in enumerate(reader, start=2):
            raw_index = row.get("global_index", "").strip()
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


def write_indices_csv(
    path: Path,
    train: list[int],
    val: list[int],
    test: list[int],
) -> None:
    """Write one split CSV with `global_index,split` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["global_index", "split"])
        writer.writeheader()
        for value in train:
            writer.writerow({"global_index": int(value), "split": "train"})
        for value in val:
            writer.writerow({"global_index": int(value), "split": "val"})
        for value in test:
            writer.writerow({"global_index": int(value), "split": "test"})
