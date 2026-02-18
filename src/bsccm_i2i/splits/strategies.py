"""Split strategy functions."""

from __future__ import annotations

import random


def random_fraction_split(
    indices: list[int],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split a flat index list into train/val/test subsets with deterministic shuffling.

    Ensures train is non-empty and leaves the remainder for test.
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
