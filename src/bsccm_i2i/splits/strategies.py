"""Split strategy functions."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict


def _stable_label_seed(seed: int, label: str) -> int:
    """Build a deterministic per-label seed that is stable across Python runs."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big", signed=False)
    return int(seed) + offset


def _allocate_largest_remainder(
    *,
    group_sizes: dict[str, int],
    target_count: int,
    fraction: float,
    capacities: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Allocate integer counts across groups with largest-remainder rounding.

    This keeps totals exact (`sum == target_count`) while respecting per-group caps.
    """
    if target_count < 0:
        raise ValueError("target_count must be non-negative")

    caps = capacities or group_sizes
    allocated: dict[str, int] = {}
    remainders: dict[str, float] = {}

    for label, size in group_sizes.items():
        cap = int(caps.get(label, 0))
        if cap < 0:
            raise ValueError(f"capacity for group {label!r} must be non-negative")
        ideal = float(size) * float(fraction)
        base = min(cap, math.floor(ideal))
        allocated[label] = base
        remainders[label] = ideal - float(base)

    current = sum(allocated.values())
    if current > target_count:
        raise ValueError(
            "internal allocation error: base allocation exceeds target "
            f"({current} > {target_count})"
        )

    remaining = target_count - current
    order = sorted(
        group_sizes,
        key=lambda label: (-remainders[label], label),
    )
    while remaining > 0:
        progressed = False
        for label in order:
            cap = int(caps.get(label, 0))
            if allocated[label] >= cap:
                continue
            allocated[label] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError(
                "unable to allocate requested target_count with current per-group capacities"
            )

    return allocated


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


def stratified_antibodies_fraction_split(
    *,
    indices: list[int],
    antibodies: list[str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split indices by fractions while stratifying on antibody labels.

    Deterministically shuffles within each antibody group and uses largest-remainder
    allocation so global split counts closely track requested fractions.
    """
    if not indices:
        raise ValueError("no dataset indices available to split")
    if len(indices) != len(antibodies):
        raise ValueError("indices and antibodies must have matching lengths")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index_value, antibody in zip(indices, antibodies, strict=True):
        grouped[str(antibody)].append(int(index_value))

    for label, values in grouped.items():
        random.Random(_stable_label_seed(seed, label)).shuffle(values)

    total = len(indices)
    train_target = int(total * train_frac)
    val_target = int(total * val_frac)
    if train_target <= 0:
        raise ValueError("train split is empty; adjust split fractions or indices CSV")
    if train_target + val_target >= total:
        val_target = max(0, total - train_target - 1)

    group_sizes = {label: len(values) for label, values in grouped.items()}
    train_counts = _allocate_largest_remainder(
        group_sizes=group_sizes,
        target_count=train_target,
        fraction=train_frac,
    )
    remaining_capacities = {
        label: group_sizes[label] - train_counts[label] for label in group_sizes
    }
    val_counts = _allocate_largest_remainder(
        group_sizes=group_sizes,
        target_count=val_target,
        fraction=val_frac,
        capacities=remaining_capacities,
    )

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for label in sorted(grouped):
        values = grouped[label]
        train_count = train_counts[label]
        val_count = val_counts[label]
        train_indices.extend(values[:train_count])
        val_indices.extend(values[train_count : train_count + val_count])
        test_indices.extend(values[train_count + val_count :])

    if not train_indices:
        raise ValueError("train split is empty after split computation")
    return train_indices, val_indices, test_indices
