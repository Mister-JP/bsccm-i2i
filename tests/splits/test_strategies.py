from __future__ import annotations

import pytest

from bsccm_i2i.splits.strategies import stratified_antibodies_fraction_split


def test_stratified_antibodies_fraction_split_is_deterministic_and_balanced() -> None:
    indices = list(range(100))
    antibodies = ["A"] * 80 + ["B"] * 20

    train_1, val_1, test_1 = stratified_antibodies_fraction_split(
        indices=indices,
        antibodies=antibodies,
        train_frac=0.8,
        val_frac=0.1,
        seed=42,
    )
    train_2, val_2, test_2 = stratified_antibodies_fraction_split(
        indices=indices,
        antibodies=antibodies,
        train_frac=0.8,
        val_frac=0.1,
        seed=42,
    )

    assert train_1 == train_2
    assert val_1 == val_2
    assert test_1 == test_2
    assert len(train_1) == 80
    assert len(val_1) == 10
    assert len(test_1) == 10

    train_labels = [antibodies[idx] for idx in train_1]
    val_labels = [antibodies[idx] for idx in val_1]
    test_labels = [antibodies[idx] for idx in test_1]
    assert train_labels.count("A") == 64
    assert train_labels.count("B") == 16
    assert val_labels.count("A") == 8
    assert val_labels.count("B") == 2
    assert test_labels.count("A") == 8
    assert test_labels.count("B") == 2


def test_stratified_antibodies_fraction_split_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="matching lengths"):
        stratified_antibodies_fraction_split(
            indices=[1, 2, 3],
            antibodies=["A", "B"],
            train_frac=0.8,
            val_frac=0.1,
            seed=42,
        )
