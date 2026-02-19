from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bsccm_i2i.config.schema import SplitTaskConfig, TrainConfig
from bsccm_i2i.splits import builder as builder_mod
from bsccm_i2i.splits.registry import (
    load_split_indices,
    load_split_metadata,
    validate_split_matches_config,
)
from tests.config_builders import make_split_task_config, make_train_config


def _write_fake_dataset_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "BSCCM_index.csv").write_text("global_index\n0\n1\n", encoding="utf-8")
    (path / "BSCCM_global_metadata.json").write_text('{"v":1}\n', encoding="utf-8")
    (path / "BSCCM_images.zarr").mkdir(parents=True, exist_ok=True)


def test_build_split_artifact_writes_expected_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_root = tmp_path / "fake_dataset"
    _write_fake_dataset_root(dataset_root)

    class _FakeBackend:
        def get_indices(self, shuffle: bool = False):
            assert shuffle is False
            return list(range(100))

    class _FakeBSCCMModule:
        __version__ = "0.0-test"

        @staticmethod
        def BSCCM(_dataset_root: str) -> _FakeBackend:
            return _FakeBackend()

    monkeypatch.setattr(builder_mod, "resolve_dataset_root", lambda *_args, **_kwargs: dataset_root)
    monkeypatch.setattr(builder_mod, "bsccm", _FakeBSCCMModule)

    task_cfg = SplitTaskConfig.model_validate(
        make_split_task_config(
            overrides={"data": {"batch_size": 8}},
        )
    )
    summary = builder_mod.build_split_artifact(task_cfg)

    artifact_dir = tmp_path / summary["artifact_dir"]
    assert artifact_dir.is_dir()
    assert (artifact_dir / "indices.csv").is_file()
    assert (artifact_dir / "split.json").is_file()
    assert (artifact_dir / "dataset_fingerprint.json").is_file()
    assert (artifact_dir / "input_split_config.json").is_file()

    indices_text = (artifact_dir / "indices.csv").read_text(encoding="utf-8").splitlines()
    assert indices_text[0] == "global_index,split"
    assert summary["counts"] == {"train": 80, "val": 10, "test": 10}

    fingerprint = json.loads(
        (artifact_dir / "dataset_fingerprint.json").read_text(encoding="utf-8")
    )
    expected_index_hash = hashlib.sha256(
        (dataset_root / "BSCCM_index.csv").read_bytes()
    ).hexdigest()
    expected_meta_hash = hashlib.sha256(
        (dataset_root / "BSCCM_global_metadata.json").read_bytes()
    ).hexdigest()
    assert fingerprint["dataset_variant"] == "tiny"
    assert fingerprint["bsccm_package_version"] == "0.0-test"
    assert fingerprint["bsccm_index_csv_sha256"] == expected_index_hash
    assert fingerprint["bsccm_global_metadata_sha256"] == expected_meta_hash

    loaded_indices = load_split_indices(summary["split_id"])
    assert loaded_indices["train"]
    assert len(loaded_indices["train"]) == 80
    assert len(loaded_indices["val"]) == 10
    assert len(loaded_indices["test"]) == 10


def test_registry_mismatch_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    split_id = "split_seed_42"
    split_dir = tmp_path / "artifacts" / "splits" / split_id
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "indices.csv").write_text(
        "global_index,split\n0,train\n1,val\n2,test\n",
        encoding="utf-8",
    )
    (split_dir / "split.json").write_text(
        json.dumps(
            {
                "split_id": split_id,
                "strategy": "random",
                "seed": 42,
                "subset_frac": 1.0,
                "train_frac": 0.8,
                "val_frac": 0.1,
                "test_frac": 0.1,
                "dataset_variant": "tiny",
                "created_at": "2026-02-18T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    (split_dir / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "dataset_variant": "tiny",
                "bsccm_index_csv_sha256": "a",
                "bsccm_global_metadata_sha256": "b",
            }
        ),
        encoding="utf-8",
    )

    train_cfg = TrainConfig.model_validate(
        make_train_config(
            split_id=split_id,
            overrides={
                "data": {"batch_size": 8},
                "split": {"seed": 123},
                "model": {"name": "unet"},
                "trainer": {"seed": 7, "max_steps": 0},
            },
        )
    )

    split_metadata = load_split_metadata(split_id)
    with pytest.raises(ValueError, match="This run config doesn't match the split artifact"):
        validate_split_matches_config(split_metadata=split_metadata, train_cfg=train_cfg)


def test_registry_mismatch_raises_when_subset_frac_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    split_id = "split_seed_42_missing_subset"
    split_dir = tmp_path / "artifacts" / "splits" / split_id
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "indices.csv").write_text(
        "global_index,split\n0,train\n1,val\n2,test\n",
        encoding="utf-8",
    )
    (split_dir / "split.json").write_text(
        json.dumps(
            {
                "split_id": split_id,
                "strategy": "random",
                "seed": 42,
                "train_frac": 0.8,
                "val_frac": 0.1,
                "test_frac": 0.1,
                "dataset_variant": "tiny",
                "created_at": "2026-02-18T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    (split_dir / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "dataset_variant": "tiny",
                "bsccm_index_csv_sha256": "a",
                "bsccm_global_metadata_sha256": "b",
            }
        ),
        encoding="utf-8",
    )

    train_cfg = TrainConfig.model_validate(
        make_train_config(
            split_id=split_id,
            overrides={
                "data": {"batch_size": 8},
                "split": {"seed": 42, "subset_frac": 1.0},
                "model": {"name": "unet"},
                "trainer": {"seed": 7, "max_steps": 0},
            },
        )
    )

    split_metadata = load_split_metadata(split_id)
    with pytest.raises(ValueError, match="subset_frac"):
        validate_split_matches_config(split_metadata=split_metadata, train_cfg=train_cfg)


def test_build_split_artifact_supports_stratified_antibodies(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_root = tmp_path / "fake_dataset"
    _write_fake_dataset_root(dataset_root)

    labels_by_index = {idx: ("A" if idx < 80 else "B") for idx in range(100)}

    class _FakeIndexDataFrame:
        columns = ("antibodies",)

        def __init__(self, labels: dict[int, str]) -> None:
            self._labels = labels
            self.loc = self

        def __getitem__(self, key: tuple[int, str]) -> str:
            index_value, column_name = key
            if column_name != "antibodies":
                raise KeyError(column_name)
            return self._labels[int(index_value)]

    class _FakeBackend:
        index_dataframe = _FakeIndexDataFrame(labels_by_index)

        def get_indices(self, shuffle: bool = False):
            assert shuffle is False
            return list(range(100))

    class _FakeBSCCMModule:
        __version__ = "0.0-test"

        @staticmethod
        def BSCCM(_dataset_root: str) -> _FakeBackend:
            return _FakeBackend()

    monkeypatch.setattr(builder_mod, "resolve_dataset_root", lambda *_args, **_kwargs: dataset_root)
    monkeypatch.setattr(builder_mod, "bsccm", _FakeBSCCMModule)

    task_cfg = SplitTaskConfig.model_validate(
        make_split_task_config(
            overrides={
                "split": {
                    "strategy": "stratified_antibodies",
                    "seed": 42,
                    "train_frac": 0.8,
                    "val_frac": 0.1,
                    "test_frac": 0.1,
                }
            },
        )
    )
    summary = builder_mod.build_split_artifact(task_cfg)
    loaded = load_split_indices(summary["split_id"])
    assert summary["counts"] == {"train": 80, "val": 10, "test": 10}

    train_labels = [labels_by_index[idx] for idx in loaded["train"]]
    val_labels = [labels_by_index[idx] for idx in loaded["val"]]
    test_labels = [labels_by_index[idx] for idx in loaded["test"]]

    assert train_labels.count("A") == 64
    assert train_labels.count("B") == 16
    assert val_labels.count("A") == 8
    assert val_labels.count("B") == 2
    assert test_labels.count("A") == 8
    assert test_labels.count("B") == 2

    split_json = json.loads(
        (Path(summary["artifact_dir"]) / "split.json").read_text(encoding="utf-8")
    )
    assert split_json["strategy"] == "stratified_antibodies"


def test_build_split_artifact_applies_subset_frac_before_stratified_split(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_root = tmp_path / "fake_dataset"
    _write_fake_dataset_root(dataset_root)

    labels_by_index = {idx: ("A" if idx < 80 else "B") for idx in range(100)}

    class _FakeIndexDataFrame:
        columns = ("antibodies",)

        def __init__(self, labels: dict[int, str]) -> None:
            self._labels = labels
            self.loc = self

        def __getitem__(self, key: tuple[int, str]) -> str:
            index_value, column_name = key
            if column_name != "antibodies":
                raise KeyError(column_name)
            return self._labels[int(index_value)]

    class _FakeBackend:
        index_dataframe = _FakeIndexDataFrame(labels_by_index)

        def get_indices(self, shuffle: bool = False):
            assert shuffle is False
            return list(range(100))

    class _FakeBSCCMModule:
        __version__ = "0.0-test"

        @staticmethod
        def BSCCM(_dataset_root: str) -> _FakeBackend:
            return _FakeBackend()

    monkeypatch.setattr(builder_mod, "resolve_dataset_root", lambda *_args, **_kwargs: dataset_root)
    monkeypatch.setattr(builder_mod, "bsccm", _FakeBSCCMModule)

    task_cfg = SplitTaskConfig.model_validate(
        make_split_task_config(
            overrides={
                "split": {
                    "strategy": "stratified_antibodies",
                    "seed": 42,
                    "subset_frac": 0.5,
                    "train_frac": 0.8,
                    "val_frac": 0.1,
                    "test_frac": 0.1,
                }
            },
        )
    )
    summary = builder_mod.build_split_artifact(task_cfg)
    loaded = load_split_indices(summary["split_id"])
    assert summary["counts"] == {"train": 40, "val": 5, "test": 5}

    train_labels = [labels_by_index[idx] for idx in loaded["train"]]
    val_labels = [labels_by_index[idx] for idx in loaded["val"]]
    test_labels = [labels_by_index[idx] for idx in loaded["test"]]

    assert train_labels.count("A") == 32
    assert train_labels.count("B") == 8
    assert val_labels.count("A") == 4
    assert val_labels.count("B") == 1
    assert test_labels.count("A") == 4
    assert test_labels.count("B") == 1

    split_json = json.loads(
        (Path(summary["artifact_dir"]) / "split.json").read_text(encoding="utf-8")
    )
    assert split_json["subset_frac"] == 0.5
