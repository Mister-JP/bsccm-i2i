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
        {
            "task_name": "i2i_23to6",
            "data": {
                "variant": "tiny",
                "root_dir": "data/bsccm_tiny",
                "num_workers": 0,
                "batch_size": 8,
                "pin_memory": False,
                "indices_csv": None,
            },
            "split": {
                "strategy": "random",
                "seed": 42,
                "train_frac": 0.8,
                "val_frac": 0.1,
                "test_frac": 0.1,
                "name": "random_80_10_10",
            },
            "run": {"run_name": "i2i_23to6_split", "tags": ["split"]},
        }
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
    assert fingerprint["variant"] == "tiny"
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
                "train_frac": 0.8,
                "val_frac": 0.1,
                "test_frac": 0.1,
                "variant": "tiny",
                "created_at": "2026-02-18T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    (split_dir / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "variant": "tiny",
                "bsccm_index_csv_sha256": "a",
                "bsccm_global_metadata_sha256": "b",
            }
        ),
        encoding="utf-8",
    )

    train_cfg = TrainConfig.model_validate(
        {
            "data": {
                "variant": "tiny",
                "root_dir": "data/bsccm_tiny",
                "num_workers": 0,
                "batch_size": 8,
                "pin_memory": False,
                "indices_csv": None,
            },
            "split": {
                "strategy": "random",
                "seed": 123,
                "train_frac": 0.8,
                "val_frac": 0.1,
                "test_frac": 0.1,
                "name": split_id,
            },
            "model": {
                "name": "unet",
                "in_channels": 23,
                "out_channels": 6,
                "base_channels": 32,
            },
            "trainer": {
                "max_epochs": 1,
                "device": "cpu",
                "precision": "32",
                "overfit_n": 0,
                "seed": 7,
                "deterministic": True,
                "max_steps": 0,
                "smoke": False,
            },
            "logging": {
                "tensorboard": True,
                "log_every_n_steps": 10,
                "image_log_every_n_steps": 100,
                "data_progress": False,
            },
            "run": {"run_name": "baseline_unet", "tags": ["baseline"]},
        }
    )

    split_metadata = load_split_metadata(split_id)
    with pytest.raises(ValueError, match="This run config doesn't match the split artifact"):
        validate_split_matches_config(split_metadata=split_metadata, train_cfg=train_cfg)
