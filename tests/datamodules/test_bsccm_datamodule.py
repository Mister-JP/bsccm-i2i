from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bsccm_i2i.datamodules import bsccm_datamodule as datamodule_mod
from bsccm_i2i.splits.io import read_indices_csv
from bsccm_i2i.splits.strategies import random_fraction_split


def test_load_indices_csv_reads_all_and_split_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "indices.csv"
    csv_path.write_text(
        "global_index,split\n"
        "11,train\n"
        "12,val\n"
        "13,test\n"
        "14,train\n",
        encoding="utf-8",
    )

    rows = read_indices_csv(csv_path)
    assert rows["all"] == [11, 12, 13, 14]
    assert rows["train"] == [11, 14]
    assert rows["val"] == [12]
    assert rows["test"] == [13]


def test_split_indices_keeps_non_empty_train() -> None:
    train, val, test = random_fraction_split(
        indices=[1, 2, 3, 4, 5],
        train_frac=0.8,
        val_frac=0.1,
        seed=123,
    )

    assert train
    assert len(train) + len(val) + len(test) == 5


def test_get_dryad_token_from_env(monkeypatch) -> None:
    monkeypatch.setenv("BSCCM_DRYAD_TOKEN", "  abc123  ")
    assert datamodule_mod.get_dryad_token() == "abc123"


def test_resolve_dataset_root_passes_token_when_present(tmp_path: Path, monkeypatch) -> None:
    class _FakeBSCCMModule:
        def __init__(self, dataset_path: Path):
            self.dataset_path = dataset_path
            self.last_kwargs: dict[str, object] = {}

        def download_dataset(self, **kwargs):
            self.last_kwargs = kwargs
            return str(self.dataset_path)

    dataset_root = tmp_path / "downloaded"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "BSCCM_global_metadata.json").write_text("{}", encoding="utf-8")
    (dataset_root / "BSCCM_index.csv").write_text("global_index\n1\n", encoding="utf-8")
    (dataset_root / "BSCCM_images.zarr").mkdir(parents=True, exist_ok=True)

    fake_bsccm = _FakeBSCCMModule(dataset_path=dataset_root)
    monkeypatch.setattr(datamodule_mod, "bsccm", fake_bsccm)
    monkeypatch.setenv("BSCCM_DRYAD_TOKEN", "token-xyz")

    resolved = datamodule_mod.resolve_dataset_root(
        str(tmp_path / "missing_root"),
        dataset_variant="tiny",
    )
    assert resolved == dataset_root
    assert fake_bsccm.last_kwargs["token"] == "token-xyz"


def test_resolve_dataset_root_omits_token_when_missing(tmp_path: Path, monkeypatch) -> None:
    class _FakeBSCCMModule:
        def __init__(self, dataset_path: Path):
            self.dataset_path = dataset_path
            self.last_kwargs: dict[str, object] = {}

        def download_dataset(self, **kwargs):
            self.last_kwargs = kwargs
            return str(self.dataset_path)

    dataset_root = tmp_path / "downloaded_no_token"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "BSCCM_global_metadata.json").write_text("{}", encoding="utf-8")
    (dataset_root / "BSCCM_index.csv").write_text("global_index\n1\n", encoding="utf-8")
    (dataset_root / "BSCCM_images.zarr").mkdir(parents=True, exist_ok=True)

    fake_bsccm = _FakeBSCCMModule(dataset_path=dataset_root)
    monkeypatch.setattr(datamodule_mod, "bsccm", fake_bsccm)
    monkeypatch.setenv("BSCCM_DRYAD_TOKEN", "")

    resolved = datamodule_mod.resolve_dataset_root(
        str(tmp_path / "missing_root_2"), dataset_variant="tiny"
    )
    assert resolved == dataset_root
    assert "token" not in fake_bsccm.last_kwargs


def test_resolve_dataset_root_uses_nested_existing_root_without_download(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakeBSCCMModule:
        def download_dataset(self, **kwargs):
            raise AssertionError("download_dataset should not be called")

    nested_root = tmp_path / "bsccm_tiny" / "BSCCM-tiny"
    nested_root.mkdir(parents=True, exist_ok=True)
    (nested_root / "BSCCM_global_metadata.json").write_text("{}", encoding="utf-8")
    (nested_root / "BSCCM_index.csv").write_text("global_index\n1\n", encoding="utf-8")
    (nested_root / "BSCCM_images.zarr").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(datamodule_mod, "bsccm", _FakeBSCCMModule())

    resolved = datamodule_mod.resolve_dataset_root(
        str(tmp_path / "bsccm_tiny"),
        dataset_variant="tiny",
    )
    assert resolved == nested_root


def test_make_dataloader_uses_seeded_generator_for_shuffle(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeGenerator:
        def __init__(self) -> None:
            self.seed: int | None = None

        def manual_seed(self, seed: int) -> _FakeGenerator:
            self.seed = seed
            return self

    class _FakeDataLoader:
        def __init__(self, dataset, **kwargs) -> None:
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

    class _FakeTorch:
        Generator = _FakeGenerator
        utils = SimpleNamespace(data=SimpleNamespace(DataLoader=_FakeDataLoader))

        @staticmethod
        def manual_seed(seed: int) -> None:
            captured["torch_manual_seed"] = seed

    monkeypatch.setattr(datamodule_mod, "torch", _FakeTorch)
    datamodule = datamodule_mod.BSCCM23to6DataModule(
        root_dir="data/bsccm_tiny",
        dataset_variant="tiny",
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        seed=321,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
    )

    dataset = object()
    datamodule._make_dataloader(dataset, shuffle=True)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["shuffle"] is True
    assert kwargs["drop_last"] is False
    assert kwargs["generator"].seed == 321
    assert callable(kwargs["worker_init_fn"])

    datamodule._make_dataloader(dataset, shuffle=False)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["drop_last"] is False
    assert kwargs["generator"] is None


def test_setup_validate_only_initializes_val_dataset(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "indices.csv"
    csv_path.write_text(
        "global_index,split\n"
        "101,train\n"
        "102,val\n"
        "103,test\n",
        encoding="utf-8",
    )
    created_indices: list[list[int]] = []

    class _FakeDataset:
        def __init__(self, bsccm_client, indices: list[int]) -> None:
            del bsccm_client
            self.indices = list(indices)
            created_indices.append(self.indices)

        def __len__(self) -> int:
            return len(self.indices)

    monkeypatch.setattr(datamodule_mod, "BSCCM23to6Dataset", _FakeDataset)

    datamodule = datamodule_mod.BSCCM23to6DataModule(
        root_dir="data/bsccm_tiny",
        dataset_variant="tiny",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        seed=42,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        indices_csv=str(csv_path),
    )
    monkeypatch.setattr(datamodule, "_build_bsccm_client", lambda: object())

    datamodule.setup("validate")

    assert datamodule._datasets["train"] is None
    assert datamodule._datasets["val"] is not None
    assert datamodule._datasets["test"] is None
    assert created_indices == [[102]]


def test_setup_fit_initializes_train_and_val_only(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "indices.csv"
    csv_path.write_text(
        "global_index,split\n"
        "1,train\n"
        "2,train\n"
        "3,val\n",
        encoding="utf-8",
    )
    created_indices: list[list[int]] = []

    class _FakeDataset:
        def __init__(self, bsccm_client, indices: list[int]) -> None:
            del bsccm_client
            self.indices = list(indices)
            created_indices.append(self.indices)

        def __len__(self) -> int:
            return len(self.indices)

    monkeypatch.setattr(datamodule_mod, "BSCCM23to6Dataset", _FakeDataset)

    datamodule = datamodule_mod.BSCCM23to6DataModule(
        root_dir="data/bsccm_tiny",
        dataset_variant="tiny",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        seed=42,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        indices_csv=str(csv_path),
    )
    monkeypatch.setattr(datamodule, "_build_bsccm_client", lambda: object())

    datamodule.setup("fit")
    datamodule.setup("fit")

    assert datamodule._datasets["train"] is not None
    assert datamodule._datasets["val"] is not None
    assert datamodule._datasets["test"] is None
    assert created_indices == [[1, 2], [3]]


def test_setup_validate_fails_when_val_split_is_empty(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "indices.csv"
    csv_path.write_text(
        "global_index,split\n"
        "1,train\n"
        "2,train\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        datamodule_mod,
        "BSCCM23to6Dataset",
        lambda bsccm_client, indices: object(),
    )

    datamodule = datamodule_mod.BSCCM23to6DataModule(
        root_dir="data/bsccm_tiny",
        dataset_variant="tiny",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        seed=42,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        indices_csv=str(csv_path),
    )
    monkeypatch.setattr(datamodule, "_build_bsccm_client", lambda: object())

    with pytest.raises(ValueError, match="val split is empty"):
        datamodule.setup("validate")


def test_build_antibody_viz_panel_groups_and_filters() -> None:
    class _FakeIndexDataFrame:
        columns = ("antibodies",)

        def __init__(self, labels_by_index: dict[int, str]) -> None:
            self._labels_by_index = labels_by_index
            self.loc = self

        def __getitem__(self, key: tuple[int, str]) -> str:
            index_value, column_name = key
            if column_name != "antibodies":
                raise KeyError(column_name)
            return self._labels_by_index[int(index_value)]

    class _FakeDataset:
        def __init__(self) -> None:
            self.indices = [10, 11, 12, 13]
            self.bsccm_client = SimpleNamespace(
                index_dataframe=_FakeIndexDataFrame(
                    {
                        10: "CD45",
                        11: "CD45",
                        12: "CD123",
                        13: "CD123",
                    }
                )
            )

        def _get_bsccm_client(self) -> object:
            return self.bsccm_client

        def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
            global_index = self.indices[item]
            fill = float(global_index % 7) / 7.0
            x = torch.full((23, 8, 8), fill_value=fill, dtype=torch.float32)
            y = torch.full((6, 8, 8), fill_value=fill, dtype=torch.float32)
            return x, y

    datamodule = datamodule_mod.BSCCM23to6DataModule(
        root_dir="data/bsccm_tiny",
        dataset_variant="tiny",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        seed=42,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
    )
    datamodule._split_indices = {"train": [0], "val": [10, 11, 12, 13], "test": [1]}
    datamodule._datasets["val"] = _FakeDataset()

    panel = datamodule.build_antibody_viz_panel(
        antibodies=["cd45"],
        samples_per_antibody=1,
    )
    assert len(panel) == 1
    assert panel[0]["antibody"] == "CD45"
    assert isinstance(panel[0]["x"], torch.Tensor)
    assert isinstance(panel[0]["y"], torch.Tensor)
    assert tuple(panel[0]["x"].shape) == (1, 23, 8, 8)
    assert tuple(panel[0]["y"].shape) == (1, 6, 8, 8)


def test_build_antibody_viz_panel_returns_empty_without_antibody_column() -> None:
    class _NoAntibodyIndexDataFrame:
        columns = ("other",)

    class _FakeDataset:
        indices = [1]
        bsccm_client = SimpleNamespace(index_dataframe=_NoAntibodyIndexDataFrame())

    datamodule = datamodule_mod.BSCCM23to6DataModule(
        root_dir="data/bsccm_tiny",
        dataset_variant="tiny",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        seed=42,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
    )
    datamodule._split_indices = {"train": [0], "val": [1], "test": [2]}
    datamodule._datasets["val"] = _FakeDataset()

    panel = datamodule.build_antibody_viz_panel(
        antibodies=[],
        samples_per_antibody=1,
    )
    assert panel == []
