from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from bsccm_i2i.datamodules import bsccm_datamodule as datamodule_mod


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

    rows = datamodule_mod.load_indices_csv(csv_path)
    assert rows["all"] == [11, 12, 13, 14]
    assert rows["train"] == [11, 14]
    assert rows["val"] == [12]
    assert rows["test"] == [13]


def test_split_indices_keeps_non_empty_train() -> None:
    train, val, test = datamodule_mod.split_indices(
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

    resolved = datamodule_mod.resolve_dataset_root(str(tmp_path / "missing_root"), variant="tiny")
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
        str(tmp_path / "missing_root_2"), variant="tiny"
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

    resolved = datamodule_mod.resolve_dataset_root(str(tmp_path / "bsccm_tiny"), variant="tiny")
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
        variant="tiny",
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
    assert kwargs["generator"].seed == 321
    assert callable(kwargs["worker_init_fn"])

    datamodule._make_dataloader(dataset, shuffle=False)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["generator"] is None
