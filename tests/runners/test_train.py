from __future__ import annotations

import json
from pathlib import Path

import torch

import bsccm_i2i.runners.train as train_runner
from bsccm_i2i.config.schema import TrainConfig
from tests.config_builders import make_train_config


def _make_train_cfg(*, split_name: str = "split_abc") -> TrainConfig:
    return TrainConfig.model_validate(
        make_train_config(
            split_name=split_name,
            overrides={
                "model": {"base_channels": 16},
                "trainer": {"max_steps": 2},
                "logging": {"num_viz_samples": 2},
            },
        )
    )


def test_run_train_writes_standardized_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    split_dir = tmp_path / "artifacts" / "splits" / "split_abc"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "indices.csv").write_text("global_index,split\n0,train\n", encoding="utf-8")

    monkeypatch.setattr(
        train_runner,
        "resolve_split_dir",
        lambda _split_id: Path("artifacts/splits/split_abc"),
    )
    monkeypatch.setattr(
        train_runner,
        "load_split_metadata",
        lambda _split_id: {"fingerprint": {"dataset_variant": "tiny"}, "split": {}},
    )
    monkeypatch.setattr(
        train_runner,
        "load_split_indices",
        lambda _split_id: {"all": [0], "train": [0], "val": [], "test": []},
    )
    monkeypatch.setattr(train_runner, "validate_split_matches_config", lambda **_: None)
    monkeypatch.setattr(train_runner, "build_model", lambda _model_cfg: object())

    captured: dict[str, object] = {}

    class _FakeTensorBoardLogger:
        def __init__(self, **kwargs) -> None:
            captured["tb"] = kwargs

    class _FakeTrainer:
        def __init__(self) -> None:
            self.callback_metrics = {"loss/val": torch.tensor(0.25)}
            self.current_epoch = 0
            self.global_step = 2

        def fit(self, model, datamodule) -> None:
            captured["fit_model"] = model
            captured["fit_datamodule"] = datamodule

    def _fake_write_env_snapshot(run_dir: Path) -> None:
        env_dir = run_dir / "env"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "git_commit.txt").write_text("abc123\n", encoding="utf-8")
        (env_dir / "pip_freeze.txt").write_text("pkg==1.0.0\n", encoding="utf-8")
        (env_dir / "system.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        train_runner,
        "build_datamodule_from_train_config",
        lambda _train_cfg, indices_csv: captured.update({"indices_csv": indices_csv}) or object(),
    )
    monkeypatch.setattr(
        train_runner,
        "make_train_trainer",
        lambda **kwargs: captured.update({"trainer_kwargs": kwargs}) or _FakeTrainer(),
    )
    monkeypatch.setattr(train_runner, "TensorBoardLogger", _FakeTensorBoardLogger)
    monkeypatch.setattr(train_runner, "write_env_snapshot", _fake_write_env_snapshot)

    run_dir = train_runner.run_train(_make_train_cfg())

    assert run_dir.is_dir()
    assert (run_dir / "config_resolved.yaml").is_file()
    assert (run_dir / "split_ref.yaml").is_file()
    assert (run_dir / "metrics" / "epoch_metrics.csv").is_file()
    assert (run_dir / "report.json").is_file()
    assert captured["indices_csv"] == "artifacts/splits/split_abc/indices.csv"

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["split_id"] == "split_abc"
    assert report["best_metric"]["name"] == "loss/val"
    assert report["git_commit"] == "abc123"
    trainer_kwargs = captured["trainer_kwargs"]
    assert isinstance(trainer_kwargs, dict)
    callbacks = trainer_kwargs["callbacks"]
    assert isinstance(callbacks, list)
    viz_callbacks = [cb for cb in callbacks if isinstance(cb, train_runner.I2IVizCallback)]
    assert len(viz_callbacks) == 1
    viz_callback = viz_callbacks[0]
    assert viz_callback.num_viz_samples == 2
    assert viz_callback.image_log_every_n_steps == 100


def test_run_train_respects_logger_and_checkpoint_toggles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    split_dir = tmp_path / "artifacts" / "splits" / "split_abc"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "indices.csv").write_text("global_index,split\n0,train\n", encoding="utf-8")

    monkeypatch.setattr(
        train_runner,
        "resolve_split_dir",
        lambda _split_id: Path("artifacts/splits/split_abc"),
    )
    monkeypatch.setattr(
        train_runner,
        "load_split_metadata",
        lambda _split_id: {"fingerprint": {"dataset_variant": "tiny"}, "split": {}},
    )
    monkeypatch.setattr(
        train_runner,
        "load_split_indices",
        lambda _split_id: {"all": [0], "train": [0], "val": [], "test": []},
    )
    monkeypatch.setattr(train_runner, "validate_split_matches_config", lambda **_: None)
    monkeypatch.setattr(train_runner, "build_model", lambda _model_cfg: object())
    monkeypatch.setattr(
        train_runner,
        "build_datamodule_from_train_config",
        lambda _train_cfg, indices_csv: object(),
    )
    monkeypatch.setattr(
        train_runner,
        "write_env_snapshot",
        lambda run_dir: (run_dir / "env").mkdir(exist_ok=True),
    )

    captured: dict[str, object] = {}

    class _FakeTrainer:
        def __init__(self) -> None:
            self.callback_metrics = {}
            self.current_epoch = 0
            self.global_step = 0

        def fit(self, model, datamodule) -> None:
            captured["fit"] = (model, datamodule)

    monkeypatch.setattr(
        train_runner,
        "make_train_trainer",
        lambda **kwargs: captured.update({"kwargs": kwargs}) or _FakeTrainer(),
    )

    cfg = _make_train_cfg()
    cfg.trainer.logger = False
    cfg.trainer.enable_checkpointing = False
    run_dir = train_runner.run_train(cfg)

    assert run_dir.is_dir()
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["logger"] is False
    callbacks = kwargs["callbacks"]
    assert isinstance(callbacks, list)
    assert len(callbacks) == 1
