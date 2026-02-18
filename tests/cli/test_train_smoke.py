from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from bsccm_i2i.cli import main as cli_main


def test_train_smoke_with_max_steps_invokes_loader(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    captured: dict[str, int] = {}

    def _fake_smoke_loader(train_cfg, max_steps: int) -> None:
        del train_cfg
        captured["max_steps"] = max_steps

    monkeypatch.setattr(cli_main, "run_train_smoke_loader", _fake_smoke_loader)
    result = runner.invoke(
        cli_main.app,
        [
            "train",
            "experiment=baseline_unet",
            "trainer.max_steps=2",
            "trainer.max_epochs=1",
        ],
    )

    assert result.exit_code == 0
    assert captured["max_steps"] == 2
    assert "CALLED train" in result.stdout


def test_train_smoke_true_defaults_to_two_steps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    captured: dict[str, int] = {}

    def _fake_smoke_loader(train_cfg, max_steps: int) -> None:
        del train_cfg
        captured["max_steps"] = max_steps

    monkeypatch.setattr(cli_main, "run_train_smoke_loader", _fake_smoke_loader)
    result = runner.invoke(
        cli_main.app,
        [
            "train",
            "experiment=baseline_unet",
            "trainer.smoke=true",
            "trainer.max_epochs=1",
        ],
    )

    assert result.exit_code == 0
    assert captured["max_steps"] == 2
    assert "CALLED train" in result.stdout


def test_train_smoke_uses_split_seed_for_datamodule(monkeypatch) -> None:
    captured: dict[str, int | bool] = {}

    class _TensorLike:
        def __init__(self, shape):
            self.shape = shape

    class _FakeDataModule:
        def __init__(self, **kwargs):
            captured["seed"] = int(kwargs["seed"])
            captured["log_progress"] = bool(kwargs["log_progress"])

        def train_dataloader(self):
            return [(_TensorLike((1, 23, 128, 128)), _TensorLike((1, 6, 128, 128)))]

    monkeypatch.setattr(cli_main, "BSCCM23to6DataModule", _FakeDataModule)
    monkeypatch.setattr(cli_main.typer, "echo", lambda _: None)

    train_cfg = cli_main.TrainConfig.model_validate(
        {
            "data": {
                "variant": "tiny",
                "root_dir": "data/bsccm_tiny",
                "num_workers": 0,
                "batch_size": 1,
                "pin_memory": False,
                "indices_csv": None,
            },
            "split": {
                "strategy": "random",
                "seed": 777,
                "train_frac": 0.8,
                "val_frac": 0.1,
                "test_frac": 0.1,
                "name": "random_80_10_10",
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
                "seed": 123,
                "deterministic": True,
                "max_steps": 0,
                "smoke": True,
            },
            "logging": {
                "tensorboard": True,
                "log_every_n_steps": 10,
                "image_log_every_n_steps": 100,
                "data_progress": True,
            },
            "run": {"run_name": "baseline_unet", "tags": ["baseline"]},
        }
    )

    cli_main.run_train_smoke_loader(train_cfg=train_cfg, max_steps=1)
    assert captured["seed"] == 777
    assert captured["log_progress"] is True


def test_train_smoke_deterministic_flag_controls_setup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    captured: dict[str, int | None] = {"seed": None, "steps": None}

    def _fake_configure(seed: int) -> None:
        captured["seed"] = seed

    def _fake_smoke_loader(train_cfg, max_steps: int) -> None:
        del train_cfg
        captured["steps"] = max_steps

    monkeypatch.setattr(cli_main, "configure_torch_determinism", _fake_configure)
    monkeypatch.setattr(cli_main, "run_train_smoke_loader", _fake_smoke_loader)

    result = runner.invoke(
        cli_main.app,
        [
            "train",
            "experiment=baseline_unet",
            "trainer.smoke=true",
            "trainer.seed=999",
            "trainer.max_epochs=1",
        ],
    )

    assert result.exit_code == 0
    assert captured["seed"] == 999
    assert captured["steps"] == 2


def test_train_smoke_deterministic_false_skips_setup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    captured: dict[str, bool] = {"called": False}

    def _fake_configure(seed: int) -> None:
        del seed
        captured["called"] = True

    monkeypatch.setattr(cli_main, "configure_torch_determinism", _fake_configure)
    monkeypatch.setattr(cli_main, "run_train_smoke_loader", lambda **_: None)

    result = runner.invoke(
        cli_main.app,
        [
            "train",
            "experiment=baseline_unet",
            "trainer.smoke=true",
            "trainer.deterministic=false",
            "trainer.max_epochs=1",
        ],
    )

    assert result.exit_code == 0
    assert captured["called"] is False


def test_train_deterministic_setup_runs_without_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    captured: dict[str, int | None] = {"seed": None}

    def _fake_configure(seed: int) -> None:
        captured["seed"] = seed

    monkeypatch.setattr(cli_main, "configure_torch_determinism", _fake_configure)
    result = runner.invoke(
        cli_main.app,
        [
            "train",
            "experiment=baseline_unet",
            "trainer.seed=2468",
            "trainer.max_epochs=1",
        ],
    )

    assert result.exit_code == 0
    assert captured["seed"] == 2468


def test_train_deterministic_false_skips_setup_without_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    captured: dict[str, bool] = {"called": False}

    def _fake_configure(seed: int) -> None:
        del seed
        captured["called"] = True

    monkeypatch.setattr(cli_main, "configure_torch_determinism", _fake_configure)
    result = runner.invoke(
        cli_main.app,
        [
            "train",
            "experiment=baseline_unet",
            "trainer.deterministic=false",
            "trainer.max_epochs=1",
        ],
    )

    assert result.exit_code == 0
    assert captured["called"] is False
