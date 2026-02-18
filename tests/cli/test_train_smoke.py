from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from bsccm_i2i.cli import main as cli_main
from tests.config_builders import make_train_config, write_config


def test_train_smoke_like_config_passes_limits_to_runner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    captured: dict[str, object] = {}
    config_path = write_config(
        tmp_path / "train_config.yaml",
        make_train_config(
            overrides={
                "trainer": {
                    "max_epochs": 1,
                    "max_steps": 2,
                    "limit_val_batches": 0,
                    "enable_checkpointing": False,
                    "logger": True,
                }
            }
        ),
    )

    def _fake_run_train(train_cfg):
        captured["cfg"] = train_cfg.model_dump(mode="json")
        return Path("runs/2026-02-18/baseline_unet/2026-02-18_00-00-00")

    monkeypatch.setattr(cli_main, "run_train", _fake_run_train)
    result = runner.invoke(
        cli_main.app,
        [
            "train",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    cfg = captured["cfg"]
    assert isinstance(cfg, dict)
    trainer_cfg = cfg["trainer"]
    assert trainer_cfg["max_epochs"] == 1
    assert trainer_cfg["max_steps"] == 2
    assert trainer_cfg["limit_val_batches"] == 0
    assert trainer_cfg["enable_checkpointing"] is False
    assert trainer_cfg["logger"] is True


def test_train_smoke_like_config_allows_custom_max_steps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    captured: dict[str, object] = {}
    config_path = write_config(
        tmp_path / "train_config.yaml",
        make_train_config(
            overrides={
                "trainer": {
                    "max_epochs": 1,
                    "max_steps": 5,
                    "limit_val_batches": 0,
                    "enable_checkpointing": False,
                    "logger": True,
                }
            }
        ),
    )

    def _fake_run_train(train_cfg):
        captured["cfg"] = train_cfg.model_dump(mode="json")
        return Path("runs/2026-02-18/baseline_unet/2026-02-18_00-00-00")

    monkeypatch.setattr(cli_main, "run_train", _fake_run_train)
    result = runner.invoke(
        cli_main.app,
        [
            "train",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    cfg = captured["cfg"]
    assert isinstance(cfg, dict)
    assert cfg["trainer"]["max_steps"] == 5
