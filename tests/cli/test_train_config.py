from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from bsccm_i2i.cli import main as cli_main
from bsccm_i2i.cli.main import app
from tests.config_builders import make_split_task_config, make_train_config, write_config


def test_train_delegates_to_runner_and_prints_run_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    captured: dict[str, object] = {}
    train_config_path = write_config(
        tmp_path / "train_config.yaml",
        make_train_config(
            split_name="train_split",
            overrides={"trainer": {"overfit_n": 2, "max_epochs": 1}},
        ),
    )

    expected_run_dir = Path("runs/2026-02-18/baseline_unet/2026-02-18_00-00-00")

    def _fake_run_train(train_cfg):
        captured["cfg"] = train_cfg.model_dump(mode="json")
        return expected_run_dir

    monkeypatch.setattr(cli_main, "run_train", _fake_run_train)

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(train_config_path),
        ],
    )

    assert result.exit_code == 0
    assert f"RUN_DIR {expected_run_dir}" in result.stdout
    assert "CALLED train" in result.stdout

    cfg = captured["cfg"]
    assert isinstance(cfg, dict)
    assert cfg["split"]["name"] == "train_split"
    assert cfg["trainer"]["overfit_n"] == 2
    assert cfg["trainer"]["max_epochs"] == 1


def test_split_command_prints_artifact_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    split_config_path = write_config(
        tmp_path / "split_config.yaml",
        make_split_task_config(),
    )
    monkeypatch.setattr(
        cli_main,
        "build_split_artifact",
        lambda _cfg: {
            "split_id": "abc",
            "artifact_dir": "artifacts/splits/abc",
            "counts": {"train": 80, "val": 10, "test": 10},
        },
    )
    runner = CliRunner()
    result = runner.invoke(app, ["split", "--config", str(split_config_path)])
    assert result.exit_code == 0
    assert "SPLIT_ID abc" in result.stdout
    assert "SPLIT_DIR artifacts/splits/abc" in result.stdout
    assert "SPLIT_COUNTS train=80 val=10 test=10" in result.stdout


def test_train_requires_explicit_split_artifact_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    train_config_path = write_config(
        tmp_path / "train_config.yaml",
        make_train_config(split_name="REQUIRED_SPLIT_ID"),
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(train_config_path),
        ],
    )

    assert result.exit_code != 0
    assert "explicit split artifact id" in result.output
    assert "split.name=<SPLIT_ID>" in result.output
