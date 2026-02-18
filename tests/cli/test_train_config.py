from __future__ import annotations

import datetime as dt
from pathlib import Path

from omegaconf import OmegaConf
from typer.testing import CliRunner

from bsccm_i2i.cli.main import app


def test_train_creates_resolved_config_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "train",
            "experiment=baseline_unet",
            "split.name=train_split",
            "trainer.overfit_n=2",
            "trainer.max_epochs=1",
        ],
    )

    assert result.exit_code == 0
    run_dir_line = next(
        line for line in result.stdout.splitlines() if line.startswith("RUN_DIR ")
    )
    run_dir = tmp_path / run_dir_line.replace("RUN_DIR ", "", 1)
    assert run_dir.parent == (tmp_path / "runs" / dt.date.today().isoformat() / "baseline_unet")
    dt.datetime.strptime(run_dir.name[:19], "%Y-%m-%d_%H-%M-%S")
    resolved_config = run_dir / "config_resolved.yaml"
    assert resolved_config.is_file()

    cfg = OmegaConf.load(resolved_config)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    assert resolved["split"]["name"] == "train_split"
    assert resolved["trainer"]["overfit_n"] == 2
    assert resolved["trainer"]["max_epochs"] == 1


def test_split_composes_task_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["split", "split.name=custom_split"])
    assert result.exit_code == 0
    assert "CALLED split" in result.stdout
