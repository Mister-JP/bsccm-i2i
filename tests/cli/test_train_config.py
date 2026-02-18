from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from typer.testing import CliRunner

from bsccm_i2i.cli import main as cli_main
from bsccm_i2i.cli.main import app


def _write_split_artifact(tmp_path: Path, split_id: str) -> Path:
    split_dir = tmp_path / "artifacts" / "splits" / split_id
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "indices.csv").write_text(
        "global_index,split\n0,train\n1,train\n2,val\n3,test\n",
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
    return split_dir


def test_train_creates_resolved_config_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_split_artifact(tmp_path=tmp_path, split_id="train_split")
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
    resolved_config = run_dir / "input_train_config.json"
    assert resolved_config.is_file()

    resolved = json.loads(resolved_config.read_text(encoding="utf-8"))
    assert isinstance(resolved, dict)
    assert resolved["split"]["name"] == "train_split"
    assert resolved["trainer"]["overfit_n"] == 2
    assert resolved["trainer"]["max_epochs"] == 1
    split_ref = run_dir / "split_ref.json"
    assert split_ref.is_file()
    split_ref_resolved = json.loads(split_ref.read_text(encoding="utf-8"))
    assert isinstance(split_ref_resolved, dict)
    assert split_ref_resolved["split_id"] == "train_split"
    assert split_ref_resolved["split_dir"] == "artifacts/splits/train_split"
    assert split_ref_resolved["counts"]["train"] == 2


def test_split_command_prints_artifact_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
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
    result = runner.invoke(app, ["split"])
    assert result.exit_code == 0
    assert "SPLIT_ID abc" in result.stdout
    assert "SPLIT_DIR artifacts/splits/abc" in result.stdout
    assert "SPLIT_COUNTS train=80 val=10 test=10" in result.stdout
