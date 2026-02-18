from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from bsccm_i2i.cli import main as cli_main
from bsccm_i2i.cli.main import app
from tests.config_builders import make_eval_task_config, write_config


def test_eval_delegates_to_runner_and_prints_metrics_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    captured: dict[str, object] = {}
    run_dir = Path("runs/2026-02-18/baseline_unet/2026-02-18_00-00-00")
    eval_config_path = write_config(
        tmp_path / "eval_config.yaml",
        make_eval_task_config(run_dir=str(run_dir)),
    )

    expected_metrics_path = run_dir / "metrics" / "test_metrics.json"

    def _fake_run_eval(eval_task_cfg):
        captured["cfg"] = eval_task_cfg.model_dump(mode="json")
        return expected_metrics_path

    monkeypatch.setattr(cli_main, "run_eval", _fake_run_eval)

    result = runner.invoke(
        app,
        [
            "eval",
            "--config",
            str(eval_config_path),
        ],
    )

    assert result.exit_code == 0
    assert f"EVAL_RUN_DIR {run_dir}" in result.stdout
    assert f"EVAL_METRICS {expected_metrics_path}" in result.stdout

    cfg = captured["cfg"]
    assert isinstance(cfg, dict)
    assert cfg["task_name"] == "eval"
    assert cfg["eval"]["run_dir"] == str(run_dir)
    assert cfg["eval"]["checkpoint"] == "best"
