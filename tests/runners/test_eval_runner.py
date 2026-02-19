from __future__ import annotations

import json
from pathlib import Path

import bsccm_i2i.runners.eval as eval_runner
from bsccm_i2i.config.schema import EvalTaskConfig
from tests.config_builders import make_eval_task_config, make_train_config


def test_run_eval_writes_metrics_and_updates_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "2026-02-18" / "baseline_unet" / "2026-02-18_00-00-00"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "best.ckpt").write_text("best", encoding="utf-8")
    (checkpoints_dir / "last.ckpt").write_text("last", encoding="utf-8")

    split_dir = tmp_path / "artifacts" / "splits" / "split_abc"
    split_dir.mkdir(parents=True, exist_ok=True)
    indices_csv = split_dir / "indices.csv"
    indices_csv.write_text("global_index,split\n0,test\n", encoding="utf-8")

    train_payload = make_train_config(
        split_id="split_abc",
        overrides={
            "data": {
                "num_workers": 0,
                "batch_size": 2,
                "pin_memory": False,
            },
            "trainer": {"device": "cpu", "precision": "32"},
        },
    )
    (run_dir / "config_resolved.yaml").write_text(json.dumps(train_payload), encoding="utf-8")
    (run_dir / "split_ref.yaml").write_text(
        json.dumps(
            {
                "split_id": "split_abc",
                "split_dir": str(split_dir),
                "indices_csv": str(indices_csv),
                "fingerprint": {"dataset_variant": "tiny"},
                "counts": {"train": 0, "val": 0, "test": 1},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "split_id": "split_abc",
                "best_metric": {"name": "loss/val", "value": 0.1},
                "best_checkpoint_path": str(checkpoints_dir / "best.ckpt"),
                "git_commit": "abc123",
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class _FakeTrainer:
        def __init__(self) -> None:
            self.callback_metrics = {"loss/test": 0.5, "metrics/test/mae": 0.25}

        def test(self, model, datamodule) -> list[dict[str, float]]:
            captured["test_model"] = model
            captured["test_datamodule"] = datamodule
            return [{"loss/test": 0.5, "metrics/test/mae": 0.25}]

    monkeypatch.setattr(
        eval_runner,
        "build_datamodule_from_train_config",
        lambda _train_cfg, indices_csv: captured.update({"indices_csv": indices_csv}) or object(),
    )
    monkeypatch.setattr(
        eval_runner,
        "make_eval_trainer",
        lambda **kwargs: captured.update({"trainer_kwargs": kwargs}) or _FakeTrainer(),
    )
    monkeypatch.setattr(
        eval_runner,
        "load_model_from_checkpoint",
        lambda _model_cfg, checkpoint_path: {"checkpoint": checkpoint_path},
    )

    eval_cfg = EvalTaskConfig.model_validate(
        make_eval_task_config(run_dir=str(run_dir), overrides={"eval": {"checkpoint": "best"}})
    )
    metrics_path = eval_runner.run_eval(eval_cfg)

    assert metrics_path == run_dir / "metrics" / "test_metrics.json"
    assert metrics_path.is_file()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["loss/test"] == 0.5
    assert metrics_payload["metrics/test/mae"] == 0.25
    assert isinstance(metrics_payload["loss/test"], float)

    report_payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report_payload["test_summary"]["checkpoint"] == str(checkpoints_dir / "best.ckpt")
    assert report_payload["test_summary"]["metrics_path"] == "metrics/test_metrics.json"

    assert captured["indices_csv"] == str(indices_csv)

    trainer_kwargs = captured["trainer_kwargs"]
    assert isinstance(trainer_kwargs, dict)
    assert trainer_kwargs["run_dir"] == run_dir
    assert trainer_kwargs["deterministic"] is True
