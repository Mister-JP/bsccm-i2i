from __future__ import annotations

import json
from pathlib import Path

import pytest

from bsccm_i2i.config.schema import RunReportArtifact
from bsccm_i2i.runners import artifacts


def test_resolve_checkpoint_path_supports_best_last_and_explicit(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "2026-02-18" / "exp" / "2026-02-18_00-00-00"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoints_dir / "best.ckpt"
    last_ckpt = checkpoints_dir / "last.ckpt"
    explicit_ckpt = checkpoints_dir / "manual.ckpt"
    best_ckpt.write_text("best", encoding="utf-8")
    last_ckpt.write_text("last", encoding="utf-8")
    explicit_ckpt.write_text("manual", encoding="utf-8")

    report = RunReportArtifact.model_validate(
        {
            "split_id": "split_abc",
            "best_checkpoint_path": str(best_ckpt),
        }
    )

    assert (
        artifacts.resolve_checkpoint_path(run_dir=run_dir, checkpoint="best", report=report)
        == best_ckpt
    )
    assert (
        artifacts.resolve_checkpoint_path(run_dir=run_dir, checkpoint="last", report=report)
        == last_ckpt
    )
    assert (
        artifacts.resolve_checkpoint_path(
            run_dir=run_dir,
            checkpoint="checkpoints/manual.ckpt",
            report=report,
        )
        == explicit_ckpt
    )


def test_update_report_test_summary_persists_payload_and_extra_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "2026-02-18" / "exp" / "2026-02-18_00-00-00"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoints" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("best", encoding="utf-8")

    report = RunReportArtifact.model_validate(
        {
            "split_id": "split_abc",
            "best_metric": {"name": "loss/val", "value": 0.1},
            "notes": {"owner": "ci"},
        }
    )

    updated = artifacts.update_report_test_summary(
        run_dir=run_dir,
        report=report,
        checkpoint_path=checkpoint_path,
    )

    assert updated.test_summary is not None
    assert updated.test_summary.metrics_path == "metrics/test_metrics.json"

    written_report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert written_report["test_summary"]["checkpoint"] == str(checkpoint_path)
    assert written_report["notes"] == {"owner": "ci"}


def test_resolve_checkpoint_best_requires_report_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "2026-02-18" / "exp" / "2026-02-18_00-00-00"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = RunReportArtifact.model_validate({"split_id": "split_abc"})

    with pytest.raises(ValueError, match="best_checkpoint_path"):
        artifacts.resolve_checkpoint_path(run_dir=run_dir, checkpoint="best", report=report)
