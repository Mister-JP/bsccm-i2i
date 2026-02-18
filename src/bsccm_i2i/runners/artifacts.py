"""Run artifact IO and run-context loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bsccm_i2i.config.loader import load_config, to_resolved_dict
from bsccm_i2i.config.schema import (
    RunReportArtifact,
    SplitRefArtifact,
    SplitRefCounts,
    TestSummaryArtifact,
    TrainConfig,
)
from bsccm_i2i.runners.paths import write_json, write_yaml

CONFIG_RESOLVED_FILENAME = "config_resolved.yaml"
SPLIT_REF_FILENAME = "split_ref.yaml"
REPORT_FILENAME = "report.json"
TEST_METRICS_REL_PATH = Path("metrics") / "test_metrics.json"


@dataclass(frozen=True)
class EvalRunContext:
    """Validated inputs required for an eval run against an existing train run."""

    run_dir: Path
    train_cfg: TrainConfig
    split_ref: SplitRefArtifact
    report: RunReportArtifact


def ensure_run_dir_exists(run_dir: Path) -> None:
    """Ensure an existing run directory is provided."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required run artifact is missing: {path}")


def load_train_config_from_run(run_dir: Path) -> TrainConfig:
    """Load a resolved train config from an existing run directory."""
    config_path = run_dir / CONFIG_RESOLVED_FILENAME
    _require_file(config_path)
    train_hydra_cfg = load_config(config_path=config_path, config_name="task/train", overrides=[])
    return TrainConfig.model_validate(to_resolved_dict(train_hydra_cfg))


def load_split_ref_from_run(run_dir: Path) -> SplitRefArtifact:
    """Load split reference metadata from an existing run directory."""
    split_ref_path = run_dir / SPLIT_REF_FILENAME
    _require_file(split_ref_path)
    split_ref_hydra_cfg = load_config(
        config_path=split_ref_path,
        config_name="task/train",
        overrides=[],
    )
    return SplitRefArtifact.model_validate(to_resolved_dict(split_ref_hydra_cfg))


def load_report_from_run(run_dir: Path) -> RunReportArtifact:
    """Load run report json as typed artifact payload."""
    report_path = run_dir / REPORT_FILENAME
    _require_file(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return RunReportArtifact.model_validate(payload)


def load_eval_run_context(run_dir: Path) -> EvalRunContext:
    """Load and validate run artifacts required for eval."""
    ensure_run_dir_exists(run_dir)
    return EvalRunContext(
        run_dir=run_dir,
        train_cfg=load_train_config_from_run(run_dir),
        split_ref=load_split_ref_from_run(run_dir),
        report=load_report_from_run(run_dir),
    )


def resolve_checkpoint_path(
    *,
    run_dir: Path,
    checkpoint: str,
    report: RunReportArtifact,
) -> Path:
    """Resolve checkpoint selector (best/last/path) to a concrete checkpoint file."""
    normalized = checkpoint.strip().lower()
    if normalized == "best":
        best_path = report.best_checkpoint_path
        if not isinstance(best_path, str) or not best_path.strip():
            raise ValueError(
                "report.json does not contain best_checkpoint_path for checkpoint=best"
            )
        resolved = Path(best_path)
        if not resolved.is_absolute():
            resolved = run_dir / resolved
    elif normalized == "last":
        resolved = run_dir / "checkpoints" / "last.ckpt"
    else:
        resolved = Path(checkpoint)
        if not resolved.is_absolute():
            run_relative = run_dir / resolved
            resolved = run_relative if run_relative.exists() else resolved

    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {resolved}")
    return resolved


def write_test_metrics(run_dir: Path, metrics: dict[str, float]) -> Path:
    """Write eval metrics payload to the standard metrics artifact path."""
    metrics_path = run_dir / TEST_METRICS_REL_PATH
    write_json(metrics_path, metrics)
    return metrics_path


def write_report(run_dir: Path, report: RunReportArtifact) -> Path:
    """Write typed run report payload to report.json."""
    report_path = run_dir / REPORT_FILENAME
    write_json(report_path, report.model_dump(mode="json"))
    return report_path


def write_split_ref(
    *,
    run_dir: Path,
    split_id: str,
    split_dir: Path,
    indices_csv: str,
    fingerprint: dict[str, object],
    counts: SplitRefCounts,
) -> SplitRefArtifact:
    """Write standardized split reference artifact and return typed payload."""
    payload = SplitRefArtifact(
        split_id=split_id,
        split_dir=str(split_dir),
        indices_csv=indices_csv,
        fingerprint=fingerprint,
        counts=counts,
    )
    write_yaml(run_dir / SPLIT_REF_FILENAME, payload.model_dump(mode="json"))
    return payload


def update_report_test_summary(
    *,
    run_dir: Path,
    report: RunReportArtifact,
    checkpoint_path: Path,
) -> RunReportArtifact:
    """Attach eval output metadata to report payload and persist it."""
    updated = report.model_copy(
        update={
            "test_summary": TestSummaryArtifact(
                checkpoint=str(checkpoint_path),
                metrics_path=str(TEST_METRICS_REL_PATH),
            )
        }
    )
    write_report(run_dir, updated)
    return updated
