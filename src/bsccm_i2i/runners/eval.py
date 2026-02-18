"""Lightning eval runner that writes metrics into an existing run directory."""

from __future__ import annotations

from pathlib import Path

from bsccm_i2i.config.schema import EvalTaskConfig
from bsccm_i2i.models.registry import load_model_from_checkpoint
from bsccm_i2i.runners.artifacts import (
    load_eval_run_context,
    resolve_checkpoint_path,
    update_report_test_summary,
    write_test_metrics,
)
from bsccm_i2i.runners.common import (
    build_datamodule_from_train_config,
    configure_torch_determinism,
    extract_scalar_metrics,
    make_eval_trainer,
)


def run_eval(eval_task_cfg: EvalTaskConfig) -> Path:
    """Run test-set evaluation for an existing run and write metrics artifacts."""
    run_context = load_eval_run_context(Path(eval_task_cfg.eval.run_dir))

    checkpoint_path = resolve_checkpoint_path(
        run_dir=run_context.run_dir,
        checkpoint=eval_task_cfg.eval.checkpoint,
        report=run_context.report,
    )

    indices_csv = run_context.split_ref.indices_csv.strip()
    if not indices_csv:
        raise ValueError("split_ref.yaml must include a non-empty indices_csv")

    if run_context.train_cfg.trainer.deterministic:
        configure_torch_determinism(seed=run_context.train_cfg.trainer.seed)

    datamodule = build_datamodule_from_train_config(
        run_context.train_cfg,
        indices_csv=indices_csv,
    )
    model = load_model_from_checkpoint(run_context.train_cfg.model, str(checkpoint_path))
    trainer = make_eval_trainer(
        run_dir=run_context.run_dir,
        eval_cfg=eval_task_cfg.eval,
        deterministic=run_context.train_cfg.trainer.deterministic,
    )

    test_results = trainer.test(model=model, datamodule=datamodule)
    metrics = extract_scalar_metrics(dict(getattr(trainer, "callback_metrics", {})))
    if isinstance(test_results, list) and test_results and isinstance(test_results[0], dict):
        metrics.update(extract_scalar_metrics(test_results[0]))

    metrics_path = write_test_metrics(run_context.run_dir, metrics)
    update_report_test_summary(
        run_dir=run_context.run_dir,
        report=run_context.report,
        checkpoint_path=checkpoint_path,
    )
    return metrics_path
