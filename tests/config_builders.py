from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _deep_update(current, value)
            continue
        target[key] = value


_BASE_DATA_CONFIG: dict[str, Any] = {
    "dataset_variant": "tiny",
    "root_dir": "data/bsccm_tiny",
    "num_workers": 0,
    "batch_size": 2,
    "pin_memory": False,
    "indices_csv": None,
}

_BASE_SPLIT_CONFIG: dict[str, Any] = {
    "strategy": "random",
    "seed": 42,
    "train_frac": 0.8,
    "val_frac": 0.1,
    "test_frac": 0.1,
    "name": "random_80_10_10",
}

_BASE_MODEL_CONFIG: dict[str, Any] = {
    "name": "unet_cnn",
    "in_channels": 23,
    "out_channels": 6,
    "base_channels": 32,
    "lr": 1e-3,
    "weight_decay": 0.0,
}

_BASE_TRAINER_CONFIG: dict[str, Any] = {
    "max_epochs": 1,
    "max_steps": 0,
    "device": "cpu",
    "precision": "32",
    "overfit_n": 0,
    "prefetch_factor": 4096,
    "seed": 123,
    "deterministic": True,
    "limit_val_batches": 1.0,
    "enable_checkpointing": True,
    "logger": True,
}

_BASE_LOGGING_CONFIG: dict[str, Any] = {
    "tensorboard": True,
    "log_every_n_steps": 10,
    "image_log_every_n_steps": 100,
    "viz_antibodies": [],
    "viz_samples_per_antibody": 2,
    "viz_log_target_once": True,
    "viz_log_error": True,
    "data_progress": False,
}

_BASE_EVAL_CONFIG: dict[str, Any] = {
    "run_dir": "runs/2026-02-18/baseline_unet/2026-02-18_00-00-00",
    "checkpoint": "best",
    "device": "cpu",
    "precision": "32",
    "limit_test_batches": 1.0,
}


def make_train_config(
    *,
    split_name: str = "random_80_10_10",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "task_name": "train",
        "data": deepcopy(_BASE_DATA_CONFIG),
        "split": deepcopy(_BASE_SPLIT_CONFIG),
        "model": deepcopy(_BASE_MODEL_CONFIG),
        "trainer": deepcopy(_BASE_TRAINER_CONFIG),
        "logging": deepcopy(_BASE_LOGGING_CONFIG),
        "run": {"run_name": "baseline_unet", "tags": ["baseline"]},
    }
    config["split"]["name"] = split_name
    if overrides:
        _deep_update(config, overrides)
    return config


def make_split_task_config(
    *,
    split_name: str = "unused",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "task_name": "split",
        "data": deepcopy(_BASE_DATA_CONFIG),
        "split": deepcopy(_BASE_SPLIT_CONFIG),
        "run": {"run_name": "split_artifact", "tags": ["split"]},
    }
    config["split"]["name"] = split_name
    if overrides:
        _deep_update(config, overrides)
    return config


def make_eval_task_config(
    *,
    run_dir: str = "runs/2026-02-18/baseline_unet/2026-02-18_00-00-00",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "task_name": "eval",
        "eval": deepcopy(_BASE_EVAL_CONFIG),
    }
    config["eval"]["run_dir"] = run_dir
    if overrides:
        _deep_update(config, overrides)
    return config


def write_config(path: Path, config: dict[str, Any]) -> Path:
    path.write_text(json.dumps(config), encoding="utf-8")
    return path
