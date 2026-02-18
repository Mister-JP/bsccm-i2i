"""Filesystem artifact and run-directory helpers."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from bsccm_i2i.callbacks.system_info import collect_system_info

_RUN_SUBDIRECTORIES = ("env", "checkpoints", "tensorboard", "metrics", "samples")


def create_run_dir(run_name: str) -> Path:
    """Create a dated run directory with a timestamp leaf and standard subdirectories."""
    normalized = run_name.strip()
    if not normalized:
        raise ValueError("run_name must be non-empty.")

    date_part = dt.date.today().isoformat()
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = Path("runs") / date_part / normalized
    run_dir = run_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = run_root / f"{timestamp}-{suffix}"
        suffix += 1

    for subdirectory in _RUN_SUBDIRECTORIES:
        (run_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, obj: Any) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(obj, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, obj: Any) -> None:
    """Write YAML if available; otherwise emit JSON (valid YAML 1.2 subset)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        path.write_text(
            f"{json.dumps(obj, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        return

    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def _run_text_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, check=False, text=True)
    except OSError as exc:
        return f"<error: {exc}>"

    if result.returncode != 0:
        error_detail = result.stderr.strip() or result.stdout.strip() or f"exit={result.returncode}"
        return f"<error: {error_detail}>"
    return result.stdout.strip()


def write_env_snapshot(run_dir: Path) -> None:
    """Write git, Python package, and hardware/software env metadata."""
    env_dir = run_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    git_commit = _run_text_command(["git", "rev-parse", "HEAD"])
    (env_dir / "git_commit.txt").write_text(f"{git_commit}\n", encoding="utf-8")

    pip_freeze = _run_text_command([sys.executable, "-m", "pip", "freeze"])
    (env_dir / "pip_freeze.txt").write_text(f"{pip_freeze}\n", encoding="utf-8")

    write_json(env_dir / "system.json", collect_system_info())
