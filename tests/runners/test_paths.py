from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from bsccm_i2i.runners.paths import create_run_dir, write_env_snapshot


def test_create_run_dir_and_write_env_snapshot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = create_run_dir("dry_run_test")

    assert run_dir.parent == Path("runs") / dt.date.today().isoformat() / "dry_run_test"
    assert run_dir.resolve().parent == (
        tmp_path / "runs" / dt.date.today().isoformat() / "dry_run_test"
    )
    dt.datetime.strptime(run_dir.name[:19], "%Y-%m-%d_%H-%M-%S")

    for subdirectory in ("env", "checkpoints", "tensorboard", "metrics", "samples"):
        assert (run_dir / subdirectory).is_dir()

    write_env_snapshot(run_dir)
    env_dir = run_dir / "env"

    assert (env_dir / "git_commit.txt").is_file()
    assert (env_dir / "pip_freeze.txt").is_file()
    system_json = env_dir / "system.json"
    assert system_json.is_file()

    system_info = json.loads(system_json.read_text(encoding="utf-8"))
    assert "os" in system_info
    assert "cpu" in system_info
    assert "ram" in system_info
    assert "gpu" in system_info


def test_create_run_dir_generates_unique_leaf_on_collision(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    first = create_run_dir("dry_run_test")
    second = create_run_dir("dry_run_test")
    assert first != second
    assert first.parent == second.parent
