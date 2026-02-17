# bsccm-i2i

Minimal bootstrap package for BSCCM image-to-image experiments.

Dependency model:
- `pyproject.toml` defines dependency intent.
- `requirements*.lock` pins exact versions and hashes for reproducible installs.

## Install (Reproducible, Recommended)

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install pip-tools
pip-sync requirements-dev.lock
bsccm-i2i --help
```

Use this path for normal setup. Do not run `pip-compile` unless you are updating dependencies.
If `requirements-dev.lock` is not present yet, use the editable install path below.

## Install (Editable Dev, Fast Iteration)

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
# Verifies installed console entrypoint wiring
bsccm-i2i --help
# Verifies direct module execution/import path
python -m bsccm_i2i.cli.main --help
```

## Build, Lint, and Test

#### Run from the repository root:

```bash
cd /Users/jignasupathak/Documents/job-search-2026-lol/bsccm-i2i
. .venv/bin/activate
```

#### Build a distributable package:

```bash
python -m pip install build
python -m build
```

#### Lint checks:

```bash
ruff check src tests
```

Lint rules are configured in `pyproject.toml` under `[tool.ruff.lint]` and currently include:
- `E`, `F` (pycodestyle/pyflakes basics)
- `I` (import sorting)
- `UP` (pyupgrade)
- `B` (bugbear)

If lint fails, first try auto-fixes:

```bash
ruff check src tests --fix
```

#### Run tests:

```bash
pytest -q
```

## Dependency Maintenance (Maintainers Only)

```bash
python -m pip install pip-tools
pip-compile pyproject.toml --generate-hashes -o requirements.lock
```

Run these only when dependencies change in `pyproject.toml`, then commit updated lock files.
