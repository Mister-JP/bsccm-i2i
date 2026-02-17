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

## Dependency Maintenance (Maintainers Only)

```bash
python -m pip install pip-tools
pip-compile pyproject.toml --generate-hashes -o requirements.lock
pip-compile pyproject.toml --extra dev --generate-hashes -o requirements-dev.lock
```

Run these only when dependencies change in `pyproject.toml`, then commit updated lock files.
