# Configuration Reference

This directory is the Hydra config tree for `bsccm-i2i`.

Runtime order:
1. Compose/load config.
2. Validate with Pydantic.
3. Consume in split/train/eval runners.

## 1) Loading Modes

### Hydra composition mode

Used when running overrides like:

```bash
bsccm-i2i train key=value ...
```

Hydra composes from `configs/task/*.yaml` and group defaults.

### Direct file mode

Used with:

```bash
bsccm-i2i train --config path/to/file.yaml
```

This bypasses Hydra composition and validates the file directly.
CLI enforces: use either `--config` or overrides, not both.

## 2) Command Entrypoints

| command | entry config | schema | runtime |
|---|---|---|---|
| `split` | `task/split.yaml` | `SplitTaskConfig` | `splits/builder.py` |
| `train` | `task/train.yaml` | `TrainConfig` | `runners/train.py` |
| `eval` | `task/eval.yaml` | `EvalTaskConfig` | `runners/eval.py` |

## 3) Composition Graph (Current Default Train)

`task/train.yaml` defaults to one experiment.

`experiment/baseline_unet.yaml` currently composes:
- `/data: bsccm_tiny`
- `/split: stratified`
- `/model: baseline_unet`
- `/trainer: default`
- `/logging: default`

Train config also defines a required split artifact selector:
- `split.id: REQUIRED_SPLIT_ID`

So train runs must set:
- `split.id=<SPLIT_ID>`

## 4) Config Groups and Semantics

### `data/*` fields

Fields:
- `dataset_variant`
- `root_dir`
- `num_workers`
- `batch_size`
- `pin_memory`

Usage:
- `split` uses `dataset_variant` and `root_dir`.
- `train` and `eval` use all fields above for dataloaders.
- Train/eval indices always come from split artifacts (`indices.csv` via `split_ref`), not from data config.

### `split/*` fields

Split-definition fields:
- `strategy`
- `seed`
- `subset_frac`
- `train_frac`
- `val_frac`
- `test_frac`

Train-only field:
- `id` (split artifact id selector)

Important:
- `split` command uses split-definition fields only and creates a new artifact id automatically.
- `train` command never creates splits; it requires an existing artifact id via `split.id`.
- Legacy `split.name` is still accepted on input for backward compatibility with older run artifacts.
- `strategy` is normalized in schema with `strip().lower()` and validated.
- Supported strategies: `random`, `stratified_antibodies`.
- Alias accepted: `stratified` -> `stratified_antibodies`.
- Fraction rules are enforced: `subset_frac in (0,1]` and `train+val+test == 1.0`.
- `test_frac` is part of the split contract and is validated/tracked with the artifact metadata.

### `model/*` fields

Fields:
- `name`
- `in_channels`
- `out_channels`
- `base_channels`
- `lr`
- `weight_decay`

Current runtime support:
- `name=unet_cnn`

### `trainer/*` fields

Fields:
- `max_epochs`
- `max_steps`
- `device`
- `precision`
- `overfit_n`
- `prefetch_factor`
- `seed`
- `deterministic`
- `limit_train_batches`
- `limit_val_batches`
- `enable_checkpointing`
- `logger`

Key behavior:
- `max_steps=0` means uncapped (`-1` to Lightning).
- `device` resolved from `auto/gpu/cuda/mps/cpu`.
- `precision` maps `"32"|"16"|"bf16"` to Lightning precision strings.
- `deterministic=true` seeds and configures deterministic torch behavior.

### `logging/*` fields

Fields:
- `tensorboard`
- `log_every_n_steps`
- `image_log_every_n_steps`
- `viz_antibodies`
- `viz_samples_per_antibody`
- `viz_log_target_once`
- `viz_log_error`
- `data_progress`

These control scalar/image logging and datamodule progress logs.

### `run` fields

Fields:
- `run_name`
- `tags`

Usage:
- `train` uses `run_name` for output path.
- `split` currently does not consume `run.*`.

### `eval` fields (`task/eval.yaml`)

Fields:
- `run_dir`
- `checkpoint`
- `device`
- `precision`
- `limit_test_batches`

Eval loads prior train artifacts from `run_dir` and reuses their train config + split reference.

## 5) Precedence

Order:
1. Composed YAML + CLI overrides.
2. Missing values filled by schema defaults.

YAML taking precedence over schema defaults is expected behavior.

## 6) Common Workflows

### Create split artifact

```bash
bsccm-i2i split \
  data=bsccm_full \
  split=stratified \
  split.subset_frac=0.01
```

### Train using existing split artifact

```bash
bsccm-i2i train \
  experiment=baseline_unet \
  split.id=<SPLIT_ID>
```

### Smoke train

```bash
bsccm-i2i train \
  experiment=baseline_unet \
  split.id=<SPLIT_ID> \
  trainer=smoke \
  data.num_workers=0
```

### Eval an existing run

```bash
bsccm-i2i eval \
  eval.run_dir=runs/<date>/<run_name>/<timestamp> \
  eval.checkpoint=best
```
