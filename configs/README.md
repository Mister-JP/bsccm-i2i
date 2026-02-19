# Config Tree

This folder is the Hydra config tree for the CLI.
At runtime, Hydra composes YAML files into one config object, then Pydantic validates the result.

## How the CLI uses these configs

`split` command:
- Entry file: `task/split.yaml`
- Validated as: `SplitTaskConfig`
- Purpose: create and register a split artifact (`indices.csv`, split metadata, dataset fingerprint)

`train` command:
- Entry file: `task/train.yaml`
- Validated as: `TrainConfig`
- Purpose: run Lightning training from a split artifact and write standardized run artifacts

Note: `train` intentionally does not create splits. You must provide an explicit split artifact id via `split.name=<SPLIT_ID>`.

## How composition works for `train`

`task/train.yaml` points to one experiment:
- `/experiment: baseline_unet`

`experiment/baseline_unet.yaml` then pulls in:
- `/data: bsccm_tiny`
- `/split: random_80_10_10`
- `/model: baseline_unet`
- `/trainer: default`
- `/logging: default`

So the final resolved config has these top-level sections:
- `data`
- `split`
- `model`
- `trainer`
- `logging`
- `run`

## Config groups in this repo

- `task/`: `split.yaml`, `train.yaml`
- `experiment/`: `baseline_unet.yaml`
- `data/`: `bsccm_tiny.yaml`, `bsccm_full.yaml`
- `split/`: `random_80_10_10.yaml`, `stratified_antibodies_80_10_10.yaml`
- `model/`: `baseline_unet.yaml`, `unet_cnn.yaml`
- `trainer/`: `default.yaml`, `smoke.yaml`
- `logging/`: `default.yaml`

## What each section controls

`data`:
- Where the dataset is read from (`root_dir`, `dataset_variant`)
- DataLoader behavior (`batch_size`, `num_workers`, `pin_memory`)
- Optional indices CSV field (`indices_csv`) at schema/datamodule level; in the current `train` CLI flow it is populated from the selected split artifact (`artifacts/splits/<split_id>/indices.csv`)

`split`:
- Split strategy and seed
- Optional pre-split dataset downsampling fraction (`subset_frac`)
- Train/val/test fractions
- Artifact id reference (`name`) used later by `train`; default is a placeholder to force explicit selection

Constraint:
- `subset_frac` must be in `(0.0, 1.0]`
- `train_frac + val_frac + test_frac` must equal `1.0`

`model`:
- Model selection (`name`)
- Channel contract (`in_channels`, `out_channels`)
- Capacity (`base_channels`)
- Optimizer hyperparameters (`lr`, `weight_decay`)

`trainer`:
- Runtime controls (`max_epochs`, `max_steps`, `device`, `precision`)
- Reproducibility (`seed`, `deterministic`)
- Trainer limits and toggles (`limit_train_batches`, `limit_val_batches`, `enable_checkpointing`, `logger`)

`logging`:
- TensorBoard enablement
- Scalar logging cadence (`log_every_n_steps`)
- Image logging cadence (`image_log_every_n_steps`)
- Optional explicit antibody subset for image panels (`viz_antibodies`)
- Number of samples per antibody in image visualizations (`viz_samples_per_antibody`)
- Whether targets are logged once per antibody tag (`viz_log_target_once`)
- Whether absolute error grids are logged (`viz_log_error`)
- Extra split/datamodule progress logs (`data_progress`)

`run`:
- Run naming (`run_name`)
- Metadata labels (`tags`)

## Defaults and precedence

- Values in YAML files are composed first.
- Any missing values are filled by schema defaults.
- If both exist, YAML value wins.

Current examples where YAML overrides schema defaults:
- `logging.log_every_n_steps`: schema default `50`, YAML sets `10`
- `logging.image_log_every_n_steps`: schema default `200`, YAML sets `100`
- `logging.data_progress`: schema default `false`, YAML sets `true`

## Common override patterns

- Turn on progress logs: `logging.data_progress=true`
- Run quick smoke steps through preset: `trainer=smoke`
- Change batch size for a run: `data.batch_size=16`
- Use antibody-stratified fractions: `split=stratified_antibodies_80_10_10`

## Expected workflow (explicit split selection)

1. Build a split artifact:
- `bsccm-i2i split`
2. Copy the printed `SPLIT_ID`.
3. Run train with that id:
- `bsccm-i2i train experiment=baseline_unet split.name=<SPLIT_ID> ...`

This is intentional so training never silently re-splits data.
