# Config Tree

Hydra config groups:
- `task/` command entrypoints (`train`, `i2i_23to6`)
- `experiment/` composed experiment recipes
- `data/`, `split/`, `model/`, `trainer/`, `logging/` component groups

Useful data flags:
- `logging.data_progress=true` prints datamodule progress (dataset root resolution, split sizes, dataloader settings).
