"""Pydantic schemas for Hydra-composed configs."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    variant: str
    root_dir: str
    num_workers: int
    batch_size: int
    pin_memory: bool
    indices_csv: str | None = None


class SplitConfig(BaseModel):
    strategy: str
    seed: int
    train_frac: float
    val_frac: float
    test_frac: float
    name: str

    @model_validator(mode="after")
    def validate_fractions_sum(self) -> SplitConfig:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1.0")
        return self


class ModelConfig(BaseModel):
    name: str
    in_channels: int = 23
    out_channels: int = 6
    base_channels: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0


class TrainerConfig(BaseModel):
    max_epochs: int
    device: str
    precision: str
    overfit_n: int
    seed: int
    deterministic: bool = True
    max_steps: int = 0
    smoke: bool = False


class LoggingConfig(BaseModel):
    tensorboard: bool = True
    log_every_n_steps: int = 50
    image_log_every_n_steps: int = 200
    num_viz_samples: int = 4
    data_progress: bool = False


class RunConfig(BaseModel):
    run_name: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class TrainConfig(BaseModel):
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    trainer: TrainerConfig
    logging: LoggingConfig
    run: RunConfig


class SplitTaskConfig(BaseModel):
    task_name: str
    data: DataConfig
    split: SplitConfig
    run: RunConfig
