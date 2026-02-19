"""Pydantic schemas for Hydra-composed configs."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

_SPLIT_STRATEGY_ALIASES = {
    "stratified": "stratified_antibodies",
}
_ALLOWED_SPLIT_STRATEGIES = frozenset({"random", "stratified_antibodies"})


class DataConfig(BaseModel):
    dataset_variant: str
    root_dir: str
    num_workers: int
    batch_size: int
    pin_memory: bool


class SplitDefinitionConfig(BaseModel):
    strategy: str
    seed: int
    subset_frac: float
    train_frac: float
    val_frac: float
    test_frac: float

    @field_validator("strategy", mode="before")
    @classmethod
    def normalize_strategy(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("split.strategy must be a string")
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("split.strategy must be non-empty")
        return _SPLIT_STRATEGY_ALIASES.get(normalized, normalized)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        if value not in _ALLOWED_SPLIT_STRATEGIES:
            allowed = ", ".join(sorted(_ALLOWED_SPLIT_STRATEGIES))
            raise ValueError(
                f"unsupported split.strategy={value!r}; expected one of: {allowed}"
            )
        return value

    @model_validator(mode="after")
    def validate_fractions_sum(self) -> SplitDefinitionConfig:
        if self.subset_frac <= 0.0 or self.subset_frac > 1.0:
            raise ValueError("subset_frac must be in (0.0, 1.0]")
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1.0")
        return self


class TrainSplitConfig(SplitDefinitionConfig):
    id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("id", "name"),
    )

    @field_validator("id", mode="before")
    @classmethod
    def normalize_split_id(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("split.id must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("split.id must be non-empty")
        return normalized


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
    prefetch_factor: int = 8
    seed: int
    deterministic: bool = True
    max_steps: int = 0
    # Throttle training [0.00, 1.00] work per run; lower for faster hyperparameter sweeps.
    limit_train_batches: float | int = 1.0
    # Throttle validation[0.00, 1.00] work per run; lower for faster debug/smoke iterations.
    limit_val_batches: float | int = 1.0
    enable_checkpointing: bool = True
    logger: bool = True


class LoggingConfig(BaseModel):
    tensorboard: bool = True
    log_every_n_steps: int = 50
    image_log_every_n_steps: int = 200
    viz_antibodies: list[str] = Field(default_factory=list)
    viz_samples_per_antibody: int = 2
    viz_log_target_once: bool = True
    viz_log_error: bool = True
    data_progress: bool = False


class RunConfig(BaseModel):
    run_name: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class TrainConfig(BaseModel):
    data: DataConfig
    split: TrainSplitConfig
    model: ModelConfig
    trainer: TrainerConfig
    logging: LoggingConfig
    run: RunConfig


class EvalConfig(BaseModel):
    run_dir: str
    checkpoint: str = "best"
    device: str = "auto"
    precision: str = "32"
    limit_test_batches: float | int = 1.0


class EvalTaskConfig(BaseModel):
    task_name: str
    eval: EvalConfig


class SplitRefCounts(BaseModel):
    train: int
    val: int
    test: int


class SplitRefArtifact(BaseModel):
    model_config = ConfigDict(extra="allow")
    split_id: str
    split_dir: str
    indices_csv: str
    fingerprint: dict[str, Any]
    counts: SplitRefCounts


class BestMetricArtifact(BaseModel):
    name: str
    value: float | None = None


class TestSummaryArtifact(BaseModel):
    checkpoint: str
    metrics_path: str


class RunReportArtifact(BaseModel):
    model_config = ConfigDict(extra="allow")
    split_id: str
    best_metric: BestMetricArtifact | None = None
    best_checkpoint_path: str | None = None
    git_commit: str | None = None
    test_summary: TestSummaryArtifact | None = None


class SplitTaskConfig(BaseModel):
    task_name: str
    data: DataConfig
    split: SplitDefinitionConfig
    run: RunConfig
