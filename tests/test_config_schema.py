from __future__ import annotations

import pytest

from bsccm_i2i.config.schema import SplitTaskConfig, TrainConfig
from tests.config_builders import make_split_task_config, make_train_config


def test_train_config_normalizes_strategy_and_supports_split_id() -> None:
    train_cfg = TrainConfig.model_validate(
        make_train_config(
            split_id="split_abc",
            overrides={"split": {"strategy": " StrATiFied "}},
        )
    )
    assert train_cfg.split.strategy == "stratified_antibodies"
    assert train_cfg.split.id == "split_abc"


def test_train_config_supports_legacy_split_name_field() -> None:
    payload = make_train_config(split_id="split_abc")
    payload["split"]["name"] = payload["split"].pop("id")
    train_cfg = TrainConfig.model_validate(payload)
    assert train_cfg.split.id == "split_abc"


def test_split_task_config_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unsupported split.strategy"):
        SplitTaskConfig.model_validate(
            make_split_task_config(overrides={"split": {"strategy": "random_v2"}})
        )
