from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from bsccm_i2i.config.schema import TrainConfig
from bsccm_i2i.runners import common
from tests.config_builders import make_train_config


def test_resolve_accelerator_gpu_uses_all_visible_devices() -> None:
    assert common.resolve_accelerator("gpu") == ("gpu", "auto")


def test_resolve_accelerator_cuda_alias_uses_all_visible_devices() -> None:
    assert common.resolve_accelerator("cuda") == ("gpu", "auto")


def test_resolve_accelerator_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)

    assert common.resolve_accelerator("auto") == ("gpu", "auto")


def test_resolve_accelerator_auto_falls_back_to_mps(monkeypatch) -> None:
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        common.torch,
        "backends",
        SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )

    assert common.resolve_accelerator("auto") == ("mps", 1)


def test_resolve_accelerator_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(common.torch, "backends", SimpleNamespace())

    assert common.resolve_accelerator("auto") == ("cpu", 1)


def test_resolve_accelerator_unknown_defaults_to_cpu() -> None:
    assert common.resolve_accelerator("gu") == ("cpu", 1)


def test_make_train_trainer_passes_batch_limits(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeTrainer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(common.pl, "Trainer", _FakeTrainer)
    train_cfg = TrainConfig.model_validate(
        make_train_config(
            overrides={
                "trainer": {
                    "limit_train_batches": 0.25,
                    "limit_val_batches": 0.5,
                }
            }
        )
    )

    trainer = common.make_train_trainer(
        run_dir=tmp_path,
        train_cfg=train_cfg,
        callbacks=[],
        logger=False,
    )
    assert isinstance(trainer, _FakeTrainer)
    assert captured["limit_train_batches"] == 0.25
    assert captured["limit_val_batches"] == 0.5
