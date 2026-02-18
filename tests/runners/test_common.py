from __future__ import annotations

from types import SimpleNamespace

from bsccm_i2i.runners import common


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
