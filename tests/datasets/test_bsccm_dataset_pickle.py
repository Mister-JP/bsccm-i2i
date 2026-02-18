from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np

from bsccm_i2i.datasets import bsccm_dataset as dataset_mod


def _make_fake_client() -> object:
    class _Client:
        led_array_channel_names = [f"led_{i}" for i in range(23)]
        fluor_channel_names = [f"fluor_{i}" for i in range(6)]

        def __init__(self) -> None:
            self.unpicklable = lambda x: x

        def read_image(self, _index: int, _channel: str, copy: bool = False):
            del copy
            return np.zeros((128, 128), dtype=np.float32)

    return _Client()


def test_dataset_pickles_without_live_bsccm_client() -> None:
    dataset = dataset_mod.BSCCM23to6Dataset(_make_fake_client(), [0])
    dataset.set_dataset_root("/tmp/bsccm")

    payload = pickle.dumps(dataset)
    restored = pickle.loads(payload)

    assert restored.bsccm_client is None
    assert restored.dataset_root == "/tmp/bsccm"
    assert len(restored.indices) == 1


def test_dataset_lazily_rebuilds_client_after_unpickle(monkeypatch) -> None:
    opened_paths: list[str] = []

    class _RebuiltClient:
        led_array_channel_names = [f"led_{i}" for i in range(23)]
        fluor_channel_names = [f"fluor_{i}" for i in range(6)]

        def read_image(self, _index: int, _channel: str, copy: bool = False):
            del copy
            return np.ones((128, 128), dtype=np.float32)

    fake_module = SimpleNamespace(
        BSCCM=lambda path: opened_paths.append(path) or _RebuiltClient()
    )
    monkeypatch.setattr(dataset_mod.importlib, "import_module", lambda _name: fake_module)

    dataset = dataset_mod.BSCCM23to6Dataset(_make_fake_client(), [0])
    dataset.set_dataset_root("/tmp/bsccm")
    restored = pickle.loads(pickle.dumps(dataset))

    x, y = restored[0]

    assert opened_paths == ["/tmp/bsccm"]
    assert tuple(x.shape) == (23, 128, 128)
    assert tuple(y.shape) == (6, 128, 128)
