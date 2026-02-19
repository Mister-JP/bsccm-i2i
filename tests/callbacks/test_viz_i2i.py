from __future__ import annotations

from types import SimpleNamespace

import torch

from bsccm_i2i.callbacks.viz_i2i import I2IVizCallback


class _FakeExperiment:
    def __init__(self) -> None:
        self.calls: list[tuple[str, torch.Tensor, int]] = []

    def add_image(self, tag: str, image: torch.Tensor, global_step: int) -> None:
        self.calls.append((tag, image, int(global_step)))


def _make_viz_cache() -> dict[str, torch.Tensor]:
    target = torch.rand(4, 6, 8, 8)
    prediction = torch.rand(4, 6, 8, 8)
    return {
        "x": torch.rand(4, 23, 8, 8),
        "y": target,
        "y_hat": prediction,
    }


def test_i2i_viz_callback_respects_image_step_cadence() -> None:
    callback = I2IVizCallback(num_viz_samples=2, image_log_every_n_steps=100)
    experiment = _FakeExperiment()
    trainer = SimpleNamespace(
        is_global_zero=True,
        sanity_checking=False,
        global_step=0,
        logger=SimpleNamespace(experiment=experiment),
    )
    module = SimpleNamespace(_viz_cache=_make_viz_cache())

    callback.on_validation_epoch_end(trainer, module)
    assert len(experiment.calls) == 3
    assert all(step == 0 for _, _, step in experiment.calls)

    trainer.global_step = 50
    callback.on_validation_epoch_end(trainer, module)
    assert len(experiment.calls) == 3

    trainer.global_step = 100
    callback.on_validation_epoch_end(trainer, module)
    assert len(experiment.calls) == 6
    assert all(step == 100 for _, _, step in experiment.calls[3:])


def test_i2i_viz_callback_skips_sanity_checking() -> None:
    callback = I2IVizCallback(num_viz_samples=2, image_log_every_n_steps=10)
    experiment = _FakeExperiment()
    trainer = SimpleNamespace(
        is_global_zero=True,
        sanity_checking=True,
        global_step=0,
        logger=SimpleNamespace(experiment=experiment),
    )
    module = SimpleNamespace(_viz_cache=_make_viz_cache())

    callback.on_validation_epoch_end(trainer, module)
    assert experiment.calls == []

    trainer.sanity_checking = False
    callback.on_validation_epoch_end(trainer, module)
    assert len(experiment.calls) == 3


def test_i2i_viz_callback_round_trips_callback_state() -> None:
    callback = I2IVizCallback(num_viz_samples=2, image_log_every_n_steps=25)
    callback._next_log_step = 125

    state = callback.state_dict()
    assert state == {"next_log_step": 125}

    restored = I2IVizCallback(num_viz_samples=1, image_log_every_n_steps=10)
    restored.load_state_dict(state)
    assert restored._next_log_step == 125
