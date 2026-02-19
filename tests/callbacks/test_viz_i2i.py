from __future__ import annotations

from types import SimpleNamespace

import torch

from bsccm_i2i.callbacks.viz_i2i import I2IVizCallback


class _FakeExperiment:
    def __init__(self) -> None:
        self.calls: list[tuple[str, torch.Tensor, int]] = []

    def add_image(self, tag: str, image: torch.Tensor, global_step: int) -> None:
        self.calls.append((tag, image, int(global_step)))


class _FakeIndexDataFrame:
    columns = ("antibodies",)

    def __init__(self, labels_by_index: dict[int, str]) -> None:
        self._labels_by_index = labels_by_index
        self.loc = self

    def __getitem__(self, key: tuple[int, str]) -> str:
        index_value, column_name = key
        if column_name != "antibodies":
            raise KeyError(column_name)
        return self._labels_by_index[int(index_value)]


class _FakeDataset:
    def __init__(self, *, indices: list[int], labels_by_index: dict[int, str]) -> None:
        self.indices = list(indices)
        self.labels_by_index = dict(labels_by_index)
        self.bsccm_client = SimpleNamespace(
            index_dataframe=_FakeIndexDataFrame(labels_by_index)
        )

    def _get_bsccm_client(self) -> object:
        return self.bsccm_client

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        global_index = self.indices[item]
        fill = float(global_index % 7) / 7.0
        x = torch.full((23, 8, 8), fill_value=fill, dtype=torch.float32)
        y = torch.full((6, 8, 8), fill_value=min(1.0, fill + 0.2), dtype=torch.float32)
        return x, y


class _FakeDataModule:
    def __init__(self, dataset: _FakeDataset) -> None:
        self.dataset = dataset

    def build_antibody_viz_panel(
        self,
        *,
        antibodies: list[str] | None,
        samples_per_antibody: int,
    ) -> list[dict[str, object]]:
        grouped: dict[str, list[int]] = {}
        for global_index in self.dataset.indices:
            label = str(self.dataset.labels_by_index[int(global_index)])
            grouped.setdefault(label, []).append(int(global_index))

        if antibodies:
            selected: list[str] = []
            lower_to_label = {label.lower(): label for label in grouped}
            for value in antibodies:
                match = lower_to_label.get(str(value).lower())
                if match is not None:
                    selected.append(match)
        else:
            selected = sorted(grouped)

        panel: list[dict[str, object]] = []
        index_to_position = {
            int(global_index): position
            for position, global_index in enumerate(self.dataset.indices)
        }
        for label in selected:
            x_samples: list[torch.Tensor] = []
            y_samples: list[torch.Tensor] = []
            for global_index in grouped[label][: int(samples_per_antibody)]:
                position = index_to_position[int(global_index)]
                x_value, y_value = self.dataset[position]
                x_samples.append(x_value)
                y_samples.append(y_value)
            panel.append(
                {
                    "antibody": label,
                    "x": torch.stack(x_samples, dim=0),
                    "y": torch.stack(y_samples, dim=0),
                }
            )
        return panel


class _FakeModule:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return batch[:, :6].detach().float().clamp(0.0, 1.0)


def _make_trainer(
    *,
    experiment: _FakeExperiment,
    dataset: _FakeDataset,
    global_step: int,
    sanity_checking: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        is_global_zero=True,
        sanity_checking=sanity_checking,
        global_step=global_step,
        logger=SimpleNamespace(experiment=experiment),
        datamodule=_FakeDataModule(dataset),
    )


def test_i2i_viz_callback_logs_antibody_tags_and_respects_cadence() -> None:
    callback = I2IVizCallback(
        viz_samples_per_antibody=1,
        image_log_every_n_steps=100,
        viz_log_target_once=True,
        viz_log_error=True,
    )
    experiment = _FakeExperiment()
    dataset = _FakeDataset(
        indices=[10, 11, 12, 13],
        labels_by_index={10: "CD45", 11: "CD45", 12: "CD123", 13: "CD123"},
    )
    module = _FakeModule()

    trainer = _make_trainer(experiment=experiment, dataset=dataset, global_step=0)
    callback.on_validation_epoch_end(trainer, module)
    assert len(experiment.calls) == 6
    first_tags = [tag for tag, _, _ in experiment.calls]
    assert first_tags == [
        "cd123/target",
        "cd123/pred",
        "cd123/error_abs",
        "cd45/target",
        "cd45/pred",
        "cd45/error_abs",
    ]
    assert all(step == 0 for _, _, step in experiment.calls)

    trainer.global_step = 50
    callback.on_validation_epoch_end(trainer, module)
    assert len(experiment.calls) == 6

    trainer.global_step = 100
    callback.on_validation_epoch_end(trainer, module)
    assert len(experiment.calls) == 10
    second_tags = [tag for tag, _, _ in experiment.calls[6:]]
    assert second_tags == [
        "cd123/pred",
        "cd123/error_abs",
        "cd45/pred",
        "cd45/error_abs",
    ]
    assert all(step == 100 for _, _, step in experiment.calls[6:])


def test_i2i_viz_callback_respects_antibody_filter() -> None:
    callback = I2IVizCallback(
        viz_antibodies=["cd45"],
        viz_samples_per_antibody=1,
        image_log_every_n_steps=1,
        viz_log_target_once=False,
        viz_log_error=True,
    )
    experiment = _FakeExperiment()
    dataset = _FakeDataset(
        indices=[10, 11, 12, 13],
        labels_by_index={10: "CD45", 11: "CD45", 12: "CD123", 13: "CD123"},
    )
    trainer = _make_trainer(experiment=experiment, dataset=dataset, global_step=0)
    callback.on_validation_epoch_end(trainer, _FakeModule())

    tags = [tag for tag, _, _ in experiment.calls]
    assert tags == [
        "cd45/target",
        "cd45/pred",
        "cd45/error_abs",
    ]


def test_i2i_viz_callback_skips_sanity_checking() -> None:
    callback = I2IVizCallback(viz_samples_per_antibody=1, image_log_every_n_steps=10)
    experiment = _FakeExperiment()
    dataset = _FakeDataset(
        indices=[10, 11],
        labels_by_index={10: "CD45", 11: "CD45"},
    )
    trainer = _make_trainer(
        experiment=experiment, dataset=dataset, global_step=0, sanity_checking=True
    )
    module = _FakeModule()

    callback.on_validation_epoch_end(trainer, module)
    assert experiment.calls == []

    trainer.sanity_checking = False
    callback.on_validation_epoch_end(trainer, module)
    assert len(experiment.calls) == 3


def test_i2i_viz_callback_round_trips_callback_state() -> None:
    callback = I2IVizCallback(viz_samples_per_antibody=2, image_log_every_n_steps=25)
    callback._next_log_step = 125
    callback._target_logged_tags = {"cd45/target"}

    state = callback.state_dict()
    assert state == {
        "next_log_step": 125,
        "target_logged_tags": ["cd45/target"],
    }

    restored = I2IVizCallback(viz_samples_per_antibody=1, image_log_every_n_steps=10)
    restored.load_state_dict(state)
    assert restored._next_log_step == 125
    assert restored._target_logged_tags == {"cd45/target"}
