"""BSCCM dataset implementation for 23-channel to 6-channel I2I."""

from __future__ import annotations

import importlib
from typing import Any

import torch

EXPECTED_INPUT_CHANNELS = 23
EXPECTED_TARGET_CHANNELS = 6
EXPECTED_SPATIAL_SHAPE = (128, 128)
NORMALIZATION_DENOMINATOR = 4095.0


class BSCCM23to6Dataset:
    """Dataset yielding `(x, y)` where x has 23 channels and y has 6 channels."""

    def __init__(self, bsccm_client: Any, indices: list[int]):
        """Create a dataset wrapper over BSCCM client for selected global indices."""
        self.bsccm_client = bsccm_client
        self.dataset_root: str | None = None
        self.indices = [int(value) for value in indices]
        if not self.indices:
            raise ValueError("dataset indices must be non-empty")

        self.input_channels = list(getattr(bsccm_client, "led_array_channel_names", []))
        self.target_channels = list(getattr(bsccm_client, "fluor_channel_names", []))

        if len(self.input_channels) != EXPECTED_INPUT_CHANNELS:
            raise ValueError(
                "expected "
                f"{EXPECTED_INPUT_CHANNELS} input channels, got {len(self.input_channels)}"
            )
        if len(self.target_channels) != EXPECTED_TARGET_CHANNELS:
            raise ValueError(
                "expected "
                f"{EXPECTED_TARGET_CHANNELS} target channels, got {len(self.target_channels)}"
            )

    def set_dataset_root(self, dataset_root: str) -> None:
        """Attach dataset root so workers can lazily rebuild BSCCM clients after pickling."""
        self.dataset_root = dataset_root

    def _get_bsccm_client(self) -> Any:
        """Return a live BSCCM client, lazily reopening from dataset root when needed."""
        if self.bsccm_client is not None:
            return self.bsccm_client
        if not self.dataset_root:
            raise RuntimeError(
                "bsccm_client is not available and dataset_root is unset; "
                "cannot lazily rebuild BSCCM client in worker process."
            )

        bsccm_module = importlib.import_module("bsccm")
        self.bsccm_client = bsccm_module.BSCCM(self.dataset_root)
        return self.bsccm_client

    def __getstate__(self) -> dict[str, Any]:
        """
        Drop live BSCCM client before multiprocessing pickling.

        Worker processes lazily restore their own client via `_get_bsccm_client`.
        """
        state = dict(self.__dict__)
        state["bsccm_client"] = None
        return state

    def __len__(self) -> int:
        """Return the number of samples available in this dataset split."""
        return len(self.indices)

    def _assert_finite(self, tensor: Any, kind: str, index: int) -> None:
        """Fail fast when a tensor contains NaN or Inf values."""
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{kind} tensor contains non-finite values at dataset index {index}")

    def _read_stack_and_normalize(
        self,
        *,
        index: int,
        channels: list[str],
        expected_channels: int,
        kind: str,
    ) -> Any:
        """
        Read channel images for one sample, enforce shape/channel gates, and normalize.

        The returned tensor is float32, channel-first, finite, and clamped to [0, 1]
        after dividing by 4095.0.
        """
        channel_tensors = []
        bsccm_client = self._get_bsccm_client()
        for channel in channels:
            image = bsccm_client.read_image(index, channel, copy=False)
            tensor = torch.as_tensor(image, dtype=torch.float32)
            if tuple(tensor.shape) != EXPECTED_SPATIAL_SHAPE:
                raise ValueError(
                    f"{kind} channel {channel!r} at index {index} has shape {tuple(tensor.shape)}; "
                    f"expected {EXPECTED_SPATIAL_SHAPE}"
                )
            channel_tensors.append(tensor)

        stacked = torch.stack(channel_tensors, dim=0)
        if tuple(stacked.shape) != (expected_channels, *EXPECTED_SPATIAL_SHAPE):
            raise ValueError(
                f"{kind} tensor has shape {tuple(stacked.shape)}; expected "
                f"({expected_channels}, {EXPECTED_SPATIAL_SHAPE[0]}, {EXPECTED_SPATIAL_SHAPE[1]})"
            )

        self._assert_finite(stacked, kind=kind, index=index)
        normalized = torch.clamp(stacked / NORMALIZATION_DENOMINATOR, 0.0, 1.0)
        self._assert_finite(normalized, kind=kind, index=index)
        return normalized

    def __getitem__(self, item: int) -> tuple[Any, Any]:
        """Return one `(x, y)` pair where `x` has 23 channels and `y` has 6 channels."""
        index = self.indices[item]
        x = self._read_stack_and_normalize(
            index=index,
            channels=self.input_channels,
            expected_channels=EXPECTED_INPUT_CHANNELS,
            kind="input",
        )
        y = self._read_stack_and_normalize(
            index=index,
            channels=self.target_channels,
            expected_channels=EXPECTED_TARGET_CHANNELS,
            kind="target",
        )
        return x, y
