"""U-Net CNN baseline and Lightning-compatible training module."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from bsccm_i2i.metrics import Fluor6Metrics


class _DoubleConv(nn.Module):
    """Two 3x3 convolutions with ReLU activations and padding-preserved spatial shape."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetCNN(nn.Module):
    """Minimal U-Net mapping `[B, in_channels, H, W]` to `[B, out_channels, H, W]`."""

    def __init__(
        self,
        *,
        in_channels: int = 23,
        out_channels: int = 6,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.enc1 = _DoubleConv(in_channels, base_channels)
        self.enc2 = _DoubleConv(base_channels, base_channels * 2)
        self.bottleneck = _DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _resize_like(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.bottleneck(self.pool(x2))

        d2 = self.up2(x3)
        d2 = self._resize_like(d2, x2)
        d2 = self.dec2(torch.cat((d2, x2), dim=1))

        d1 = self.up1(d2)
        d1 = self._resize_like(d1, x1)
        d1 = self.dec1(torch.cat((d1, x1), dim=1))
        return self.out_conv(d1)


class UNetCNNModule(pl.LightningModule):
    """Lightning training wrapper with strict input/output and loss checks."""

    EXPECTED_INPUT_CHANNELS = 23
    EXPECTED_TARGET_CHANNELS = 6

    def __init__(
        self,
        *,
        in_channels: int = 23,
        out_channels: int = 6,
        base_channels: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = UNetCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
        self.loss_fn = nn.L1Loss()
        self.val_metrics = Fluor6Metrics()
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _validate_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        allowed_x_dtypes = (torch.float16, torch.float32, torch.float64)
        if x.dtype not in allowed_x_dtypes:
            raise TypeError(
                f"x dtype must be one of {allowed_x_dtypes}, got {x.dtype}"
            )
        if x.ndim != 4 or x.shape[1] != self.EXPECTED_INPUT_CHANNELS:
            raise ValueError(
                f"x must have shape [B, {self.EXPECTED_INPUT_CHANNELS}, H, W], got {tuple(x.shape)}"
            )
        if y.ndim != 4 or y.shape[1] != self.EXPECTED_TARGET_CHANNELS:
            raise ValueError(
                "y must have shape "
                f"[B, {self.EXPECTED_TARGET_CHANNELS}, H, W], got {tuple(y.shape)}"
            )

    def _forward_and_loss(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        self._validate_batch(x, y)
        y_hat = self(x)
        if y_hat.shape != y.shape:
            raise ValueError(
                f"y_hat shape {tuple(y_hat.shape)} does not match y shape {tuple(y.shape)}"
            )
        loss = self.loss_fn(y_hat, y)
        if not bool(torch.isfinite(loss).all().item()):
            raise RuntimeError("Non-finite loss encountered.")
        return y_hat, loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        _, loss = self._forward_and_loss(batch)
        self.log("loss/train", loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        x, y = batch
        y_hat, loss = self._forward_and_loss(batch)
        self.log("loss/val", loss, prog_bar=False, on_step=False, on_epoch=True)
        metrics = self.val_metrics.compute(y_hat, y)
        for key, value in metrics.items():
            self.log(
                f"metrics/val/{key}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
        self._viz_cache = {
            "x": x.detach(),
            "y": y.detach(),
            "y_hat": y_hat.detach(),
        }
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
