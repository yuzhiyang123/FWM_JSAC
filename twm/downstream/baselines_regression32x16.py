from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from twm.downstream.channel_pred.training import ChannelPredTrainOutput, _nmse_loss
from twm.downstream.channel_pred_regression32x16 import PilotCrossAttentionViT32x16

class _HistoryToCurrentPredictor(nn.Module):
    def __init__(self, *, history_steps: int, num_tokens: int, token_dim: int) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.in_channels = self.history_steps * 2
        self.history_conv = nn.Conv2d(self.in_channels, 3, kernel_size=1, bias=True)
        self.input_mlp = nn.Sequential(
            nn.LayerNorm(self.in_channels),
            nn.Linear(self.in_channels, self.num_tokens * self.token_dim),
            nn.GELU(),
            nn.Linear(self.num_tokens * self.token_dim, self.num_tokens * self.token_dim),
        )
        self.decoder = PilotCrossAttentionViT32x16(token_dim=self.token_dim, pilot_channels=3, out_channels=2)

    def forward(self, history_channel: Tensor) -> Tensor:
        if history_channel.ndim != 5:
            raise ValueError(f'Expected history_channel [B, T, 2, H, W], got {tuple(history_channel.shape)}')
        batch_size, history_steps, complex_dim, height, width = history_channel.shape
        if complex_dim != 2:
            raise ValueError(f'Expected complex_dim=2, got {complex_dim}')
        if history_steps < self.history_steps:
            pad = history_channel.new_zeros(batch_size, self.history_steps - history_steps, complex_dim, height, width)
            history_channel = torch.cat([pad, history_channel], dim=1)
        elif history_steps > self.history_steps:
            history_channel = history_channel[:, -self.history_steps :]
        flattened = history_channel.reshape(batch_size, self.history_steps * complex_dim, height, width)
        repr3 = self.history_conv(flattened)
        pooled = flattened.mean(dim=(2, 3))
        tokens = self.input_mlp(pooled).view(batch_size, self.num_tokens, self.token_dim)
        return self.decoder(repr3, tokens)

@dataclass
class SimplePredictionConfig:
    num_tokens: int = 3
    token_dim: int = 192
    history_steps: int = 15

class SimplePredictionRegressionTask(nn.Module):
    def __init__(self, *, preprocessor: nn.Module, config: SimplePredictionConfig) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.config = config
        self.model = _HistoryToCurrentPredictor(
            history_steps=int(config.history_steps),
            num_tokens=int(config.num_tokens),
            token_dim=int(config.token_dim),
        )

    def preprocess_channels(self, x: Tensor) -> Tensor:
        if x.ndim not in {5, 6}:
            raise ValueError(f'Expected x shape [B, N, 2, H, W] or [B, T, N, 2, H, W], got {tuple(x.shape)}')
        flat = x.reshape(-1, *x.shape[-3:])
        preprocessed = self.preprocessor(flat)
        return preprocessed.view(*x.shape[:-3], *preprocessed.shape[1:])

    def preprocess_channel_pair(self, channel: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if channel.shape != target.shape:
            raise ValueError(f'Expected paired shapes to match, got {tuple(channel.shape)} vs {tuple(target.shape)}')
        pair = torch.stack([channel, target], dim=1)
        pair_preprocessed = self.preprocess_channels(pair)
        return pair_preprocessed[:, 0], pair_preprocessed[:, 1]

    def _prepare_history(self, history_channel: Tensor, history_mask: Tensor) -> Tensor:
        history_preprocessed = self.preprocess_channels(history_channel)
        mask = history_mask.to(dtype=history_preprocessed.dtype)[..., None, None, None]
        history_preprocessed = history_preprocessed * mask
        batch_size, steps, num_bands = history_preprocessed.shape[:3]
        return history_preprocessed.permute(0, 2, 1, 3, 4, 5).reshape(batch_size * num_bands, steps, *history_preprocessed.shape[-3:])

    def training_step(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor, *, current_gt: Tensor | None = None) -> ChannelPredTrainOutput:
        target_source = current_gt if current_gt is not None else current_channel
        _, target_preprocessed = self.preprocess_channel_pair(current_channel, target_source)
        history_input = self._prepare_history(history_channel, history_mask)
        predicted = self.model(history_input)
        flat_target = target_preprocessed.reshape(-1, *target_preprocessed.shape[-3:])
        x_loss = F.mse_loss(predicted, flat_target)
        recon_loss = _nmse_loss(predicted, flat_target)
        gt_recon_loss = recon_loss if current_gt is not None else None
        zero = predicted.new_tensor(0.0)
        return ChannelPredTrainOutput(
            loss=x_loss,
            x_loss=x_loss,
            recon_loss=recon_loss,
            practical_recon_loss=recon_loss,
            gt_recon_loss=gt_recon_loss,
            latent_target_power=zero,
            latent_nmse=zero,
            predicted_x0=predicted,
            recovered=predicted,
            predicted_latents=predicted.new_zeros(predicted.shape[0], 1, 1),
            pilot_latents=predicted.new_zeros(predicted.shape[0], 1, 1),
            calibrated_latents=predicted.new_zeros(predicted.shape[0], 1, 1),
            pilot_estimate=predicted.new_zeros(predicted.shape[0], 3, predicted.shape[-2], predicted.shape[-1]),
        )

    @torch.no_grad()
    def testing_step(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor, *, current_gt: Tensor | None = None) -> ChannelPredTrainOutput:
        return self.training_step(history_channel, history_mask, current_channel, current_gt=current_gt)

from twm.downstream.channel_pred_regression32x16 import ChannelPredRegressionTask

class DirectTrainingRegressionTask(ChannelPredRegressionTask):
    """End-to-end direct training baseline using the JEPA-shaped backbone without pretraining."""

    pass
