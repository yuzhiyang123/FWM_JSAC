from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from twm.downstream.channel_pred.training import (
    ChannelPredBatch,
    ChannelPredTask,
    ChannelPredTrainOutput,
    build_channel_pred_three_way_dataloaders,
    _latent_metric_triplet,
    _nmse_loss,
)
from twm.downstream.channel_pred_pointwise import (
    ChannelPredCDiT32x16Batch,
    build_channel_pred_cdit32x16_three_way_dataloaders,
)

import torch
from torch import Tensor, nn

class _RegressorBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=max(1, int(num_heads)), batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=max(1, int(num_heads)), batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

    def forward(self, x: Tensor, token_ctx: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x)
        x = x + self.cross_attn(h, token_ctx, token_ctx, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x

class PilotCrossAttentionViT32x16(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        pilot_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.pilot_channels = int(pilot_channels)
        self.out_channels = int(out_channels)
        self.height = 32
        self.width = 16
        self.hidden_dim = int(hidden_dim)
        self.pilot_embed = nn.Conv2d(self.pilot_channels, self.hidden_dim, kernel_size=1, bias=True)
        self.token_proj = nn.Linear(self.token_dim, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.height * self.width, self.hidden_dim))
        self.blocks = nn.ModuleList([
            _RegressorBlock(self.hidden_dim, num_heads=int(num_heads), mlp_ratio=float(mlp_ratio))
            for _ in range(max(1, int(depth)))
        ])
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.final = nn.Linear(self.hidden_dim, self.out_channels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, pilot_estimate: Tensor, tokens: Tensor) -> Tensor:
        if pilot_estimate.ndim != 4 or tuple(pilot_estimate.shape[1:]) != (self.pilot_channels, self.height, self.width):
            raise ValueError(f"Expected pilot_estimate [B, {self.pilot_channels}, {self.height}, {self.width}], got {tuple(pilot_estimate.shape)}")
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Expected tokens [B, Nt, {self.token_dim}], got {tuple(tokens.shape)}")
        x = self.pilot_embed(pilot_estimate).flatten(2).transpose(1, 2) + self.pos_embed
        token_ctx = self.token_proj(tokens)
        for block in self.blocks:
            x = block(x, token_ctx)
        out = self.final(self.final_norm(x)).transpose(1, 2).reshape(pilot_estimate.shape[0], self.out_channels, self.height, self.width)
        return out

class ChannelPredRegressionTask(ChannelPredTask):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.regressor = PilotCrossAttentionViT32x16(token_dim=self.feature_dim, pilot_channels=2, out_channels=2)

    def _regress_current_target(
        self,
        history_channel: Tensor,
        history_mask: Tensor,
        current_channel: Tensor,
        subband_mask: Tensor | None = None,
        current_gt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        target_source = current_gt if current_gt is not None else current_channel
        current_preprocessed, target_preprocessed = self.preprocess_channel_pair(current_channel, target_source)
        target_latents, _ = self.encode_channels(current_channel)
        predicted_latents = self._predict_current_latents(history_channel, history_mask, current_channel)
        _, latent_target_power, latent_nmse = _latent_metric_triplet(predicted_latents, target_latents)
        predicted_latents, pilot_latents, calibrated_latents, pilot_estimate, _ = self.compute_condition(
            history_channel,
            history_mask,
            current_channel,
            subband_mask=subband_mask,
            current_preprocessed=current_preprocessed,
            predicted_latents=predicted_latents,
        )
        flat_target = target_preprocessed.reshape(-1, *target_preprocessed.shape[-3:])
        flat_pilot_estimate = pilot_estimate.reshape(-1, *pilot_estimate.shape[-3:])[:, :2]
        condition_tokens = calibrated_latents.reshape(-1, 1, calibrated_latents.shape[-1])
        predicted = self.regressor(flat_pilot_estimate, condition_tokens)
        return predicted, flat_target, latent_target_power, latent_nmse, predicted_latents, pilot_latents, calibrated_latents, pilot_estimate

    def training_step(
        self,
        history_channel: Tensor,
        history_mask: Tensor,
        current_channel: Tensor,
        subband_mask: Tensor | None = None,
        current_gt: Tensor | None = None,
    ) -> ChannelPredTrainOutput:
        predicted, flat_target, latent_target_power, latent_nmse, predicted_latents, pilot_latents, calibrated_latents, pilot_estimate = self._regress_current_target(
            history_channel,
            history_mask,
            current_channel,
            subband_mask=subband_mask,
            current_gt=current_gt,
        )
        x_loss = F.mse_loss(predicted, flat_target)
        recon_loss = _nmse_loss(predicted, flat_target)
        gt_recon_loss = recon_loss if current_gt is not None else None
        return ChannelPredTrainOutput(
            loss=x_loss,
            x_loss=x_loss,
            recon_loss=recon_loss,
            practical_recon_loss=recon_loss,
            gt_recon_loss=gt_recon_loss,
            latent_target_power=latent_target_power,
            latent_nmse=latent_nmse,
            predicted_x0=predicted,
            recovered=predicted,
            predicted_latents=predicted_latents,
            pilot_latents=pilot_latents,
            calibrated_latents=calibrated_latents,
            pilot_estimate=pilot_estimate,
        )

    @torch.no_grad()
    def testing_step(
        self,
        history_channel: Tensor,
        history_mask: Tensor,
        current_channel: Tensor,
        subband_mask: Tensor | None = None,
        current_gt: Tensor | None = None,
    ) -> ChannelPredTrainOutput:
        return self.training_step(
            history_channel,
            history_mask,
            current_channel,
            subband_mask=subband_mask,
            current_gt=current_gt,
        )

def build_channel_pred_token_only_three_way_dataloaders(*args, **kwargs):
    return build_channel_pred_three_way_dataloaders(*args, **kwargs)

def build_channel_pred_token_only_pointwise_three_way_dataloaders(*args, **kwargs):
    return build_channel_pred_cdit32x16_three_way_dataloaders(*args, **kwargs)

def pointwise_batch_to_channel_pred_batch(batch: ChannelPredCDiT32x16Batch, device: torch.device) -> ChannelPredBatch:
    current_channel = batch.current_channel.to(device=device, dtype=torch.float32)
    subband_mask = batch.subband_mask.to(device=device, dtype=torch.bool)
    current_gt = None if batch.current_gt is None else batch.current_gt.to(device=device, dtype=torch.float32)
    batch_size, num_bands = current_channel.shape[:2]
    history_channel = torch.zeros(batch_size, 0, num_bands, current_channel.shape[2], current_channel.shape[3], current_channel.shape[4], device=device, dtype=torch.float32)
    history_mask = torch.zeros(batch_size, 0, num_bands, device=device, dtype=torch.bool)
    pilot_mask = current_channel.abs().sum(dim=2, keepdim=True) > 0
    return ChannelPredBatch(
        history_channel=history_channel,
        history_mask=history_mask,
        current_channel=current_channel,
        pilot_mask=pilot_mask,
        subband_mask=subband_mask,
        current_gt=current_gt,
    )

class ChannelPredTokenizerOnlyRegressionTask(ChannelPredRegressionTask):
    def _predict_current_latents(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor) -> Tensor:
        predicted_latents, _ = self.encode_channels(current_channel)
        return predicted_latents

class ChannelPredPilotOnlyRegressionTask(ChannelPredRegressionTask):
    def _regress_current_target(
        self,
        history_channel: Tensor,
        history_mask: Tensor,
        current_channel: Tensor,
        subband_mask: Tensor | None = None,
        current_gt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        target_source = current_gt if current_gt is not None else current_channel
        current_preprocessed, target_preprocessed = self.preprocess_channel_pair(current_channel, target_source)
        if subband_mask is None:
            subband_mask = torch.ones(current_channel.shape[:2], device=current_channel.device, dtype=torch.bool)
        effective_mask = subband_mask.to(dtype=torch.bool)
        pilot_estimate = self._build_pilot_estimate_from_preprocessed(current_preprocessed)
        pilot_estimate = pilot_estimate * effective_mask[..., None, None, None].to(dtype=pilot_estimate.dtype)
        batch_size, num_bands = current_channel.shape[:2]
        calibrated_latents = torch.zeros(batch_size, num_bands, self.feature_dim, device=current_channel.device, dtype=current_channel.dtype)
        predicted_latents = calibrated_latents
        pilot_latents = calibrated_latents
        flat_target = target_preprocessed.reshape(-1, *target_preprocessed.shape[-3:])
        flat_pilot_estimate = pilot_estimate.reshape(-1, *pilot_estimate.shape[-3:])[:, :2]
        condition_tokens = calibrated_latents.reshape(-1, 1, calibrated_latents.shape[-1])
        predicted = self.regressor(flat_pilot_estimate, condition_tokens)
        zero_metric = predicted.new_tensor(0.0)
        return predicted, flat_target, zero_metric, zero_metric, predicted_latents, pilot_latents, calibrated_latents, pilot_estimate
