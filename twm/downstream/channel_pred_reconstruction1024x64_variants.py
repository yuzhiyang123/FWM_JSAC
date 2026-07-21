from __future__ import annotations

import torch
from torch import Tensor

from twm.downstream.channel_pred.training import ChannelPredBatch, build_channel_pred_three_way_dataloaders
from twm.downstream.channel_pred_pointwise import (
    ChannelPredCDiT32x16Batch,
    build_channel_pred_cdit32x16_three_way_dataloaders,
)
from twm.downstream.channel_pred_reconstruction1024x64.training import ChannelPredReconstruction1024x64Task

def build_channel_pred_token_only_reconstruction1024x64_three_way_dataloaders(*args, **kwargs):
    return build_channel_pred_three_way_dataloaders(*args, **kwargs)

def build_channel_pred_token_only_reconstruction1024x64_pointwise_three_way_dataloaders(*args, **kwargs):
    return build_channel_pred_cdit32x16_three_way_dataloaders(*args, **kwargs)

def pointwise_batch_to_channel_pred_reconstruction_batch(batch: ChannelPredCDiT32x16Batch, device: torch.device) -> ChannelPredBatch:
    current_channel = batch.current_channel.to(device=device, dtype=torch.float32)
    subband_mask = batch.subband_mask.to(device=device, dtype=torch.bool)
    current_gt = None if batch.current_gt is None else batch.current_gt.to(device=device, dtype=torch.float32)
    batch_size, num_bands = current_channel.shape[:2]
    history_channel = torch.zeros(
        batch_size,
        0,
        num_bands,
        current_channel.shape[2],
        current_channel.shape[3],
        current_channel.shape[4],
        device=device,
        dtype=torch.float32,
    )
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

class ChannelPredReconstruction1024x64TokenizerOnlyTask(ChannelPredReconstruction1024x64Task):
    def _predict_current_latents(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor) -> Tensor:
        predicted_latents, _ = self.encode_channels(current_channel)
        return predicted_latents

class ChannelPredReconstruction1024x64ZeroConditionTask(ChannelPredReconstruction1024x64Task):
    def _predict_current_latents(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor) -> Tensor:
        del history_channel, history_mask
        batch_size, num_bands = current_channel.shape[:2]
        return current_channel.new_zeros(batch_size, num_bands, self.feature_dim)
