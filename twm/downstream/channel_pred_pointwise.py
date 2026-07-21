from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from twm.datasets.shard_access import TrackRef
from twm.downstream.channel_pred.data import load_channel_pred_track_pairs, split_channel_pred_tracks_train_intra_cross
from twm.downstream.channel_pred.training import _subband_mask_from_channel


@dataclass
class ChannelPredCDiT32x16Batch:
    current_channel: Tensor
    subband_mask: Tensor
    current_gt: Tensor | None = None


class _PointwiseChannelPredTrackDataset(Dataset):
    def __init__(self, track_refs: Sequence[TrackRef], *, input_residual_scale: float = 1.0) -> None:
        self.track_refs = list(track_refs)
        self.input_residual_scale = float(input_residual_scale)
        self.index: list[tuple[int, int]] = []
        for track_idx, ref in enumerate(self.track_refs):
            for point_idx in range(int(ref.shape[0])):
                self.index.append((track_idx, point_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        track_idx, point_idx = self.index[index]
        ref = self.track_refs[track_idx]
        channel = torch.tensor(np.asarray(ref.load_gt()[point_idx], dtype=np.float32))
        gt = torch.tensor(np.asarray(ref.load_gt()[point_idx], dtype=np.float32)) if ref.gt_key is not None else None
        if gt is not None and self.input_residual_scale != 1.0:
            channel = gt + self.input_residual_scale * (channel - gt)
        if channel.ndim != 4:
            raise ValueError(f'Expected channel point [N, 2, H, W], got {tuple(channel.shape)}')
        subband_mask = _subband_mask_from_channel(channel.unsqueeze(0)).squeeze(0)
        return {'current_channel': channel, 'subband_mask': subband_mask, 'current_gt': gt}


def _collate_batch(items: list[dict[str, Tensor]]) -> ChannelPredCDiT32x16Batch:
    if not items:
        raise ValueError('Cannot collate an empty batch.')
    batch_size = len(items)
    num_bands = int(items[0]['current_channel'].shape[0])
    complex_dim = int(items[0]['current_channel'].shape[1])
    height = int(items[0]['current_channel'].shape[2])
    width = int(items[0]['current_channel'].shape[3])
    current_channel = torch.zeros(batch_size, num_bands, complex_dim, height, width, dtype=torch.float32)
    subband_mask = torch.zeros(batch_size, num_bands, dtype=torch.bool)
    current_gt: Tensor | None = None
    if any(item['current_gt'] is not None for item in items):
        current_gt = torch.zeros(batch_size, num_bands, complex_dim, height, width, dtype=torch.float32)
    for idx, item in enumerate(items):
        current_channel[idx] = item['current_channel'].float()
        subband_mask[idx] = item['subband_mask'].bool()
        if current_gt is not None and item['current_gt'] is not None:
            current_gt[idx] = item['current_gt'].float()
    return ChannelPredCDiT32x16Batch(current_channel=current_channel, subband_mask=subband_mask, current_gt=current_gt)


def build_channel_pred_cdit32x16_three_way_dataloaders(
    dataset_file: str | Path | Sequence[str | Path],
    *,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    input_residual_scale: float = 1.0,
    train_paths: int = 512,
    test_paths: int = 88,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    track_refs, scenario_names = load_channel_pred_track_pairs(dataset_file=dataset_file, return_scenarios=True)
    train_refs, test_refs, cross_refs = split_channel_pred_tracks_train_intra_cross(
        tracks=track_refs,
        seed=seed,
        scenario_names=scenario_names,
        train_paths=train_paths,
        test_paths=test_paths,
    )
    return (
        DataLoader(_PointwiseChannelPredTrackDataset(train_refs, input_residual_scale=input_residual_scale), batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate_batch),
        DataLoader(_PointwiseChannelPredTrackDataset(test_refs, input_residual_scale=input_residual_scale), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_batch),
        DataLoader(_PointwiseChannelPredTrackDataset(cross_refs, input_residual_scale=input_residual_scale), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_batch),
    )
