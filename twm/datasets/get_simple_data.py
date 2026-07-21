from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .get_sequential_data import load_tracks, split_tracks_train_intra_cross, split_tracks_train_test
from .shard_access import TrackRef, load_track_refs


def load_track_pairs(
    dataset_file: str | Path | Sequence[str | Path],
    *,
    return_scenarios: bool = False,
):
    track_refs, scenario_names = load_track_refs(dataset_file)
    paired_refs = [ref for ref in track_refs if ref.gt_key is not None or ref.raw_key is not None]
    if not paired_refs:
        raise ValueError('Dataset file contains no usable tracks with gt data.')
    if return_scenarios:
        paired_scenarios = [ref.scenario_name for ref in paired_refs]
        return paired_refs, paired_scenarios
    return paired_refs


def split_track_pairs_train_test(
    track_pairs: list[TrackRef],
    test_ratio: float = 0.2,
    seed: int = 42,
    *,
    scenario_names: Sequence[str] | None = None,
    train_paths: int = 512,
    test_paths: int = 88,
) -> tuple[list[TrackRef], list[TrackRef]]:
    return split_tracks_train_test(
        tracks=track_pairs,
        test_ratio=test_ratio,
        seed=seed,
        scenario_names=scenario_names,
        train_paths=train_paths,
        test_paths=test_paths,
    )


class _SimpleFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        track_refs: Sequence[TrackRef],
        *,
        n1: int,
        band: str,
        kind: str = 'gt',
        sample_transform: Callable | None = None,
    ):
        if band not in {'3p5', '28'}:
            raise ValueError(f'Unknown band: {band}')
        self.track_refs = list(track_refs)
        self.n1 = int(n1)
        self.band = band
        self.kind = kind
        self.sample_transform = sample_transform
        self.index: list[tuple[int, int, int]] = []
        for track_idx, track_ref in enumerate(self.track_refs):
            t_len, n_bands = int(track_ref.shape[0]), int(track_ref.shape[1])
            if self.n1 <= 0 or self.n1 >= n_bands:
                raise ValueError(f'n1 must be in (0, N), got n1={self.n1}, N={n_bands}')
            band_offset = 0 if self.band == '3p5' else self.n1
            band_count = self.n1 if self.band == '3p5' else (n_bands - self.n1)
            for t_idx in range(t_len):
                for local_band_idx in range(band_count):
                    self.index.append((track_idx, t_idx, band_offset + local_band_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        track_idx, t_idx, band_idx = self.index[index]
        track = self.track_refs[track_idx].load_kind(self.kind)
        frame = torch.tensor(np.asarray(track[t_idx, band_idx]), dtype=torch.float32)
        if self.sample_transform is not None:
            frame = self.sample_transform(frame)
        return frame


class _SimplePairedFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        track_refs: Sequence[TrackRef],
        *,
        n1: int,
        band: str,
        sample_transform: Callable | None = None,
    ):
        if band not in {'3p5', '28'}:
            raise ValueError(f'Unknown band: {band}')
        self.track_refs = list(track_refs)
        self.n1 = int(n1)
        self.band = band
        self.sample_transform = sample_transform
        self.index: list[tuple[int, int, int]] = []
        for track_idx, track_ref in enumerate(self.track_refs):
            t_len, n_bands = int(track_ref.shape[0]), int(track_ref.shape[1])
            if self.n1 <= 0 or self.n1 >= n_bands:
                raise ValueError(f'n1 must be in (0, N), got n1={self.n1}, N={n_bands}')
            band_offset = 0 if self.band == '3p5' else self.n1
            band_count = self.n1 if self.band == '3p5' else (n_bands - self.n1)
            for t_idx in range(t_len):
                for local_band_idx in range(band_count):
                    self.index.append((track_idx, t_idx, band_offset + local_band_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        track_idx, t_idx, band_idx = self.index[index]
        ref = self.track_refs[track_idx]
        gt = torch.tensor(np.asarray(ref.load_gt()[t_idx, band_idx]), dtype=torch.float32)
        gt_value = torch.tensor(np.asarray(ref.load_gt()[t_idx, band_idx]), dtype=torch.float32)
        if self.sample_transform is not None:
            gt_value = self.sample_transform(gt_value)
        return gt, gt_value


def build_independent_channel_dataloaders(
    dataset_file: str | Path | Sequence[str | Path],
    n1: int,
    batch_size: int,
    return_pair: bool = False,
    num_workers: int = 0,
    test_ratio: float = 0.2,
    seed: int = 42,
    sample_transform: Callable | None = None,
    *,
    train_paths: int = 512,
    test_paths: int = 88,
) -> dict[str, tuple[DataLoader, DataLoader]]:
    if return_pair:
        track_refs, scenario_names = load_track_pairs(dataset_file=dataset_file, return_scenarios=True)
        train_refs, test_refs = split_track_pairs_train_test(
            track_pairs=track_refs,
            test_ratio=test_ratio,
            seed=seed,
            scenario_names=scenario_names,
            train_paths=train_paths,
            test_paths=test_paths,
        )
        ds_train_3p5 = _SimplePairedFrameDataset(train_refs, n1=n1, band='3p5', sample_transform=sample_transform)
        ds_test_3p5 = _SimplePairedFrameDataset(test_refs, n1=n1, band='3p5', sample_transform=sample_transform)
        ds_train_28 = _SimplePairedFrameDataset(train_refs, n1=n1, band='28', sample_transform=sample_transform)
        ds_test_28 = _SimplePairedFrameDataset(test_refs, n1=n1, band='28', sample_transform=sample_transform)
    else:
        track_refs, scenario_names = load_tracks(dataset_file=dataset_file, kind='gt', return_scenarios=True)
        train_refs, test_refs = split_tracks_train_test(
            tracks=track_refs,
            test_ratio=test_ratio,
            seed=seed,
            scenario_names=scenario_names,
            train_paths=train_paths,
            test_paths=test_paths,
        )
        ds_train_3p5 = _SimpleFrameDataset(train_refs, n1=n1, band='3p5', kind='gt', sample_transform=sample_transform)
        ds_test_3p5 = _SimpleFrameDataset(test_refs, n1=n1, band='3p5', kind='gt', sample_transform=sample_transform)
        ds_train_28 = _SimpleFrameDataset(train_refs, n1=n1, band='28', kind='gt', sample_transform=sample_transform)
        ds_test_28 = _SimpleFrameDataset(test_refs, n1=n1, band='28', kind='gt', sample_transform=sample_transform)

    return {
        '3p5': (
            DataLoader(ds_train_3p5, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(ds_test_3p5, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        ),
        '28': (
            DataLoader(ds_train_28, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(ds_test_28, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        ),
    }


def build_independent_channel_three_way_dataloaders(
    dataset_file: str | Path | Sequence[str | Path],
    n1: int,
    batch_size: int,
    return_pair: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    sample_transform: Callable | None = None,
    *,
    train_paths: int = 512,
    test_paths: int = 88,
) -> dict[str, tuple[DataLoader, DataLoader, DataLoader]]:
    if return_pair:
        track_refs, scenario_names = load_track_pairs(dataset_file=dataset_file, return_scenarios=True)
        train_refs, intra_refs, cross_refs = split_tracks_train_intra_cross(
            tracks=track_refs,
            seed=seed,
            scenario_names=scenario_names,
            train_paths=train_paths,
            test_paths=test_paths,
        )
        ds_train_3p5 = _SimplePairedFrameDataset(train_refs, n1=n1, band='3p5', sample_transform=sample_transform)
        ds_intra_3p5 = _SimplePairedFrameDataset(intra_refs, n1=n1, band='3p5', sample_transform=sample_transform)
        ds_cross_3p5 = _SimplePairedFrameDataset(cross_refs, n1=n1, band='3p5', sample_transform=sample_transform)
        ds_train_28 = _SimplePairedFrameDataset(train_refs, n1=n1, band='28', sample_transform=sample_transform)
        ds_intra_28 = _SimplePairedFrameDataset(intra_refs, n1=n1, band='28', sample_transform=sample_transform)
        ds_cross_28 = _SimplePairedFrameDataset(cross_refs, n1=n1, band='28', sample_transform=sample_transform)
    else:
        track_refs, scenario_names = load_tracks(dataset_file=dataset_file, kind='gt', return_scenarios=True)
        train_refs, intra_refs, cross_refs = split_tracks_train_intra_cross(
            tracks=track_refs,
            seed=seed,
            scenario_names=scenario_names,
            train_paths=train_paths,
            test_paths=test_paths,
        )
        ds_train_3p5 = _SimpleFrameDataset(train_refs, n1=n1, band='3p5', kind='gt', sample_transform=sample_transform)
        ds_intra_3p5 = _SimpleFrameDataset(intra_refs, n1=n1, band='3p5', kind='gt', sample_transform=sample_transform)
        ds_cross_3p5 = _SimpleFrameDataset(cross_refs, n1=n1, band='3p5', kind='gt', sample_transform=sample_transform)
        ds_train_28 = _SimpleFrameDataset(train_refs, n1=n1, band='28', kind='gt', sample_transform=sample_transform)
        ds_intra_28 = _SimpleFrameDataset(intra_refs, n1=n1, band='28', kind='gt', sample_transform=sample_transform)
        ds_cross_28 = _SimpleFrameDataset(cross_refs, n1=n1, band='28', kind='gt', sample_transform=sample_transform)

    return {
        '3p5': (
            DataLoader(ds_train_3p5, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(ds_intra_3p5, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(ds_cross_3p5, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        ),
        '28': (
            DataLoader(ds_train_28, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(ds_intra_28, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(ds_cross_28, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        ),
    }
