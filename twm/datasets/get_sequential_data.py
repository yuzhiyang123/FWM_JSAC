from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .shard_access import TrackRef, load_track_refs


def load_tracks(
    dataset_file: str | Path | Sequence[str | Path],
    *,
    kind: str = 'gt',
    return_scenarios: bool = False,
):
    track_refs, _ = load_track_refs(dataset_file)
    if kind == 'gt':
        track_refs = [ref for ref in track_refs if ref.gt_key is not None or ref.raw_key is not None]
    elif kind == 'gt':
        track_refs = [ref for ref in track_refs if ref.gt_key is not None or ref.raw_key is not None]
    else:
        raise ValueError(f'Unknown track kind: {kind}')
    if not track_refs:
        raise ValueError(f"Dataset file contains no '{kind}' tracks.")
    if return_scenarios:
        return track_refs, [ref.scenario_name for ref in track_refs]
    return track_refs


def _random_train_test_split_indices(
    num_tracks: int,
    *,
    seed: int = 42,
    train_paths: int = 512,
    test_paths: int = 88,
) -> tuple[list[int], list[int]]:
    if num_tracks <= 0:
        raise ValueError('No tracks provided for split.')
    train_paths = max(0, int(train_paths))
    test_paths = max(0, int(test_paths))
    required = train_paths + test_paths
    if required <= 0:
        raise ValueError('train_paths + test_paths must be positive.')
    if num_tracks < required:
        raise ValueError(
            f'Not enough tracks for requested split: have {num_tracks}, '
            f'need {required} (= {train_paths} train + {test_paths} test).'
        )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_tracks).tolist()
    train_idx = sorted(int(i) for i in permutation[:train_paths])
    test_idx = sorted(int(i) for i in permutation[train_paths : train_paths + test_paths])
    return train_idx, test_idx

def split_tracks_train_intra_cross(
    tracks: list,
    *,
    seed: int = 42,
    scenario_names: Sequence[str],
    train_paths: int = 512,
    test_paths: int = 88,
) -> tuple[list, list, list]:
    if not tracks:
        raise ValueError('No tracks provided for split.')
    if len(scenario_names) != len(tracks):
        raise ValueError('scenario_names length must match tracks length.')
    train_idx, test_idx = _random_train_test_split_indices(
        len(tracks),
        seed=seed,
        train_paths=train_paths,
        test_paths=test_paths,
    )
    return [tracks[i] for i in train_idx], [tracks[i] for i in test_idx], []


def split_tracks_train_test(
    tracks: list,
    test_ratio: float = 0.2,
    seed: int = 42,
    *,
    scenario_names: Sequence[str] | None = None,
    train_paths: int = 512,
    test_paths: int = 88,
) -> tuple[list, list]:
    if not tracks:
        raise ValueError('No tracks provided for split.')
    if scenario_names is not None and len(scenario_names) != len(tracks):
        raise ValueError('scenario_names length must match tracks length.')
    train_idx, test_idx = _random_train_test_split_indices(
        len(tracks),
        seed=seed,
        train_paths=train_paths,
        test_paths=test_paths,
    )
    return [tracks[i] for i in train_idx], [tracks[i] for i in test_idx]


def load_train_test_samples(
    dataset_file: str | Path | Sequence[str | Path],
    test_ratio: float = 0.2,
    seed: int = 42,
    *,
    train_paths: int = 512,
    test_paths: int = 88,
) -> tuple[list[TrackRef], list[TrackRef]]:
    tracks, scenario_names = load_tracks(dataset_file, kind='gt', return_scenarios=True)
    return split_tracks_train_test(
        tracks=tracks,
        test_ratio=test_ratio,
        seed=seed,
        scenario_names=scenario_names,
        train_paths=train_paths,
        test_paths=test_paths,
    )


def load_train_intra_cross_samples(
    dataset_file: str | Path | Sequence[str | Path],
    seed: int = 42,
    *,
    train_paths: int = 512,
    test_paths: int = 88,
) -> tuple[list[TrackRef], list[TrackRef], list[TrackRef]]:
    tracks, scenario_names = load_tracks(dataset_file, kind='gt', return_scenarios=True)
    return split_tracks_train_intra_cross(
        tracks=tracks,
        seed=seed,
        scenario_names=scenario_names,
        train_paths=train_paths,
        test_paths=test_paths,
    )


class _TimeBandTrackDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        track_refs: Sequence[TrackRef],
        *,
        window_length: int,
        random_windows_per_sequence: int = 1,
        sample_transform: Callable | None = None,
        seed: int = 42,
    ):
        self.track_refs = list(track_refs)
        self.window_length = max(0, int(window_length))
        self.random_windows_per_sequence = max(1, int(random_windows_per_sequence))
        self.sample_transform = sample_transform
        self._rng = np.random.default_rng(seed)
        self.index: list[tuple[int, int, int]] = []

        for track_idx, ref in enumerate(self.track_refs):
            track_length = int(ref.shape[0])
            if self.window_length <= 0 or self.window_length >= track_length:
                self.index.append((track_idx, 0, track_length))
                continue

            max_start = track_length - self.window_length
            num_windows = min(self.random_windows_per_sequence, max_start + 1)
            start_indices = self._rng.choice(max_start + 1, size=num_windows, replace=False)
            for start in sorted(int(item) for item in np.atleast_1d(start_indices).tolist()):
                self.index.append((track_idx, start, self.window_length))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        track_idx, start_idx, length = self.index[index]
        sample = self.track_refs[track_idx].load_kind('gt')
        window = np.array(sample[start_idx : start_idx + length], dtype=np.float32, copy=True)
        tensor = torch.from_numpy(window)
        if self.sample_transform is not None:
            tensor = self.sample_transform(tensor)
        return tensor


def _pad_time_band_collate(batch: list[torch.Tensor]):
    max_t = int(max(item.shape[0] for item in batch))
    padded = torch.zeros(
        len(batch),
        max_t,
        batch[0].shape[1],
        batch[0].shape[2],
        batch[0].shape[3],
        batch[0].shape[4],
        dtype=torch.float32,
    )
    for i, item in enumerate(batch):
        padded[i, : item.shape[0]] = item
    return padded


def build_time_band_channel_dataloaders(
    dataset_file: str | Path | Sequence[str | Path],
    batch_size: int,
    window_length: int,
    random_windows_per_sequence: int = 1,
    num_workers: int = 0,
    test_ratio: float = 0.2,
    seed: int = 42,
    sample_transform: Callable | None = None,
    *,
    train_paths: int = 512,
    test_paths: int = 88,
) -> tuple[DataLoader, DataLoader]:
    train_refs, test_refs = load_train_test_samples(
        dataset_file=dataset_file,
        test_ratio=test_ratio,
        seed=seed,
        train_paths=train_paths,
        test_paths=test_paths,
    )
    train_dataset = _TimeBandTrackDataset(
        train_refs,
        window_length=window_length,
        random_windows_per_sequence=random_windows_per_sequence,
        sample_transform=sample_transform,
        seed=seed,
    )
    test_dataset = _TimeBandTrackDataset(
        test_refs,
        window_length=window_length,
        random_windows_per_sequence=max(1, random_windows_per_sequence),
        sample_transform=sample_transform,
        seed=seed + 1,
    )
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_pad_time_band_collate,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_pad_time_band_collate,
        ),
    )


def build_time_band_channel_three_way_dataloaders(
    dataset_file: str | Path | Sequence[str | Path],
    batch_size: int,
    window_length: int,
    random_windows_per_sequence: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    sample_transform: Callable | None = None,
    *,
    train_paths: int = 512,
    test_paths: int = 88,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_refs, test_refs, cross_refs = load_train_intra_cross_samples(
        dataset_file=dataset_file,
        seed=seed,
        train_paths=train_paths,
        test_paths=test_paths,
    )
    train_dataset = _TimeBandTrackDataset(
        train_refs,
        window_length=window_length,
        random_windows_per_sequence=random_windows_per_sequence,
        sample_transform=sample_transform,
        seed=seed,
    )
    test_dataset = _TimeBandTrackDataset(
        test_refs,
        window_length=window_length,
        random_windows_per_sequence=max(1, random_windows_per_sequence),
        sample_transform=sample_transform,
        seed=seed + 1,
    )
    cross_dataset = _TimeBandTrackDataset(
        cross_refs,
        window_length=window_length,
        random_windows_per_sequence=max(1, random_windows_per_sequence),
        sample_transform=sample_transform,
        seed=seed + 2,
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_pad_time_band_collate),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_pad_time_band_collate),
        DataLoader(cross_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_pad_time_band_collate),
    )
