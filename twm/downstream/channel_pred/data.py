from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from twm.datasets.shard_access import TrackRef, load_track_refs


def build_window_slices(
    track_length: int,
    *,
    window_length: int,
    random_windows_per_sequence: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    track_length = int(track_length)
    window_length = max(0, int(window_length))
    random_windows_per_sequence = max(1, int(random_windows_per_sequence))
    if track_length < 2:
        raise ValueError(f'Track length must be at least 2, got {track_length}')
    if window_length <= 0 or window_length >= track_length:
        return [(0, track_length)]
    max_start = track_length - window_length
    num_windows = min(random_windows_per_sequence, max_start + 1)
    start_indices = rng.choice(max_start + 1, size=num_windows, replace=False)
    return [(int(start), window_length) for start in sorted(int(item) for item in np.atleast_1d(start_indices).tolist())]


def load_channel_pred_track_pairs(
    dataset_file: str | Path | Sequence[str | Path],
    *,
    return_scenarios: bool = False,
):
    track_refs, _ = load_track_refs(dataset_file)
    paired_refs = [ref for ref in track_refs if ref.gt_key is not None or ref.raw_key is not None]
    if not paired_refs:
        raise ValueError('Dataset file contains no usable tracks with gt data.')
    if return_scenarios:
        return paired_refs, [ref.scenario_name for ref in paired_refs]
    return paired_refs


def split_channel_pred_tracks_train_intra_cross(
    tracks: list[TrackRef],
    *,
    seed: int = 42,
    scenario_names: Sequence[str],
    train_paths: int = 512,
    test_paths: int = 88,
) -> tuple[list[TrackRef], list[TrackRef], list[TrackRef]]:
    if not tracks:
        raise ValueError('No tracks provided for split.')
    if len(scenario_names) != len(tracks):
        raise ValueError('scenario_names length must match tracks length.')
    rng = np.random.default_rng(seed)
    train_paths = max(0, int(train_paths))
    test_paths = max(0, int(test_paths))
    required = train_paths + test_paths
    if len(tracks) < required:
        raise ValueError(
            f'Not enough paired tracks for requested split: have {len(tracks)}, '
            f'need {required} (= {train_paths} train + {test_paths} test).'
        )
    permutation = rng.permutation(len(tracks)).tolist()
    train_idx = set(int(i) for i in permutation[:train_paths])
    test_idx = set(int(i) for i in permutation[train_paths : train_paths + test_paths])
    train_refs = [tracks[i] for i in range(len(tracks)) if i in train_idx]
    test_refs = [tracks[i] for i in range(len(tracks)) if i in test_idx]
    return train_refs, test_refs, []
