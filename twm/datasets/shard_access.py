from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TrackRef:
    source: Path
    source_kind: str
    scenario_name: str
    suffix: str
    shape: tuple[int, ...]
    raw_key: str | None = None
    gt_key: str | None = None
    xyz_key: str | None = None
    velocity_key: str | None = None
    bs_id_key: str | None = None

    def _load_key(self, key: str) -> np.ndarray:
        if self.source_kind == 'npy_dir':
            return np.load(self.source / f'{key}.npy', mmap_mode='r', allow_pickle=False)
        if self.source_kind == 'npz':
            with np.load(self.source, allow_pickle=False) as obj:
                return np.asarray(obj[key])
        raise ValueError(f'Unsupported source kind: {self.source_kind}')

    def load_raw(self) -> np.ndarray:
        if self.raw_key is None:
            raise KeyError('TrackRef has no raw key')
        return self._load_key(self.raw_key)

    def load_gt(self) -> np.ndarray:
        if self.gt_key is not None:
            return self._load_key(self.gt_key)
        if self.raw_key is not None:
            return self._load_key(self.raw_key)
        raise KeyError('TrackRef has no gt key')

    def load_kind(self, kind: str) -> np.ndarray:
        if kind == 'gt':
            return self.load_gt()
        raise ValueError(f'Unsupported track kind: {kind}')

    def load_xyz(self) -> np.ndarray:
        if self.xyz_key is None:
            raise KeyError('TrackRef has no xyz key')
        return self._load_key(self.xyz_key)

    def load_velocity(self) -> np.ndarray:
        if self.velocity_key is None:
            raise KeyError('TrackRef has no velocity key')
        return self._load_key(self.velocity_key)

    def load_bs_id(self) -> np.ndarray:
        if self.bs_id_key is None:
            raise KeyError('TrackRef has no bs_id key')
        return self._load_key(self.bs_id_key)


def _normalize_dataset_files(dataset_file: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(dataset_file, (str, Path)):
        files = [dataset_file]
    else:
        files = list(dataset_file)
    paths = [Path(item) for item in files if str(item)]
    if not paths:
        raise ValueError('At least one dataset file is required.')
    return paths


def _scenario_name_from_stem(stem: str) -> str:
    for suffix in (
        '_snapshot_gt_64ant',
        '_snapshot_fixed_64ant',
        '_gt_64ant',
        '_fixed_64ant',
        '_gt',
        '_gt',
    ):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _extract_scenario_name_from_payload(dataset_config_json) -> str | None:
    raw_value = np.asarray(dataset_config_json)
    if raw_value.shape != ():
        return None
    scalar = raw_value.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode('utf-8')
    if not isinstance(scalar, str):
        return None
    metadata = json.loads(scalar)
    for key in ('scenario_name', 'scene_name'):
        value = metadata.get(key)
        if value:
            return str(value)
    scenario_data = metadata.get('scenario_data')
    if isinstance(scenario_data, dict):
        for key in ('scenario_name', 'scene_name', 'name'):
            value = scenario_data.get(key)
            if value:
                return str(value)
    return None


def _resolve_shard_dir(path: Path) -> Path | None:
    if path.is_dir() and (path / 'manifest.json').exists():
        return path
    if path.suffix.lower() != '.npz':
        return None
    scenario_name = _scenario_name_from_stem(path.stem)
    scenario_root = path.parent / scenario_name
    candidates = [
        scenario_root / f'{path.stem}_npy',
        scenario_root / f'{scenario_name}_gt_npy',
        scenario_root / f'{scenario_name}_npy',
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / 'manifest.json').exists():
            return candidate
    return None


def _track_refs_from_npy_dir(path: Path) -> tuple[list[TrackRef], str]:
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    arrays: dict[str, dict] = manifest['arrays']
    scenario_name = str(manifest.get('scenario_name') or path.parent.name)
    refs: list[TrackRef] = []

    raw_keys = sorted(k for k in arrays if k.startswith('sample_') and k.count('_') == 1)
    if raw_keys:
        for key in raw_keys:
            suffix = key.removeprefix('sample_')
            refs.append(
                TrackRef(
                    source=path,
                    source_kind='npy_dir',
                    scenario_name=scenario_name,
                    suffix=suffix,
                    shape=tuple(arrays[key]['shape']),
                    raw_key=key,
                    xyz_key=f'sample_xyz_{suffix}' if f'sample_xyz_{suffix}' in arrays else None,
                    velocity_key=f'sample_velocity_{suffix}' if f'sample_velocity_{suffix}' in arrays else None,
                    bs_id_key=f'sample_bs_id_{suffix}' if f'sample_bs_id_{suffix}' in arrays else None,
                )
            )
        return refs, scenario_name

    gt_keys = sorted(k for k in arrays if k.startswith('sample_gt_'))
    if gt_keys:
        for gt_key in gt_keys:
            suffix = gt_key.removeprefix('sample_gt_')
            refs.append(
                TrackRef(
                    source=path,
                    source_kind='npy_dir',
                    scenario_name=scenario_name,
                    suffix=suffix,
                    shape=tuple(arrays[gt_key]['shape']),
                    gt_key=gt_key,
                    xyz_key=f'sample_xyz_{suffix}' if f'sample_xyz_{suffix}' in arrays else None,
                    velocity_key=f'sample_velocity_{suffix}' if f'sample_velocity_{suffix}' in arrays else None,
                    bs_id_key=f'sample_bs_id_{suffix}' if f'sample_bs_id_{suffix}' in arrays else None,
                )
            )
        return refs, scenario_name

    gt_keys = sorted(k for k in arrays if k.startswith('sample_gt_'))
    for gt_key in gt_keys:
        suffix = gt_key.removeprefix('sample_gt_')
        refs.append(
            TrackRef(
                source=path,
                source_kind='npy_dir',
                scenario_name=scenario_name,
                suffix=suffix,
                shape=tuple(arrays[gt_key]['shape']),
                gt_key=gt_key,
                xyz_key=f'sample_xyz_{suffix}' if f'sample_xyz_{suffix}' in arrays else None,
                velocity_key=f'sample_velocity_{suffix}' if f'sample_velocity_{suffix}' in arrays else None,
                bs_id_key=f'sample_bs_id_{suffix}' if f'sample_bs_id_{suffix}' in arrays else None,
            )
        )
    return refs, scenario_name


def _track_refs_from_npz(path: Path) -> tuple[list[TrackRef], str]:
    refs: list[TrackRef] = []
    with np.load(path, allow_pickle=False) as obj:
        scenario_name = _extract_scenario_name_from_payload(obj['dataset_config_json']) if 'dataset_config_json' in obj.files else None
        if not scenario_name:
            scenario_name = _scenario_name_from_stem(path.stem)

        raw_keys = sorted(k for k in obj.files if k.startswith('sample_') and k.count('_') == 1)
        if raw_keys:
            for key in raw_keys:
                suffix = key.removeprefix('sample_')
                refs.append(
                    TrackRef(
                        source=path,
                        source_kind='npz',
                        scenario_name=scenario_name,
                        suffix=suffix,
                        shape=tuple(np.asarray(obj[key]).shape),
                        raw_key=key,
                        xyz_key=f'sample_xyz_{suffix}' if f'sample_xyz_{suffix}' in obj.files else None,
                        velocity_key=f'sample_velocity_{suffix}' if f'sample_velocity_{suffix}' in obj.files else None,
                        bs_id_key=f'sample_bs_id_{suffix}' if f'sample_bs_id_{suffix}' in obj.files else None,
                    )
                )
            return refs, scenario_name

        gt_keys = sorted(k for k in obj.files if k.startswith('sample_gt_'))
        if gt_keys:
            for gt_key in gt_keys:
                suffix = gt_key.removeprefix('sample_gt_')
                refs.append(
                    TrackRef(
                        source=path,
                        source_kind='npz',
                        scenario_name=scenario_name,
                        suffix=suffix,
                        shape=tuple(np.asarray(obj[gt_key]).shape),
                        gt_key=gt_key,
                        xyz_key=f'sample_xyz_{suffix}' if f'sample_xyz_{suffix}' in obj.files else None,
                        velocity_key=f'sample_velocity_{suffix}' if f'sample_velocity_{suffix}' in obj.files else None,
                        bs_id_key=f'sample_bs_id_{suffix}' if f'sample_bs_id_{suffix}' in obj.files else None,
                    )
                )
            return refs, scenario_name

        gt_keys = sorted(k for k in obj.files if k.startswith('sample_gt_'))
        for gt_key in gt_keys:
            suffix = gt_key.removeprefix('sample_gt_')
            refs.append(
                TrackRef(
                    source=path,
                    source_kind='npz',
                    scenario_name=scenario_name,
                    suffix=suffix,
                    shape=tuple(np.asarray(obj[gt_key]).shape),
                    gt_key=gt_key,
                    xyz_key=f'sample_xyz_{suffix}' if f'sample_xyz_{suffix}' in obj.files else None,
                    velocity_key=f'sample_velocity_{suffix}' if f'sample_velocity_{suffix}' in obj.files else None,
                    bs_id_key=f'sample_bs_id_{suffix}' if f'sample_bs_id_{suffix}' in obj.files else None,
                )
            )
    return refs, scenario_name


def load_track_refs(dataset_file: str | Path | Sequence[str | Path]) -> tuple[list[TrackRef], list[str]]:
    refs: list[TrackRef] = []
    scenario_names: list[str] = []
    for path in _normalize_dataset_files(dataset_file):
        shard_dir = _resolve_shard_dir(path)
        if shard_dir is not None:
            file_refs, scenario_name = _track_refs_from_npy_dir(shard_dir)
        elif path.suffix.lower() == '.npz':
            file_refs, scenario_name = _track_refs_from_npz(path)
        else:
            raise ValueError(f'Unsupported dataset source: {path}')
        refs.extend(file_refs)
        scenario_names.extend([scenario_name] * len(file_refs))
    if not refs:
        raise ValueError('Dataset source contains no track references.')
    return refs, scenario_names
