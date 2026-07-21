from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _scenario_name_for_npz(npz_path: Path) -> str:
    with np.load(npz_path, allow_pickle=False) as obj:
        if 'dataset_config_json' in obj.files:
            raw = np.asarray(obj['dataset_config_json'])
            if raw.shape == ():
                scalar = raw.item()
                if isinstance(scalar, bytes):
                    scalar = scalar.decode('utf-8')
                if isinstance(scalar, str):
                    meta = json.loads(scalar)
                    for key in ('scenario_name', 'scene_name'):
                        value = meta.get(key)
                        if value:
                            return str(value)
    stem = npz_path.stem
    for suffix in ('_paired_3e-5', '_pair'):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _output_dir_for_npz(npz_path: Path, output_root: Path | None = None) -> Path:
    scenario_name = _scenario_name_for_npz(npz_path)
    root = output_root if output_root is not None else npz_path.parent
    return root / scenario_name / f'{npz_path.stem}_npy'


def convert_npz_to_npy_dir(npz_path: str | Path, output_dir: str | Path | None = None, *, overwrite: bool = False) -> Path:
    src = Path(npz_path)
    if not src.exists():
        raise FileNotFoundError(f'Input dataset not found: {src}')
    if src.suffix.lower() != '.npz':
        raise ValueError(f'Expected .npz input, got: {src}')

    dst = Path(output_dir) if output_dir is not None else _output_dir_for_npz(src)
    if dst.exists() and any(dst.iterdir()) and not overwrite:
        raise FileExistsError(f'Output directory already exists and is not empty: {dst}')
    dst.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        'format': 'twm_npy_shards_v1',
        'source_npz': str(src),
        'scenario_name': _scenario_name_for_npz(src),
        'arrays': {},
    }

    with np.load(src, allow_pickle=False) as obj:
        for key in obj.files:
            arr = np.asarray(obj[key])
            fname = f'{key}.npy'
            np.save(dst / fname, arr, allow_pickle=False)
            manifest['arrays'][key] = {
                'file': fname,
                'shape': list(arr.shape),
                'dtype': str(arr.dtype),
            }

    (dst / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    return dst


def convert_many(npz_files: Iterable[str | Path], output_root: str | Path | None = None, *, overwrite: bool = False) -> list[Path]:
    root = Path(output_root) if output_root is not None else None
    outputs = []
    for item in npz_files:
        out_dir = _output_dir_for_npz(Path(item), output_root=root)
        outputs.append(convert_npz_to_npy_dir(item, output_dir=out_dir, overwrite=overwrite))
    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert a TWM .npz dataset into scenario-local .npy shards.')
    parser.add_argument('npz_files', nargs='+', help='Input .npz dataset files to convert')
    parser.add_argument('--output-root', default=None, help='Optional dataset root. Scenario subfolders are created under this root.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite an existing non-empty output directory')
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    outputs = convert_many(args.npz_files, output_root=args.output_root, overwrite=bool(args.overwrite))
    for out in outputs:
        print(str(out))


if __name__ == '__main__':
    main()
