from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .shard_access import TrackRef, load_track_refs


def _select_refs(
    refs: Sequence[TrackRef],
    *,
    scenario_name: str,
    bs_id: int,
    num_paths: int,
) -> list[TrackRef]:
    matches: list[TrackRef] = []
    for ref in refs:
        if str(ref.scenario_name) != str(scenario_name):
            continue
        if ref.bs_id_key is None:
            continue
        bs_ids = np.asarray(ref.load_bs_id(), dtype=np.int32)
        if bs_ids.size == 0 or not np.all(bs_ids == int(bs_id)):
            continue
        matches.append(ref)

    if len(matches) < num_paths:
        raise ValueError(
            f"Requested {num_paths} paths for scenario={scenario_name!r}, bs_id={bs_id}, "
            f"but found only {len(matches)} matching paths."
        )
    return matches[:num_paths]


def _infer_trim_length(selected_refs: Sequence[TrackRef], *, approx_samples: int) -> int:
    if not selected_refs:
        raise ValueError("No track refs provided.")
    num_paths = len(selected_refs)
    num_bands = int(selected_refs[0].shape[1])
    target_steps = max(1, int(round(float(approx_samples) / float(num_paths * num_bands))))
    max_available = min(int(ref.shape[0]) for ref in selected_refs)
    return min(target_steps, max_available)


def _build_subset_metadata(
    *,
    source_dataset_file: str | Path,
    scenario_name: str,
    bs_id: int,
    selected_refs: Sequence[TrackRef],
    trim_length: int,
    approx_samples: int,
) -> np.ndarray:
    metadata = {
        "format_version": 1,
        "subset_type": "bs1_trial_subset",
        "source_dataset_file": str(source_dataset_file),
        "scenario_name": str(scenario_name),
        "bs_id": int(bs_id),
        "num_paths": len(selected_refs),
        "selected_suffixes": [ref.suffix for ref in selected_refs],
        "trim_length": int(trim_length),
        "approx_frame_band_samples": int(len(selected_refs) * trim_length * int(selected_refs[0].shape[1])),
        "requested_approx_samples": int(approx_samples),
    }
    return np.asarray(json.dumps(metadata), dtype=np.str_)


def build_bs1_trial_subset(
    *,
    source_dataset_file: str | Path,
    output_file: str | Path,
    scenario_name: str = "01chicago",
    bs_id: int = 1,
    num_paths: int = 2,
    approx_samples: int = 500,
    trim_length: int | None = None,
) -> Path:
    refs, _ = load_track_refs(source_dataset_file)
    selected_refs = _select_refs(refs, scenario_name=scenario_name, bs_id=bs_id, num_paths=num_paths)
    trim_length = int(trim_length) if trim_length is not None else _infer_trim_length(selected_refs, approx_samples=approx_samples)
    if trim_length <= 0:
        raise ValueError(f"trim_length must be positive, got {trim_length}")

    payload: dict[str, np.ndarray] = {}
    for out_idx, ref in enumerate(selected_refs):
        suffix = f"{out_idx:06d}"
        gt = np.asarray(ref.load_gt(), dtype=np.float32)[:trim_length]
        payload[f"sample_gt_{suffix}"] = gt
        if ref.xyz_key is not None:
            payload[f"sample_xyz_{suffix}"] = np.asarray(ref.load_xyz(), dtype=np.float32)[:trim_length]
        if ref.velocity_key is not None:
            payload[f"sample_velocity_{suffix}"] = np.asarray(ref.load_velocity(), dtype=np.float32)[:trim_length]
        if ref.bs_id_key is not None:
            payload[f"sample_bs_id_{suffix}"] = np.asarray(ref.load_bs_id(), dtype=np.int32)[:trim_length]

    payload["dataset_config_json"] = _build_subset_metadata(
        source_dataset_file=source_dataset_file,
        scenario_name=scenario_name,
        bs_id=bs_id,
        selected_refs=selected_refs,
        trim_length=trim_length,
        approx_samples=approx_samples,
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a tiny BS-specific trial subset from an existing TWM dataset.")
    parser.add_argument("--source-dataset-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--scenario-name", default="01chicago")
    parser.add_argument("--bs-id", type=int, default=1)
    parser.add_argument("--num-paths", type=int, default=2)
    parser.add_argument("--approx-samples", type=int, default=500)
    parser.add_argument("--trim-length", type=int, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    out = build_bs1_trial_subset(
        source_dataset_file=args.source_dataset_file,
        output_file=args.output_file,
        scenario_name=args.scenario_name,
        bs_id=int(args.bs_id),
        num_paths=int(args.num_paths),
        approx_samples=int(args.approx_samples),
        trim_length=args.trim_length,
    )
    print(str(out))


if __name__ == "__main__":
    main()
