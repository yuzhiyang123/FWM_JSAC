from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_metadata(obj: np.lib.npyio.NpzFile) -> dict:
    if "dataset_config_json" not in obj.files:
        return {}
    raw = np.asarray(obj["dataset_config_json"])
    if raw.shape != ():
        return {}
    scalar = raw.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str):
        return {}
    return json.loads(scalar)


def _collect_sample_suffixes(obj: np.lib.npyio.NpzFile) -> list[str]:
    return sorted(key.removeprefix("sample_gt_") for key in obj.files if key.startswith("sample_gt_"))


def _subset_payload(
    obj: np.lib.npyio.NpzFile,
    *,
    suffixes: list[str],
    metadata: dict,
    split_name: str,
    split_indices: list[int],
    split_seed: int,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for new_idx, suffix in enumerate(suffixes):
        out_suffix = f"{new_idx:06d}"
        payload[f"sample_gt_{out_suffix}"] = np.asarray(obj[f"sample_gt_{suffix}"])
        if f"sample_xyz_{suffix}" in obj.files:
            payload[f"sample_xyz_{out_suffix}"] = np.asarray(obj[f"sample_xyz_{suffix}"])
        if f"sample_velocity_{suffix}" in obj.files:
            payload[f"sample_velocity_{out_suffix}"] = np.asarray(obj[f"sample_velocity_{suffix}"])
        if f"sample_bs_id_{suffix}" in obj.files:
            payload[f"sample_bs_id_{out_suffix}"] = np.asarray(obj[f"sample_bs_id_{suffix}"])

    split_metadata = dict(metadata)
    split_metadata["split_name"] = split_name
    split_metadata["split_seed"] = int(split_seed)
    split_metadata["num_paths"] = len(suffixes)
    split_metadata["selected_suffixes"] = list(suffixes)
    split_metadata["selected_original_indices"] = [int(i) for i in split_indices]
    split_metadata["path_length"] = 16
    split_metadata["slicing"] = "disabled_full_path"
    payload["dataset_config_json"] = np.asarray(json.dumps(split_metadata), dtype=np.str_)
    return payload


def split_gt_dataset(
    *,
    input_file: str | Path,
    train_output_file: str | Path,
    test_output_file: str | Path,
    train_paths: int = 512,
    test_paths: int = 88,
    seed: int = 42,
) -> tuple[Path, Path]:
    src = Path(input_file)
    train_dst = Path(train_output_file)
    test_dst = Path(test_output_file)
    train_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.parent.mkdir(parents=True, exist_ok=True)

    with np.load(src, allow_pickle=False) as obj:
        metadata = _load_metadata(obj)
        suffixes = _collect_sample_suffixes(obj)
        required = int(train_paths) + int(test_paths)
        if len(suffixes) < required:
            raise ValueError(
                f"Not enough GT paths in {src}: have {len(suffixes)}, need {required} "
                f"(train={train_paths}, test={test_paths})."
            )
        rng = np.random.default_rng(int(seed))
        permutation = rng.permutation(len(suffixes)).tolist()
        train_idx = sorted(int(i) for i in permutation[: int(train_paths)])
        test_idx = sorted(int(i) for i in permutation[int(train_paths) : required])
        train_suffixes = [suffixes[i] for i in train_idx]
        test_suffixes = [suffixes[i] for i in test_idx]

        np.savez_compressed(
            train_dst,
            **_subset_payload(
                obj,
                suffixes=train_suffixes,
                metadata=metadata,
                split_name="train",
                split_indices=train_idx,
                split_seed=seed,
            ),
        )
        np.savez_compressed(
            test_dst,
            **_subset_payload(
                obj,
                suffixes=test_suffixes,
                metadata=metadata,
                split_name="test",
                split_indices=test_idx,
                split_seed=seed,
            ),
        )

    return train_dst, test_dst


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a GT-only TWM dataset into train/test path datasets.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--train-output-file", required=True)
    parser.add_argument("--test-output-file", required=True)
    parser.add_argument("--train-paths", type=int, default=512)
    parser.add_argument("--test-paths", type=int, default=88)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    train_out, test_out = split_gt_dataset(
        input_file=args.input_file,
        train_output_file=args.train_output_file,
        test_output_file=args.test_output_file,
        train_paths=int(args.train_paths),
        test_paths=int(args.test_paths),
        seed=int(args.seed),
    )
    print(str(train_out))
    print(str(test_out))


if __name__ == "__main__":
    main()
