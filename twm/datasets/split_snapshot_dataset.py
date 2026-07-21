from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def _load_snapshot(snapshot_file: str | Path) -> dict:
    path = Path(snapshot_file)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def _save_snapshot(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True)
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _snapshot_has_signal(snapshot: dict) -> bool:
    delay = np.asarray(snapshot.get("delay", []), dtype=np.float32)
    if delay.size == 0 or bool(np.all(delay < 0.0)):
        return False
    gain_r = np.asarray(snapshot.get("gain_r", []), dtype=np.float32)
    gain_i = np.asarray(snapshot.get("gain_i", []), dtype=np.float32)
    if gain_r.size == 0 or gain_i.size == 0:
        return False
    power = np.sum(gain_r.astype(np.float64) ** 2 + gain_i.astype(np.float64) ** 2)
    return bool(power > 0.0)


def _path_is_valid(path_sequence: list[dict]) -> bool:
    if not path_sequence:
        return False
    return all(_snapshot_has_signal(snapshot) for snapshot in path_sequence)


def _subset_payload(payload: dict, indices: list[int], split_name: str, seed: int, valid_indices: list[int]) -> dict:
    subset = dict(payload)
    subset["path_sequences"] = [payload["path_sequences"][i] for i in indices]
    if "coordinate_sequences" in payload:
        subset["coordinate_sequences"] = [payload["coordinate_sequences"][i] for i in indices]
    subset["path_split"] = {
        "split_name": split_name,
        "split_seed": int(seed),
        "selected_original_indices": [int(i) for i in indices],
        "num_paths": int(len(indices)),
        "path_length": 16,
        "slicing": "disabled_full_path",
        "num_valid_paths_before_split": int(len(valid_indices)),
    }
    return subset


def split_snapshot_dataset(
    *,
    input_file: str | Path,
    train_output_file: str | Path,
    test_output_file: str | Path,
    train_paths: int = 512,
    test_paths: int = 88,
    seed: int = 42,
    report_path: str | Path | None = None,
) -> tuple[Path, Path]:
    payload = _load_snapshot(input_file)
    path_sequences = payload.get("path_sequences")
    if not isinstance(path_sequences, list):
        raise ValueError("Snapshot payload must contain a list field path_sequences.")

    valid_indices = [idx for idx, seq in enumerate(path_sequences) if _path_is_valid(seq)]
    required = int(train_paths) + int(test_paths)
    if len(valid_indices) < required:
        raise ValueError(
            f"Not enough valid paths in snapshot: have {len(valid_indices)}, need {required} "
            f"(train={train_paths}, test={test_paths})."
        )

    rng = np.random.default_rng(int(seed))
    shuffled = valid_indices.copy()
    rng.shuffle(shuffled)
    train_indices = sorted(int(i) for i in shuffled[: int(train_paths)])
    test_indices = sorted(int(i) for i in shuffled[int(train_paths) : required])

    train_out = _save_snapshot(_subset_payload(payload, train_indices, "train", int(seed), valid_indices), train_output_file)
    test_out = _save_snapshot(_subset_payload(payload, test_indices, "test", int(seed), valid_indices), test_output_file)

    if report_path is not None:
        valid_set = set(valid_indices)
        invalid_indices = [int(i) for i in range(len(path_sequences)) if i not in valid_set]
        report = {
            "input_paths": int(len(path_sequences)),
            "valid_paths": int(len(valid_indices)),
            "invalid_paths": int(len(invalid_indices)),
            "invalid_original_indices": invalid_indices,
            "train_paths": int(len(train_indices)),
            "test_paths": int(len(test_indices)),
            "seed": int(seed),
            "train_original_indices": train_indices,
            "test_original_indices": test_indices,
        }
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    return train_out, test_out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a Sionna snapshot dataset into train/test snapshot files.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--train-output-file", required=True)
    parser.add_argument("--test-output-file", required=True)
    parser.add_argument("--train-paths", type=int, default=512)
    parser.add_argument("--test-paths", type=int, default=88)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    train_out, test_out = split_snapshot_dataset(
        input_file=args.input_file,
        train_output_file=args.train_output_file,
        test_output_file=args.test_output_file,
        train_paths=int(args.train_paths),
        test_paths=int(args.test_paths),
        seed=int(args.seed),
        report_path=args.report_path,
    )
    print(str(train_out))
    print(str(test_out))


if __name__ == "__main__":
    main()
