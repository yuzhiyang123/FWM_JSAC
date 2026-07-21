from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np
import torch


C = 299792458.0


@dataclass(frozen=True)
class FixedBand:
    name: str
    center_freq_hz: float
    subcarrier_spacing_hz: float
    num_subcarriers: int = 1024


LOW_BANDS: tuple[FixedBand, ...] = (
    FixedBand("3.1G", 3.1e9, 15e3, 1024),
    FixedBand("3.3G", 3.3e9, 15e3, 1024),
    FixedBand("3.5G", 3.5e9, 15e3, 1024),
    FixedBand("3.7G", 3.7e9, 15e3, 1024),
    FixedBand("3.9G", 3.9e9, 15e3, 1024),
)

HIGH_BANDS: tuple[FixedBand, ...] = (
    FixedBand("27G", 27.0e9, 240e3, 1024),
    FixedBand("28G", 28.0e9, 240e3, 1024),
    FixedBand("29G", 29.0e9, 240e3, 1024),
)

DEFAULT_FIXED_BANDS: tuple[FixedBand, ...] = (*LOW_BANDS, *HIGH_BANDS)


def load_sionna_snapshot_file(snapshot_file: str | Path) -> dict:
    path = Path(snapshot_file)
    if not path.exists():
        raise FileNotFoundError(f"Sionna snapshot file not found: {path}")
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset_metadata(
    *,
    snapshot_payload: dict,
    snapshot_file: str | Path,
    num_ant: int,
    ant_type: str,
    bands: list[dict],
) -> dict:
    return {
        "format_version": 6,
        "scenario_name": snapshot_payload["scenario_name"],
        "scene_name": snapshot_payload.get("scene_name", snapshot_payload["scenario_name"]),
        "raw_data_dir": snapshot_payload.get("raw_data_dir", ""),
        "xyz_paths_file": snapshot_payload.get("xyz_paths_file", ""),
        "snapshot_file": str(snapshot_file),
        "snapshot_format_version": int(snapshot_payload.get("format_version", 1)),
        "snapshot_generator": snapshot_payload.get("generator", "twm.datasets.data_gen_sionna"),
        "batch_size": int(snapshot_payload.get("batch_size", 0)),
        "num_ant": int(num_ant),
        "ant_type": str(ant_type),
        "bands": bands,
        "base_stations": snapshot_payload.get("runtime_base_stations", []),
        "scenario_data": snapshot_payload.get("scenario_data", {}),
        "path_split": snapshot_payload.get("path_split", {}),
    }


def get_dft_mat(n: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    phase = idx.view(-1, 1) * idx.view(1, -1) * (2.0 * pi / float(n))
    return torch.exp(-1j * phase)


def _rotation_matrix_from_angles(angles: list[float] | tuple[float, float, float], device: torch.device) -> torch.Tensor:
    azimuth, elevation, roll = [float(v) for v in angles]
    cz = torch.cos(torch.tensor(azimuth, device=device, dtype=torch.float32))
    sz = torch.sin(torch.tensor(azimuth, device=device, dtype=torch.float32))
    cy = torch.cos(torch.tensor(elevation, device=device, dtype=torch.float32))
    sy = torch.sin(torch.tensor(elevation, device=device, dtype=torch.float32))
    cx = torch.cos(torch.tensor(roll, device=device, dtype=torch.float32))
    sx = torch.sin(torch.tensor(roll, device=device, dtype=torch.float32))

    rx = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, cx.item(), -sx.item()],
        [0.0, sx.item(), cx.item()],
    ], device=device, dtype=torch.float32)
    ry = torch.tensor([
        [cy.item(), 0.0, sy.item()],
        [0.0, 1.0, 0.0],
        [-sy.item(), 0.0, cy.item()],
    ], device=device, dtype=torch.float32)
    rz = torch.tensor([
        [cz.item(), -sz.item(), 0.0],
        [sz.item(), cz.item(), 0.0],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=torch.float32)
    return rz @ ry @ rx


def _direction_to_sionna_orientation(direction_xyz: list[float] | tuple[float, float, float]) -> list[float]:
    direction = np.asarray(direction_xyz, dtype=np.float32)
    if direction.shape != (3,):
        raise ValueError(f"orientation direction must have shape [3], got {direction.shape}")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("orientation direction vector must be non-zero.")
    unit = direction / norm
    theta = float(np.arccos(np.clip(unit[2], -1.0, 1.0)))
    phi = float(np.arctan2(unit[1], unit[0]))
    return [phi, theta - float(np.pi) / 2.0, 0.0]


def get_bs_ant(
    num_ant: int,
    ant_type: str = "rectangle",
    ant_rotate: list[float] | tuple[float, float, float] | None = None,
    *,
    device: torch.device,
) -> torch.Tensor:
    ant_type = str(ant_type).lower()
    if num_ant <= 0:
        raise ValueError(f"num_ant must be positive, got {num_ant}")

    if ant_type == "ula":
        positions = (torch.arange(num_ant, device=device, dtype=torch.float32) - (num_ant - 1) / 2.0) * 0.5
        zeros = torch.zeros(num_ant, 2, device=device, dtype=torch.float32)
        bs_ant = torch.cat([positions.unsqueeze(1), zeros], dim=1)
    elif ant_type == "rectangle":
        rows = int(round(math.sqrt(num_ant)))
        cols = int(math.ceil(num_ant / rows))
        if rows * cols != num_ant:
            raise ValueError(
                f"rectangle antenna requires num_ant to factor into a grid cleanly, got num_ant={num_ant}"
            )
        row_positions = (torch.arange(rows, device=device, dtype=torch.float32) - (rows - 1) / 2.0) * 0.5
        col_positions = (torch.arange(cols, device=device, dtype=torch.float32) - (cols - 1) / 2.0) * 0.5
        yy, zz = torch.meshgrid(row_positions, col_positions, indexing="ij")
        xx = torch.zeros_like(yy)
        bs_ant = torch.stack([xx, yy, zz], dim=-1).reshape(num_ant, 3)
    else:
        raise ValueError(f"Unsupported ant_type: {ant_type}. Expected rectangle or ULA.")

    if ant_rotate is None:
        return bs_ant

    rotation = _rotation_matrix_from_angles(ant_rotate, device=device)
    return bs_ant @ rotation.T


def _sub_min_delay(delay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = delay > 0
    x_nonzero = delay.clone()
    x_nonzero[~mask] = float("inf")
    min_vals = x_nonzero.min(dim=1, keepdim=True).values
    delay = delay.clone()
    delay[mask] = delay[mask] - min_vals.expand_as(delay)[mask]
    return delay, min_vals


def _get_moving_channel_simplified(
    gain: torch.Tensor,
    delay: torch.Tensor,
    aod: torch.Tensor,
    aoa: torch.Tensor,
    *,
    num_ant: int,
    wc: float,
    fs: float,
    bs_ant: torch.Tensor,
    num_car: int,
    dft_mat: torch.Tensor,
    velocity: torch.Tensor,
    v_radial_offset: float = 0.0,
) -> torch.Tensor:
    bsz = gain.shape[0]
    delta_t = 0.0
    v_radial = velocity.view(bsz, 1, 3).mul(aoa).sum(dim=2, keepdim=True)
    delay, delay_offset = _sub_min_delay(delay)
    n = torch.floor(delay * fs + 0.5)

    tmp = torch.arange(num_car, device=gain.device).view(1, 1, -1) - n
    phase_doppler = (v_radial - v_radial_offset) * (tmp / fs + delta_t - delay_offset) * (wc / C)

    antenna_delay = bs_ant.view(1, num_ant, 1, 3).mul(aod.unsqueeze(1)).sum(dim=3, keepdim=True) / C
    phase = delay.unsqueeze(1) - antenna_delay
    path_gain = torch.exp(1j * wc * phase)
    path_doppler_gain = torch.exp(1j * phase_doppler.unsqueeze(1))
    path_gain = gain.unsqueeze(1) * path_gain * path_doppler_gain

    path_gain = path_gain.mean(-1, keepdim=True)
    n = n.unsqueeze(1).squeeze(-1).long().clamp_min(0).clamp_max(num_car - 1)
    path_gain = dft_mat[n] * path_gain
    return path_gain.sum(dim=2)


def _resolve_snapshot_ant_rotate(snapshot: dict, bs_runtime_by_id: dict[int, dict] | None = None):
    if "bs_orientation" in snapshot and snapshot["bs_orientation"] is not None:
        return [float(v) for v in snapshot["bs_orientation"]]
    if "bs_orientation_vector" in snapshot and snapshot["bs_orientation_vector"] is not None:
        return _direction_to_sionna_orientation(snapshot["bs_orientation_vector"])
    if bs_runtime_by_id is not None:
        bs_id = snapshot.get("bs_id")
        if bs_id is not None and int(bs_id) in bs_runtime_by_id:
            runtime = bs_runtime_by_id[int(bs_id)]
            if runtime.get("sionna_orientation") is not None:
                return [float(v) for v in runtime["sionna_orientation"]]
            if runtime.get("orientation") is not None:
                return _direction_to_sionna_orientation(runtime["orientation"])
    return None


def path_snapshot_to_channel(
    snapshot: dict,
    band: FixedBand,
    *,
    num_ant: int,
    ant_type: str = "rectangle",
    bs_ant: torch.Tensor | None = None,
) -> torch.Tensor:
    device = bs_ant.device if bs_ant is not None else torch.device("cpu")
    if bs_ant is None:
        bs_ant = get_bs_ant(num_ant=num_ant, ant_type=ant_type, device=device)

    gain_r = torch.as_tensor(snapshot["gain_r"], device=device, dtype=torch.float32)
    gain_i = torch.as_tensor(snapshot["gain_i"], device=device, dtype=torch.float32)
    delay = torch.as_tensor(snapshot["delay"], device=device, dtype=torch.float32)
    aod = torch.as_tensor(snapshot["AoD"], device=device, dtype=torch.float32)
    aoa = torch.as_tensor(snapshot["AoA"], device=device, dtype=torch.float32)

    if delay.numel() == 0 or bool(torch.all(delay < 0)):
        return torch.zeros((2, band.num_subcarriers, num_ant), device=device, dtype=torch.float32)

    gain = (gain_r + 1j * gain_i).unsqueeze(0)
    delay = delay.unsqueeze(0)
    aod = aod.unsqueeze(0)
    aoa = aoa.unsqueeze(0)
    velocity = torch.as_tensor(snapshot["velocity"], device=device, dtype=torch.float32).view(1, 1, 3)

    fs = float(band.subcarrier_spacing_hz) * float(band.num_subcarriers)
    dft_mat = get_dft_mat(band.num_subcarriers, device=device)
    channel_complex = _get_moving_channel_simplified(
        gain=gain,
        delay=delay,
        aod=aod,
        aoa=aoa,
        num_ant=num_ant,
        wc=float(band.center_freq_hz),
        fs=float(fs),
        bs_ant=bs_ant,
        num_car=band.num_subcarriers,
        dft_mat=dft_mat,
        velocity=velocity,
    )

    channel_complex = channel_complex.transpose(1, 2).unsqueeze(1)
    channel_real = torch.cat([channel_complex.real, channel_complex.imag], dim=1).squeeze(0)
    return channel_real.to(dtype=torch.float32)


def _build_bs_runtime_lookup(snapshot_payload: dict) -> dict[int, dict]:
    base_stations = snapshot_payload.get("runtime_base_stations") or snapshot_payload.get("base_stations") or []
    lookup: dict[int, dict] = {}
    for bs in base_stations:
        if not isinstance(bs, dict) or "bs_id" not in bs:
            continue
        lookup[int(bs["bs_id"])] = dict(bs)
    return lookup


def _generate_sample_from_path_sequence(
    sequence: list[dict],
    *,
    bands: tuple[FixedBand, ...] = DEFAULT_FIXED_BANDS,
    num_ant: int = 64,
    ant_type: str = "rectangle",
    bs_runtime_by_id: dict[int, dict] | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    dev = torch.device(device)
    bs_ant_cache: dict[tuple[int | None, tuple[float, float, float] | None], torch.Tensor] = {}
    per_t = []
    for snapshot in sequence:
        bs_id = int(snapshot["bs_id"]) if snapshot.get("bs_id") is not None else None
        ant_rotate = _resolve_snapshot_ant_rotate(snapshot, bs_runtime_by_id=bs_runtime_by_id)
        ant_rotate_key = None if ant_rotate is None else tuple(float(v) for v in ant_rotate)
        cache_key = (bs_id, ant_rotate_key)
        if cache_key not in bs_ant_cache:
            bs_ant_cache[cache_key] = get_bs_ant(
                num_ant=num_ant,
                ant_type=ant_type,
                ant_rotate=ant_rotate,
                device=dev,
            )
        bs_ant = bs_ant_cache[cache_key]

        per_band = []
        for band in bands:
            per_band.append(
                path_snapshot_to_channel(
                    snapshot,
                    band,
                    num_ant=num_ant,
                    ant_type=ant_type,
                    bs_ant=bs_ant,
                )
            )
        per_t.append(torch.stack(per_band, dim=0))
    sample = torch.stack(per_t, dim=0)
    return sample.detach().cpu().numpy().astype(np.float32, copy=False)


def _channel_gain_stats(sample: np.ndarray) -> dict[str, float]:
    sample64 = np.asarray(sample, dtype=np.float64)
    mag2 = np.square(sample64[:, :, 0, :, :]) + np.square(sample64[:, :, 1, :, :])
    mag = np.sqrt(mag2)
    return {
        "count": int(mag2.size),
        "magnitude_sum": float(mag.sum()),
        "magnitude_sq_sum": float(mag2.sum()),
        "magnitude_min": float(mag.min()),
        "magnitude_max": float(mag.max()),
        "power_sum": float(mag2.sum()),
        "power_sq_sum": float(np.square(mag2).sum()),
        "path_rms": float(np.sqrt(np.mean(sample64 ** 2))),
    }


def _merge_gain_stats(acc: dict[str, object], sample_stats: dict[str, float]) -> None:
    acc["count"] += int(sample_stats["count"])
    acc["magnitude_sum"] += float(sample_stats["magnitude_sum"])
    acc["magnitude_sq_sum"] += float(sample_stats["magnitude_sq_sum"])
    acc["power_sum"] += float(sample_stats["power_sum"])
    acc["power_sq_sum"] += float(sample_stats["power_sq_sum"])
    acc["magnitude_min"] = min(float(acc["magnitude_min"]), float(sample_stats["magnitude_min"]))
    acc["magnitude_max"] = max(float(acc["magnitude_max"]), float(sample_stats["magnitude_max"]))
    acc["path_rms_values"].append(float(sample_stats["path_rms"]))


def _finalize_gain_stats(acc: dict[str, object], *, kept_paths: int, dropped_zero_paths: int, dropped_low_power_paths: int) -> dict[str, object]:
    count = int(acc["count"])
    magnitude_mean = float(acc["magnitude_sum"] / count) if count else 0.0
    magnitude_var = float(acc["magnitude_sq_sum"] / count - magnitude_mean * magnitude_mean) if count else 0.0
    power_mean = float(acc["power_sum"] / count) if count else 0.0
    power_var = float(acc["power_sq_sum"] / count - power_mean * power_mean) if count else 0.0
    path_rms_values = np.asarray(acc["path_rms_values"], dtype=np.float64)
    if path_rms_values.size:
        path_rms = {
            "mean": float(path_rms_values.mean()),
            "std": float(path_rms_values.std()),
            "min": float(path_rms_values.min()),
            "max": float(path_rms_values.max()),
        }
    else:
        path_rms = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "kept_paths": int(kept_paths),
        "dropped_zero_paths": int(dropped_zero_paths),
        "dropped_low_power_paths": int(dropped_low_power_paths),
        "num_channel_elements": count,
        "magnitude": {
            "mean": magnitude_mean,
            "std": float(np.sqrt(max(0.0, magnitude_var))),
            "min": 0.0 if count == 0 else float(acc["magnitude_min"]),
            "max": float(acc["magnitude_max"]),
        },
        "power": {
            "mean": power_mean,
            "std": float(np.sqrt(max(0.0, power_var))),
        },
        "path_rms": path_rms,
    }


def _empty_gain_stats_accumulator() -> dict[str, object]:
    return {
        "count": 0,
        "magnitude_sum": 0.0,
        "magnitude_sq_sum": 0.0,
        "power_sum": 0.0,
        "power_sq_sum": 0.0,
        "magnitude_min": float("inf"),
        "magnitude_max": 0.0,
        "path_rms_values": [],
    }


def _point_powers(sample: np.ndarray) -> np.ndarray:
    sample64 = np.asarray(sample, dtype=np.float64)
    return (np.square(sample64[:, :, 0, :, :]) + np.square(sample64[:, :, 1, :, :])).mean(axis=(1, 2, 3))


def _sample_has_zero_points(sample: np.ndarray) -> tuple[bool, int]:
    point_power = _point_powers(sample)
    zero_count = int(np.count_nonzero(point_power == 0.0))
    return zero_count > 0, zero_count


def _sample_has_low_power_points(sample: np.ndarray, min_point_power: float) -> tuple[bool, int, float]:
    point_power = _point_powers(sample)
    low_count = int(np.count_nonzero(point_power < float(min_point_power)))
    return low_count > 0, low_count, float(point_power.min())


def save_dataset_npz(
    samples: list[np.ndarray],
    output_path: str | Path,
    *,
    coordinate_sequences: list[list[dict]] | None = None,
    dataset_metadata: dict | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"sample_gt_{i:06d}": np.asarray(s, dtype=np.float32) for i, s in enumerate(samples)}
    if coordinate_sequences is not None:
        for i, seq in enumerate(coordinate_sequences):
            payload[f"sample_xyz_{i:06d}"] = np.asarray([step["xyz"] for step in seq], dtype=np.float32)
            payload[f"sample_velocity_{i:06d}"] = np.asarray([step["velocity"] for step in seq], dtype=np.float32)
            payload[f"sample_bs_id_{i:06d}"] = np.asarray([step["bs_id"] for step in seq], dtype=np.int32)
    if dataset_metadata is not None:
        payload["dataset_config_json"] = np.asarray(json.dumps(dataset_metadata), dtype=np.str_)
    np.savez_compressed(output_path, **payload)


def save_dataset_npy_dir(
    samples: list[np.ndarray],
    output_dir: str | Path,
    *,
    coordinate_sequences: list[list[dict]] | None = None,
    dataset_metadata: dict | None = None,
    gain_report: dict | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "twm_npy_shards_v1",
        "scenario_name": str((dataset_metadata or {}).get("scenario_name", out_dir.parent.name)),
        "arrays": {},
    }
    for i, sample in enumerate(samples):
        suffix = f"{i:06d}"
        key = f"sample_gt_{suffix}"
        arr = np.asarray(sample, dtype=np.float32)
        np.save(out_dir / f"{key}.npy", arr, allow_pickle=False)
        manifest["arrays"][key] = {"file": f"{key}.npy", "shape": list(arr.shape), "dtype": str(arr.dtype)}
        if coordinate_sequences is not None:
            seq = coordinate_sequences[i]
            extra_arrays = {
                f"sample_xyz_{suffix}": np.asarray([step["xyz"] for step in seq], dtype=np.float32),
                f"sample_velocity_{suffix}": np.asarray([step["velocity"] for step in seq], dtype=np.float32),
                f"sample_bs_id_{suffix}": np.asarray([step["bs_id"] for step in seq], dtype=np.int32),
            }
            for extra_key, extra_arr in extra_arrays.items():
                np.save(out_dir / f"{extra_key}.npy", extra_arr, allow_pickle=False)
                manifest["arrays"][extra_key] = {
                    "file": f"{extra_key}.npy",
                    "shape": list(extra_arr.shape),
                    "dtype": str(extra_arr.dtype),
                }
    if dataset_metadata is not None:
        meta_arr = np.asarray(json.dumps(dataset_metadata), dtype=np.str_)
        np.save(out_dir / "dataset_config_json.npy", meta_arr, allow_pickle=False)
        manifest["arrays"]["dataset_config_json"] = {
            "file": "dataset_config_json.npy",
            "shape": list(meta_arr.shape),
            "dtype": str(meta_arr.dtype),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if gain_report is not None:
        (out_dir / "channel_gain_report.json").write_text(json.dumps(gain_report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return out_dir


def generate_and_save_dataset(
    *,
    snapshot_file: str,
    output_path: str | Path,
    bands: tuple[FixedBand, ...] = DEFAULT_FIXED_BANDS,
    num_ant: int = 64,
    ant_type: str = "rectangle",
    device: str | torch.device = "cpu",
    output_format: str = "auto",
    report_path: str | Path | None = None,
    min_point_power: float = 1e-10,
) -> Path:
    snapshot_payload = load_sionna_snapshot_file(snapshot_file)
    path_sequences = snapshot_payload["path_sequences"]
    coordinate_sequences = snapshot_payload.get("coordinate_sequences")
    bs_runtime_by_id = _build_bs_runtime_lookup(snapshot_payload)

    kept_samples: list[np.ndarray] = []
    kept_coordinate_sequences: list[list[dict]] | None = [] if coordinate_sequences is not None else None
    zero_dropped_indices: list[int] = []
    zero_dropped_counts: list[int] = []
    low_power_dropped_indices: list[int] = []
    low_power_dropped_counts: list[int] = []
    low_power_min_values: list[float] = []
    gain_acc = _empty_gain_stats_accumulator()

    for path_index, sequence in enumerate(path_sequences):
        sample = _generate_sample_from_path_sequence(
            sequence,
            bands=bands,
            num_ant=num_ant,
            ant_type=ant_type,
            bs_runtime_by_id=bs_runtime_by_id,
            device=device,
        )
        has_zero_points, zero_count = _sample_has_zero_points(sample)
        if has_zero_points:
            zero_dropped_indices.append(int(path_index))
            zero_dropped_counts.append(int(zero_count))
            continue
        has_low_power_points, low_power_count, min_power_value = _sample_has_low_power_points(sample, float(min_point_power))
        if has_low_power_points:
            low_power_dropped_indices.append(int(path_index))
            low_power_dropped_counts.append(int(low_power_count))
            low_power_min_values.append(float(min_power_value))
            continue
        kept_samples.append(sample)
        if kept_coordinate_sequences is not None and coordinate_sequences is not None:
            kept_coordinate_sequences.append(coordinate_sequences[path_index])
        _merge_gain_stats(gain_acc, _channel_gain_stats(sample))

    dataset_metadata = build_dataset_metadata(
        snapshot_payload=snapshot_payload,
        snapshot_file=snapshot_file,
        num_ant=num_ant,
        ant_type=ant_type,
        bands=[asdict(band) for band in bands],
    )
    dataset_metadata["zero_point_filter"] = {
        "input_paths": int(len(path_sequences)),
        "kept_paths": int(len(kept_samples)),
        "dropped_paths": int(len(zero_dropped_indices)),
        "dropped_path_indices": zero_dropped_indices,
        "dropped_zero_point_counts": zero_dropped_counts,
    }
    dataset_metadata["low_point_power_filter"] = {
        "threshold": float(min_point_power),
        "input_paths": int(len(path_sequences)),
        "kept_paths": int(len(kept_samples)),
        "dropped_paths": int(len(low_power_dropped_indices)),
        "dropped_path_indices": low_power_dropped_indices,
        "dropped_low_point_counts": low_power_dropped_counts,
        "dropped_min_point_powers": low_power_min_values,
    }
    gain_report = _finalize_gain_stats(
        gain_acc,
        kept_paths=len(kept_samples),
        dropped_zero_paths=len(zero_dropped_indices),
        dropped_low_power_paths=len(low_power_dropped_indices),
    )
    dataset_metadata["channel_gain_report"] = gain_report

    out = Path(output_path)
    resolved_format = str(output_format).lower()
    if resolved_format == "auto":
        resolved_format = "npz" if out.suffix.lower() == ".npz" else "npy_dir"
    if resolved_format == "npz":
        save_dataset_npz(
            kept_samples,
            out,
            coordinate_sequences=kept_coordinate_sequences,
            dataset_metadata=dataset_metadata,
        )
    elif resolved_format == "npy_dir":
        save_dataset_npy_dir(
            kept_samples,
            out,
            coordinate_sequences=kept_coordinate_sequences,
            dataset_metadata=dataset_metadata,
            gain_report=gain_report,
        )
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    if report_path is not None:
        report_out = Path(report_path)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(gain_report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed-band dataset from stored Sionna path snapshots.")
    parser.add_argument("--snapshot-file", required=True, help="Input Sionna snapshot file (.json or .json.gz).")
    parser.add_argument("--output-path", required=True, help="Output dataset path. Use a directory for npy shards or .npz for archive output.")
    parser.add_argument("--num-ant", type=int, default=64)
    parser.add_argument("--ant-type", default="rectangle", choices=["rectangle", "ULA", "ula"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-format", default="auto", choices=["auto", "npz", "npy_dir"])
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--min-point-power", type=float, default=1e-10)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    out = generate_and_save_dataset(
        snapshot_file=args.snapshot_file,
        output_path=args.output_path,
        num_ant=int(args.num_ant),
        ant_type=str(args.ant_type),
        device=args.device,
        output_format=str(args.output_format),
        report_path=args.report_path,
        min_point_power=float(args.min_point_power),
    )
    print(str(out))


if __name__ == "__main__":
    main()
