from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json
import pickle
from pathlib import Path
import zipfile

import numpy as np
import torch
from sionna.rt import PlanarArray, PathSolver, Receiver, Transmitter, load_scene


SCENE_CONFIG_FILENAME = "scene_config.json"
XYZ_PATHS_FILENAME = "xyz_paths.json"


@dataclass(frozen=True)
class ScenePaths:
    root_dir: Path
    scene_dir: Path
    xml_path: Path
    scene_config_path: Path
    xyz_paths_path: Path


def resolve_scene_paths(raw_data_dir: str | Path, scenario_name: str, *, xml_filename: str | None = None) -> ScenePaths:
    root_dir = Path(raw_data_dir)
    scene_dir = root_dir / scenario_name
    xml_name = xml_filename or f"{scenario_name}.xml"
    return ScenePaths(
        root_dir=root_dir,
        scene_dir=scene_dir,
        xml_path=scene_dir / xml_name,
        scene_config_path=scene_dir / SCENE_CONFIG_FILENAME,
        xyz_paths_path=scene_dir / XYZ_PATHS_FILENAME,
    )


def _validate_base_station(bs: dict, idx: int) -> dict:
    if not isinstance(bs, dict):
        raise ValueError(f"base_stations[{idx}] must be a dict.")
    for key in ("bs_id", "position", "orientation"):
        if key not in bs:
            raise ValueError(f"base_stations[{idx}] must contain '{key}'.")
    bs_id = int(bs["bs_id"])
    position = [float(v) for v in bs["position"]]
    orientation = [float(v) for v in bs["orientation"]]
    if len(position) != 3:
        raise ValueError(f"base_stations[{idx}]['position'] must have length 3.")
    if len(orientation) != 3:
        raise ValueError(f"base_stations[{idx}]['orientation'] must have length 3.")
    norm = sum(v * v for v in orientation) ** 0.5
    if norm <= 0.0:
        raise ValueError(f"base_stations[{idx}]['orientation'] must be a non-zero xyz direction vector.")
    orientation = [v / norm for v in orientation]
    return {
        "bs_id": bs_id,
        "name": str(bs.get("name", f"bs_{bs_id}")),
        "position": position,
        "orientation": orientation,
    }


def validate_scene_config(data: dict, *, allow_empty_base_stations: bool = False) -> dict:
    if not isinstance(data, dict):
        raise ValueError("scene config must be a dict.")
    if "scenario_name" not in data:
        raise ValueError("scene config must contain 'scenario_name'.")
    if "xml_filename" not in data:
        raise ValueError("scene config must contain 'xml_filename'.")
    if "base_stations" not in data:
        raise ValueError("scene config must contain 'base_stations'.")

    base_stations = data["base_stations"]
    if not isinstance(base_stations, list):
        raise ValueError("'base_stations' must be a list.")
    if not base_stations and not allow_empty_base_stations:
        raise ValueError("'base_stations' must be a non-empty list.")

    normalized_bs = [_validate_base_station(bs, idx) for idx, bs in enumerate(base_stations)]
    bs_ids = [bs["bs_id"] for bs in normalized_bs]
    if len(bs_ids) != len(set(bs_ids)):
        raise ValueError("base station ids must be unique.")

    return {
        "scenario_name": str(data["scenario_name"]),
        "scene_name": str(data.get("scene_name", data["scenario_name"])),
        "xml_filename": str(data["xml_filename"]),
        "base_stations": normalized_bs,
    }


def load_scene_config(raw_data_dir: str | Path, scenario_name: str) -> tuple[dict, ScenePaths]:
    scene_paths = resolve_scene_paths(raw_data_dir, scenario_name)
    if not scene_paths.scene_config_path.exists():
        raise FileNotFoundError(f"scene config not found: {scene_paths.scene_config_path}")
    with open(scene_paths.scene_config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = validate_scene_config(data, allow_empty_base_stations=True)
    if config["scenario_name"] != scenario_name:
        raise ValueError(
            f"scene config scenario_name mismatch: expected '{scenario_name}', got '{config['scenario_name']}'."
        )
    return config, resolve_scene_paths(raw_data_dir, scenario_name, xml_filename=config["xml_filename"])


@dataclass
class SionnaDataConfig:
    raw_data_dir: str
    scenario_name: str
    xyz_paths_file: str | None = None
    batch_size: int = 1024
    max_depth: int = 5


def _parse_coord_with_velocity(coord: object) -> tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(coord, dict):
        raise ValueError("coord must be dict with keys 'xyz', 'velocity', and 'bs_id'.")
    if "xyz" not in coord or "velocity" not in coord or "bs_id" not in coord:
        raise ValueError("coord must contain 'xyz', 'velocity', and 'bs_id'.")

    xyz = np.asarray(coord["xyz"], dtype=np.float32)
    velocity = np.asarray(coord["velocity"], dtype=np.float32)
    try:
        bs_id = int(coord["bs_id"])
    except (TypeError, ValueError) as e:
        raise ValueError("coord['bs_id'] must be an integer-like value.") from e
    if xyz.shape != (3,):
        raise ValueError(f"coord['xyz'] must have shape [3], got {xyz.shape}")
    if velocity.shape != (3,):
        raise ValueError(f"coord['velocity'] must have shape [3], got {velocity.shape}")
    return xyz, velocity, bs_id


def _normalize_path_bs_id(path_item: dict, path_idx: int) -> int:
    if "bs_id" not in path_item:
        raise ValueError(f"paths[{path_idx}] must contain 'bs_id'.")
    try:
        return int(path_item["bs_id"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"paths[{path_idx}]['bs_id'] must be an integer-like value.") from e


def _load_plain_torch_archive(path: Path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        data_pkl = next((name for name in names if name.endswith('data.pkl')), None)
        if data_pkl is None:
            raise ValueError(f"Invalid torch archive, missing data.pkl: {path}")
        return pickle.loads(archive.read(data_pkl))


def _load_xyz_paths(xyz_paths_file: str) -> list[dict]:
    path = Path(xyz_paths_file)
    if not path.exists():
        raise FileNotFoundError(f"xyz paths file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
    elif suffix in {'.pt', '.pth'}:
        data = _load_plain_torch_archive(path)
    else:
        raise ValueError(f"Unsupported xyz paths file format: {path.suffix}. Supported: .json/.pt/.pth")

    if not isinstance(data, dict) or "paths" not in data:
        raise ValueError("xyz paths file must be a dict with key 'paths'.")

    raw_paths = data["paths"]
    if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
        raise ValueError("'paths' must be a non-empty list.")

    parsed_paths: list[dict] = []
    for path_idx, path_item in enumerate(raw_paths):
        if not isinstance(path_item, dict) or "steps" not in path_item:
            raise ValueError(f"paths[{path_idx}] must be a dict with key 'steps'.")

        bs_id = _normalize_path_bs_id(path_item, path_idx)
        raw_steps = path_item["steps"]
        if not isinstance(raw_steps, (list, tuple)) or not raw_steps:
            raise ValueError(f"paths[{path_idx}]['steps'] must be a non-empty list.")

        parsed_steps: list[dict] = []
        for step_idx, step_item in enumerate(raw_steps):
            if not isinstance(step_item, dict):
                raise ValueError(f"paths[{path_idx}]['steps'][{step_idx}] must be a dict.")
            if "xyz" not in step_item or "velocity" not in step_item:
                raise ValueError(f"paths[{path_idx}]['steps'][{step_idx}] must contain 'xyz' and 'velocity'.")

            xyz = np.asarray(step_item["xyz"], dtype=np.float32)
            velocity = np.asarray(step_item["velocity"], dtype=np.float32)
            if xyz.shape != (3,):
                raise ValueError(f"paths[{path_idx}]['steps'][{step_idx}]['xyz'] must have shape [3].")
            if velocity.shape != (3,):
                raise ValueError(f"paths[{path_idx}]['steps'][{step_idx}]['velocity'] must have shape [3].")

            parsed_steps.append({"xyz": xyz, "velocity": velocity, "bs_id": bs_id})

        parsed_paths.append({"bs_id": bs_id, "steps": parsed_steps})

    return parsed_paths


def _coordinate_transform(theta, phi):
    theta = torch.as_tensor(theta)
    phi = torch.as_tensor(phi)
    sin_theta = torch.sin(theta)
    x = torch.cos(phi) * sin_theta
    y = torch.sin(phi) * sin_theta
    z = torch.cos(theta)
    return x, y, z


def _to_numpy(value, dtype=np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


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


def _build_scene(xml_file: str, bs_config: dict):
    scene = load_scene(xml_file)
    scene.frequency = 3.50e9
    sionna_orientation = bs_config.get("sionna_orientation")
    if sionna_orientation is None:
        sionna_orientation = _direction_to_sionna_orientation(bs_config["orientation"])
    scene.add(
        Transmitter(
            name="tx",
            position=bs_config["position"],
            orientation=sionna_orientation,
        )
    )
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.synthetic_array = True
    return scene


def _load_scene_runtime(raw_data_dir: str | Path, scenario_name: str) -> tuple[dict, Path, dict[int, dict]]:
    params, scene_paths = load_scene_config(raw_data_dir, scenario_name)
    base_stations = []
    for bs in params["base_stations"]:
        bs_runtime = dict(bs)
        bs_runtime["orientation_vector"] = list(bs["orientation"])
        bs_runtime["sionna_orientation"] = _direction_to_sionna_orientation(bs["orientation"])
        base_stations.append(bs_runtime)
    bs_lookup = {int(bs["bs_id"]): bs for bs in base_stations}
    if not bs_lookup:
        raise ValueError(f"scenario '{scenario_name}' defines no base stations.")
    if not scene_paths.xml_path.exists():
        raise FileNotFoundError(f"scene xml file not found: {scene_paths.xml_path}")
    return params, scene_paths.xml_path, bs_lookup


def _get_channel_from_xyz_batch(solver, scene, xyz_array: np.ndarray, *, max_depth: int = 5) -> list[dict]:
    n = len(xyz_array)
    rx_names = []
    for i in range(n):
        rx_name = f"rx-{i}"
        scene.add(Receiver(name=rx_name, position=xyz_array[i].tolist()))
        rx_names.append(rx_name)

    paths = solver(
        scene,
        max_depth=int(max_depth),
        diffuse_reflection=True,
        diffraction=True,
        edge_diffraction=True,
    )
    gain_r, gain_i = paths.a
    theta_r, phi_r, theta_t, phi_t = paths.theta_r, paths.phi_r, paths.theta_t, paths.phi_t
    delay = paths.tau

    x_r, y_r, z_r = _coordinate_transform(theta_r, phi_r)
    x_t, y_t, z_t = _coordinate_transform(theta_t, phi_t)
    aoa = _to_numpy(torch.stack([x_r, y_r, z_r], dim=-1)).reshape(n, -1, 3)
    aod = _to_numpy(torch.stack([x_t, y_t, z_t], dim=-1)).reshape(n, -1, 3)
    delay = _to_numpy(delay).reshape(n, -1, 1)
    gain_r = _to_numpy(gain_r).reshape(n, -1, 1)
    gain_i = _to_numpy(gain_i).reshape(n, -1, 1)

    snapshots = []
    for i in range(n):
        snapshots.append(
            {
                "gain_r": gain_r[i].tolist(),
                "gain_i": gain_i[i].tolist(),
                "delay": delay[i].tolist(),
                "AoD": aod[i].tolist(),
                "AoA": aoa[i].tolist(),
            }
        )

    for name in rx_names:
        rx = scene.receivers.get(name)
        scene.remove(name)
        del rx
    return snapshots


def generate_sionna_snapshots(config: SionnaDataConfig) -> dict:
    scenario_name = config.scenario_name
    params, xml_path, bs_lookup = _load_scene_runtime(config.raw_data_dir, scenario_name)
    xyz_paths_file = config.xyz_paths_file
    if not xyz_paths_file:
        xyz_paths_file = str(resolve_scene_paths(config.raw_data_dir, scenario_name).xyz_paths_path)
    raw_paths = _load_xyz_paths(xyz_paths_file)
    xml_file = str(xml_path)

    runtime_by_bs = {
        bs_id: {
            "scene": _build_scene(xml_file, bs_cfg),
            "solver": PathSolver(),
            "bs": bs_cfg,
        }
        for bs_id, bs_cfg in bs_lookup.items()
    }

    path_sequences: list[list[dict]] = [[] for _ in raw_paths]
    flattened_xyz: list[np.ndarray] = []
    flattened_velocity: list[np.ndarray] = []
    flattened_path_idx: list[int] = []
    flattened_bs_id: list[int] = []

    for path_idx, path_item in enumerate(raw_paths):
        bs_id = int(path_item["bs_id"])
        if bs_id not in runtime_by_bs:
            raise KeyError(f"paths[{path_idx}] references unknown bs_id={bs_id} for scenario '{scenario_name}'.")
        for step in path_item["steps"]:
            flattened_xyz.append(step["xyz"])
            flattened_velocity.append(step["velocity"])
            flattened_path_idx.append(path_idx)
            flattened_bs_id.append(bs_id)

    if not flattened_xyz:
        raise ValueError("No xyz samples found in xyz paths file.")

    batch_size = int(config.batch_size)
    indices_by_bs: dict[int, list[int]] = {}
    for global_idx, bs_id in enumerate(flattened_bs_id):
        indices_by_bs.setdefault(bs_id, []).append(global_idx)

    for bs_id, indices in indices_by_bs.items():
        runtime = runtime_by_bs[bs_id]
        cursor = 0
        while cursor < len(indices):
            batch_indices = indices[cursor : cursor + batch_size]
            xyz_batch = np.stack([flattened_xyz[i] for i in batch_indices], axis=0)
            snapshots = _get_channel_from_xyz_batch(
                runtime["solver"],
                runtime["scene"],
                xyz_batch,
                max_depth=int(config.max_depth),
            )

            for local_idx, snapshot in enumerate(snapshots):
                global_idx = batch_indices[local_idx]
                snapshot["velocity"] = flattened_velocity[global_idx].astype(np.float32).tolist()
                snapshot["scenario"] = params["scenario_name"]
                snapshot["scene_name"] = params.get("scene_name", params["scenario_name"])
                snapshot["bs_id"] = bs_id
                snapshot["bs_name"] = runtime["bs"].get("name", f"bs_{bs_id}")
                snapshot["bs_position"] = list(runtime["bs"]["position"])
                snapshot["bs_orientation_vector"] = list(runtime["bs"]["orientation_vector"])
                snapshot["bs_orientation"] = list(runtime["bs"]["sionna_orientation"])
                path_sequences[flattened_path_idx[global_idx]].append(snapshot)
            cursor += batch_size

    coordinate_sequences = [
        [
            {
                "xyz": step["xyz"].astype(np.float32).tolist(),
                "velocity": step["velocity"].astype(np.float32).tolist(),
                "bs_id": int(path_item["bs_id"]),
            }
            for step in path_item["steps"]
        ]
        for path_item in raw_paths
    ]
    return {
        "format_version": 1,
        "generator": "twm.datasets.data_gen_sionna",
        "scenario_name": scenario_name,
        "raw_data_dir": config.raw_data_dir,
        "xyz_paths_file": xyz_paths_file,
        "batch_size": int(config.batch_size),
        "max_depth": int(config.max_depth),
        "path_sequences": path_sequences,
        "coordinate_sequences": coordinate_sequences,
        "base_stations": [dict(bs) for bs in params["base_stations"]],
        "runtime_base_stations": [dict(bs) for bs in bs_lookup.values()],
        "scene_name": params.get("scene_name", params["scenario_name"]),
        "scenario_data": json.loads(json.dumps(params)),
    }


def save_sionna_snapshot_file(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True)
    if path.suffix.lower() == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(text)
    else:
        path.write_text(text, encoding='utf-8')
    return path


def load_sionna_snapshot_file(snapshot_file: str | Path) -> dict:
    path = Path(snapshot_file)
    if not path.exists():
        raise FileNotFoundError(f"Sionna snapshot file not found: {path}")
    if path.suffix.lower() == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def generate_and_save_sionna_snapshots(
    *,
    raw_data_dir: str,
    scenario_name: str,
    xyz_paths_file: str | None,
    output_path: str | Path,
    batch_size: int = 1024,
    max_depth: int = 5,
) -> Path:
    payload = generate_sionna_snapshots(
        SionnaDataConfig(
            raw_data_dir=raw_data_dir,
            scenario_name=scenario_name,
            xyz_paths_file=xyz_paths_file,
            batch_size=batch_size,
            max_depth=max_depth,
        )
    )
    return save_sionna_snapshot_file(payload, output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and save Sionna path snapshots using PyTorch tensors.")
    parser.add_argument("--raw-data-dir", required=True, help="Root directory containing scenario folders.")
    parser.add_argument("--scenario-name", required=True, help="Scenario folder name under raw-data-dir.")
    parser.add_argument("--xyz-paths-file", default=None, help="Optional override path to xyz_paths (.json/.pt/.pth).")
    parser.add_argument("--output-path", required=True, help="Output snapshot file path (.json or .json.gz).")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum Sionna path depth / bounce count.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    out = generate_and_save_sionna_snapshots(
        raw_data_dir=args.raw_data_dir,
        scenario_name=args.scenario_name,
        xyz_paths_file=args.xyz_paths_file,
        output_path=args.output_path,
        batch_size=int(args.batch_size),
        max_depth=int(args.max_depth),
    )
    print(str(out))


if __name__ == "__main__":
    main()
