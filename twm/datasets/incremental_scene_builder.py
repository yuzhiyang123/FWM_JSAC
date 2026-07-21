from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .scene_store import (
    create_scene,
    load_scene_config,
    load_xyz_paths,
    save_xyz_paths,
    update_base_stations,
    write_base_station_template,
)


DT_SECONDS = 0.05
XY_NOISE_STD = 0.05


def _load_builder_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"builder config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "raw_data_dir" not in data:
        raise ValueError("builder config must contain 'raw_data_dir'.")
    return data


def _ensure_bs_exists(raw_data_dir: str | Path, scenario_name: str, bs_id: int) -> None:
    scene_config, _ = load_scene_config(raw_data_dir, scenario_name)
    valid_ids = {int(bs["bs_id"]) for bs in scene_config["base_stations"]}
    if int(bs_id) not in valid_ids:
        raise KeyError(f"Unknown bs_id={bs_id} for scenario '{scenario_name}'. Known ids: {sorted(valid_ids)}")


def _build_linear_steps(
    *,
    start_xyz: list[float],
    end_xyz: list[float],
    num_points: int,
    bs_id: int,
    noise_std_xy: float = XY_NOISE_STD,
    dt_seconds: float = DT_SECONDS,
    rng: np.random.Generator,
) -> list[dict]:
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    start = np.asarray(start_xyz, dtype=np.float32)
    end = np.asarray(end_xyz, dtype=np.float32)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("start_xyz and end_xyz must each have shape [3].")

    base_points = np.linspace(start, end, num=num_points, endpoint=True, dtype=np.float32)
    if noise_std_xy > 0:
        noise_xy = rng.normal(loc=0.0, scale=float(noise_std_xy), size=(num_points, 2)).astype(np.float32)
        base_points[:, :2] += noise_xy

    velocity = (end - start) / float((num_points - 1) * dt_seconds)
    velocity = velocity.astype(np.float32)
    return [
        {
            "xyz": point.tolist(),
            "velocity": velocity.tolist(),
            "bs_id": int(bs_id),
        }
        for point in base_points
    ]


def initialize_scene(
    *,
    builder_config_path: str | Path,
    scenario_name: str,
    xml_filename: str | None = None,
) -> dict:
    builder_config = _load_builder_config(builder_config_path)
    return create_scene(
        raw_data_dir=builder_config["raw_data_dir"],
        scenario_name=scenario_name,
        xml_filename=xml_filename,
    )


def create_bs_template_file(output_path: str | Path, *, example_count: int = 2) -> Path:
    return write_base_station_template(output_path, example_count=example_count)


def set_scene_base_stations(
    *,
    builder_config_path: str | Path,
    scenario_name: str,
    base_stations_file: str | Path,
) -> Path:
    builder_config = _load_builder_config(builder_config_path)
    with open(base_stations_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "base_stations" not in payload:
        raise ValueError("base station file must contain 'base_stations'.")
    return update_base_stations(
        raw_data_dir=builder_config["raw_data_dir"],
        scenario_name=scenario_name,
        base_stations=payload["base_stations"],
    )


def append_linear_path(
    *,
    builder_config_path: str | Path,
    scenario_name: str,
    bs_id: int,
    start_xyz: list[float],
    end_xyz: list[float],
    num_points: int,
    noise_std_xy: float = XY_NOISE_STD,
    dt_seconds: float = DT_SECONDS,
    random_seed: int | None = None,
) -> Path:
    builder_config = _load_builder_config(builder_config_path)
    raw_data_dir = builder_config["raw_data_dir"]
    _ensure_bs_exists(raw_data_dir, scenario_name, int(bs_id))

    _, scene_paths = load_scene_config(raw_data_dir, scenario_name)
    payload = load_xyz_paths(scene_paths.xyz_paths_path)
    rng = np.random.default_rng(random_seed)
    steps = _build_linear_steps(
        start_xyz=start_xyz,
        end_xyz=end_xyz,
        num_points=int(num_points),
        bs_id=int(bs_id),
        noise_std_xy=float(noise_std_xy),
        dt_seconds=float(dt_seconds),
        rng=rng,
    )
    payload["paths"].append(
        {
            "bs_id": int(bs_id),
            "steps": steps,
        }
    )
    return save_xyz_paths(scene_paths.xyz_paths_path, payload)


def list_paths(
    *,
    builder_config_path: str | Path,
    scenario_name: str,
) -> list[dict]:
    builder_config = _load_builder_config(builder_config_path)
    raw_data_dir = builder_config["raw_data_dir"]
    _, scene_paths = load_scene_config(raw_data_dir, scenario_name)
    payload = load_xyz_paths(scene_paths.xyz_paths_path)

    summaries: list[dict] = []
    for idx, path_item in enumerate(payload.get("paths", [])):
        steps = path_item.get("steps", [])
        if steps:
            start_xyz = steps[0].get("xyz")
            end_xyz = steps[-1].get("xyz")
        else:
            start_xyz = None
            end_xyz = None
        summaries.append(
            {
                "path_index": idx,
                "bs_id": int(path_item.get("bs_id", -1)),
                "num_points": len(steps),
                "start_xyz": start_xyz,
                "end_xyz": end_xyz,
            }
        )
    return summaries


def remove_last_path(
    *,
    builder_config_path: str | Path,
    scenario_name: str,
) -> Path:
    builder_config = _load_builder_config(builder_config_path)
    raw_data_dir = builder_config["raw_data_dir"]
    _, scene_paths = load_scene_config(raw_data_dir, scenario_name)
    payload = load_xyz_paths(scene_paths.xyz_paths_path)
    paths = payload.get("paths", [])
    if not paths:
        raise ValueError(f"No paths available to remove in scenario '{scenario_name}'.")
    paths.pop()
    payload["paths"] = paths
    return save_xyz_paths(scene_paths.xyz_paths_path, payload)


def remove_path(
    *,
    builder_config_path: str | Path,
    scenario_name: str,
    path_index: int,
) -> Path:
    builder_config = _load_builder_config(builder_config_path)
    raw_data_dir = builder_config["raw_data_dir"]
    _, scene_paths = load_scene_config(raw_data_dir, scenario_name)
    payload = load_xyz_paths(scene_paths.xyz_paths_path)
    paths = payload.get("paths", [])
    idx = int(path_index)
    if idx < 0 or idx >= len(paths):
        raise IndexError(
            f"path_index out of range for scenario '{scenario_name}': {idx}. "
            f"Valid range is [0, {max(0, len(paths) - 1)}]."
        )
    del paths[idx]
    payload["paths"] = paths
    return save_xyz_paths(scene_paths.xyz_paths_path, payload)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incrementally build scene configs and xyz path files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-scene", help="Create scene folder and initialize files.")
    init_parser.add_argument("--builder-config", required=True)
    init_parser.add_argument("--scenario-name", required=True)
    init_parser.add_argument("--xml-filename", default=None)

    template_parser = subparsers.add_parser("make-bs-template", help="Write a base station template json file.")
    template_parser.add_argument("--output-path", required=True)
    template_parser.add_argument("--example-count", type=int, default=2)

    set_bs_parser = subparsers.add_parser("set-base-stations", help="Write base stations into scene config.")
    set_bs_parser.add_argument("--builder-config", required=True)
    set_bs_parser.add_argument("--scenario-name", required=True)
    set_bs_parser.add_argument("--base-stations-file", required=True)

    append_parser = subparsers.add_parser("append-line-path", help="Append one noisy linear path to xyz_paths.json.")
    append_parser.add_argument("--builder-config", required=True)
    append_parser.add_argument("--scenario-name", required=True)
    append_parser.add_argument("--bs-id", type=int, required=True)
    append_parser.add_argument("--start-x", type=float, required=True)
    append_parser.add_argument("--start-y", type=float, required=True)
    append_parser.add_argument("--start-z", type=float, required=True)
    append_parser.add_argument("--end-x", type=float, required=True)
    append_parser.add_argument("--end-y", type=float, required=True)
    append_parser.add_argument("--end-z", type=float, required=True)
    append_parser.add_argument("--num-points", type=int, required=True)
    append_parser.add_argument("--noise-std-xy", type=float, default=XY_NOISE_STD)
    append_parser.add_argument("--dt-seconds", type=float, default=DT_SECONDS)
    append_parser.add_argument("--random-seed", type=int, default=None)

    list_parser = subparsers.add_parser("list-paths", help="List path summaries from xyz_paths.json.")
    list_parser.add_argument("--builder-config", required=True)
    list_parser.add_argument("--scenario-name", required=True)

    remove_last_parser = subparsers.add_parser("remove-last-path", help="Remove the last appended path.")
    remove_last_parser.add_argument("--builder-config", required=True)
    remove_last_parser.add_argument("--scenario-name", required=True)

    remove_parser = subparsers.add_parser("remove-path", help="Remove one path by index.")
    remove_parser.add_argument("--builder-config", required=True)
    remove_parser.add_argument("--scenario-name", required=True)
    remove_parser.add_argument("--path-index", type=int, required=True)

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "init-scene":
        result = initialize_scene(
            builder_config_path=args.builder_config,
            scenario_name=args.scenario_name,
            xml_filename=args.xml_filename,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return

    if args.command == "make-bs-template":
        out = create_bs_template_file(args.output_path, example_count=int(args.example_count))
        print(str(out))
        return

    if args.command == "set-base-stations":
        out = set_scene_base_stations(
            builder_config_path=args.builder_config,
            scenario_name=args.scenario_name,
            base_stations_file=args.base_stations_file,
        )
        print(str(out))
        return

    if args.command == "append-line-path":
        out = append_linear_path(
            builder_config_path=args.builder_config,
            scenario_name=args.scenario_name,
            bs_id=int(args.bs_id),
            start_xyz=[args.start_x, args.start_y, args.start_z],
            end_xyz=[args.end_x, args.end_y, args.end_z],
            num_points=int(args.num_points),
            noise_std_xy=float(args.noise_std_xy),
            dt_seconds=float(args.dt_seconds),
            random_seed=args.random_seed,
        )
        print(str(out))
        return

    if args.command == "list-paths":
        items = list_paths(
            builder_config_path=args.builder_config,
            scenario_name=args.scenario_name,
        )
        print(json.dumps(items, ensure_ascii=True, indent=2))
        return

    if args.command == "remove-last-path":
        out = remove_last_path(
            builder_config_path=args.builder_config,
            scenario_name=args.scenario_name,
        )
        print(str(out))
        return

    if args.command == "remove-path":
        out = remove_path(
            builder_config_path=args.builder_config,
            scenario_name=args.scenario_name,
            path_index=int(args.path_index),
        )
        print(str(out))
        return

    raise NotImplementedError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
