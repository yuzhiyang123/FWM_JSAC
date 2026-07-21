from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path



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


def save_scene_config(raw_data_dir: str | Path, scenario_name: str, config: dict) -> Path:
    config = validate_scene_config(config, allow_empty_base_stations=True)
    scene_paths = resolve_scene_paths(raw_data_dir, scenario_name, xml_filename=config["xml_filename"])
    scene_paths.scene_dir.mkdir(parents=True, exist_ok=True)
    with open(scene_paths.scene_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=True, indent=2)
    return scene_paths.scene_config_path


def create_scene(
    *,
    raw_data_dir: str | Path,
    scenario_name: str,
    xml_filename: str | None = None,
) -> dict:
    scene_paths = resolve_scene_paths(raw_data_dir, scenario_name, xml_filename=xml_filename)
    scene_paths.scene_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "scenario_name": scenario_name,
        "scene_name": scenario_name,
        "xml_filename": scene_paths.xml_path.name,
        "base_stations": [],
    }
    save_scene_config(raw_data_dir, scenario_name, config)

    if not scene_paths.xyz_paths_path.exists():
        scene_paths.xyz_paths_path.write_text(json.dumps({"paths": []}, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "scene_dir": str(scene_paths.scene_dir),
        "xml_path": str(scene_paths.xml_path),
        "scene_config_path": str(scene_paths.scene_config_path),
        "xyz_paths_path": str(scene_paths.xyz_paths_path),
    }


def write_base_station_template(
    output_path: str | Path,
    *,
    example_count: int = 2,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "base_stations": [
            {
                "bs_id": idx + 1,
                "name": f"bs_{idx + 1}",
                "position": [0.0, 0.0, 30.0],
                "orientation": [1.0, 0.0, 0.0],
            }
            for idx in range(max(1, int(example_count)))
        ]
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=True, indent=2)
    return output


def update_base_stations(
    *,
    raw_data_dir: str | Path,
    scenario_name: str,
    base_stations: list[dict],
) -> Path:
    config, _ = load_scene_config(raw_data_dir, scenario_name)
    config["base_stations"] = base_stations
    return save_scene_config(raw_data_dir, scenario_name, config)


def load_xyz_paths(xyz_paths_file: str | Path) -> dict:
    path = Path(xyz_paths_file)
    if not path.exists():
        return {"paths": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "paths" not in data or not isinstance(data["paths"], list):
        raise ValueError(f"Invalid xyz paths file structure: {path}")
    return data


def save_xyz_paths(xyz_paths_file: str | Path, payload: dict) -> Path:
    path = Path(xyz_paths_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return path
