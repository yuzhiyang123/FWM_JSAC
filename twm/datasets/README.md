# `twm/datasets`

## English

### Purpose

`twm/datasets` contains the dataset-side code used by this repository.
It supports the full data pipeline behind the experiments, including:

- raw channel-data generation,
- dataset conversion between storage formats,
- dataset splitting and subset construction,
- sequential and independent dataset loading for training,
- sharded scene storage and access utilities.

In the current project, this directory is the main entry point for preparing and loading channel datasets before JEPA and downstream training.

The repository also includes a minimal runnable raw-scene example under:

- `example_data/01chicago`

This example provides the raw files expected by the dataset-generation code:

- `scene_config.json`
- `xyz_paths.json`
- `01chicago.xml`
- `meshes/`

### Main components

- `data_gen_sionna.py`
  Generates channel datasets from the Sionna-based simulation pipeline.

- `fixed_band_dataset.py`
  Builds fixed-band datasets used by later training and evaluation code.

- `split_snapshot_dataset.py`
  Splits large snapshot datasets into smaller reusable pieces.

- `split_gt_dataset.py`
  Splits ground-truth datasets into train/test style subsets.

- `convert_npz_to_npy.py`
  Converts `.npz` archives into `.npy` layout when later stages require that format.

- `build_bs1_trial_subset.py`
  Creates reduced trial subsets for debugging or smaller-scale experiments.

- `get_sequential_data.py`
  Provides the sequential time-band dataset and dataloader builders used by JEPA-style temporal training.

- `get_simple_data.py`
  Provides simpler independent-sample dataset builders.

- `shard_access.py`
  Provides access helpers for sharded dataset storage.

- `scene_store.py`
  Defines scene-level storage helpers.

- `incremental_scene_builder.py`
  Supports incremental construction of scene datasets.

### Exported interface

`__init__.py` exposes dataset builder functions used by the rest of the codebase.
In practice, this directory provides the dataset-loading entry points for:

- sequential time-band channel data,
- independent channel data,
- other task-specific dataset families referenced by the training code.

---

## 中文

### 作用

`twm/datasets` 包含本仓库使用的数据侧代码。
它支撑了实验所需的完整数据流程，包括：

- 原始信道数据生成；
- 不同存储格式之间的数据转换；
- 数据集切分与子集构造；
- 面向训练的顺序式与独立式数据加载；
- shard 化场景数据的存储与访问工具。

在当前项目中，这个目录是 JEPA 和 downstream 训练前进行数据准备与加载的主要入口。

仓库中还附带了一份最小可运行原始场景示例，位置在：

- `example_data/01chicago`

该示例提供了数据生成代码所需的原始文件：

- `scene_config.json`
- `xyz_paths.json`
- `01chicago.xml`
- `meshes/`

### 主要组成

- `data_gen_sionna.py`
  基于 Sionna 仿真流程生成信道数据集。

- `fixed_band_dataset.py`
  构建后续训练和评估使用的固定 band 数据集。

- `split_snapshot_dataset.py`
  将较大的 snapshot 数据集切分成更小、可复用的部分。

- `split_gt_dataset.py`
  将 ground-truth 数据集切分成类似 train/test 的子集。

- `convert_npz_to_npy.py`
  在后续流程需要时，将 `.npz` 数据转换成 `.npy` 形式。

- `build_bs1_trial_subset.py`
  构造较小的 trial 子集，用于调试或小规模实验。

- `get_sequential_data.py`
  提供 JEPA 类时序训练使用的 sequential time-band dataset 与 dataloader 构建逻辑。

- `get_simple_data.py`
  提供更简单的独立样本式数据集构建逻辑。

- `shard_access.py`
  提供 shard 化数据存储的访问辅助函数。

- `scene_store.py`
  定义 scene 级别的数据存储辅助逻辑。

- `incremental_scene_builder.py`
  支持以增量方式构建 scene 数据集。

### 导出接口

`__init__.py` 为代码库其他部分导出 dataset builder 函数。
实际使用中，这个目录主要提供以下数据加载入口：

- sequential time-band channel data；
- independent channel data；
- 训练代码中引用的其他任务型数据接口。
