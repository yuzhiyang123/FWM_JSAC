# `example_data`

## English

### Purpose

`example_data` contains a minimal raw-scene example that is sufficient to run the dataset-generation pipeline shipped with this repository.

The included example is:

- `01chicago/`

It contains the raw files required by the code:

- `scene_config.json`
- `xyz_paths.json`
- `01chicago.xml`
- `meshes/`

These files are used by:

- `tools/slurm_dataset_gen_01chicago_gt_only.sh`
- `twm.datasets.data_gen_sionna`

### Default usage

The dataset-generation Slurm script now defaults to:

- `RAW_DATA_ROOT=$REPO_ROOT/example_data`
- `RUN_ROOT=$REPO_ROOT/example_runs`

so the bundled `01chicago` example can be used directly.

### Note

The repository includes the raw scene files needed by the code, but running the generator still requires the external Sionna / Dr.Jit runtime environment used by `twm.datasets.data_gen_sionna`.

---

## 中文

### 作用

`example_data` 提供了一份最小原始场景示例，使仓库自带的数据生成流程可以直接运行。

当前附带的示例是：

- `01chicago/`

其中包含代码所需的原始文件：

- `scene_config.json`
- `xyz_paths.json`
- `01chicago.xml`
- `meshes/`

这些文件会被以下代码使用：

- `tools/slurm_dataset_gen_01chicago_gt_only.sh`
- `twm.datasets.data_gen_sionna`

### 默认使用方式

现在数据生成 Slurm 脚本默认使用：

- `RAW_DATA_ROOT=$REPO_ROOT/example_data`
- `RUN_ROOT=$REPO_ROOT/example_runs`

因此可以直接使用仓库内附带的 `01chicago` 示例。

### 说明

仓库已经包含代码所需的原始场景文件，但真正运行生成流程时，仍需要安装 `twm.datasets.data_gen_sionna` 所依赖的外部 Sionna / Dr.Jit 运行环境。
