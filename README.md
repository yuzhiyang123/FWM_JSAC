# TWM Dataset JEPA Regression

## To Begin With ...

This repository corresponds to a paper entitled "Field-Layer World Model: Efficient OFDM Receiver by Bridging Channel Prediction and Estimation," submitted to IEEE JSAC.

We provide README in both English and Chinese. The Chinese version is located after the English version.

本仓库所有README均含有中文说明，中文说明在英文说明之后。

## English

### Overview

This repository contains the code used for JEPA pretraining and downstream channel reconstruction experiments on the TWM dataset pipeline.

The open-source package is organized around three main parts:

- `twm/`
  Main Python package containing dataset code, JEPA pipelines, and runnable training entrypoints.

- `tools/`
  Utility scripts, including Slurm submission templates and helper launchers that map override configuration ids to concrete training runs.

- `tmp/`
  Optional temporary manifests or intermediate bookkeeping files. This directory is not required for the core training flows.

### Dataset generation

The main Slurm entrypoint currently provided for dataset generation is:

- `tools/slurm_dataset_gen_01chicago_gt_only.sh`

This script drives the end-to-end generation flow for the `01chicago` scenario. In the current setup, it performs the following stages:

1. generate raw scenario snapshots with `twm.datasets.data_gen_sionna`,
2. split snapshots into train/test subsets with `twm.datasets.split_snapshot_dataset`,
3. build train ground-truth shards with `twm.datasets.fixed_band_dataset`,
4. build test ground-truth shards with `twm.datasets.fixed_band_dataset`.

The generated outputs include:

- full snapshot file,
- train/test snapshot files,
- train/test ground-truth shard directories,
- split report,
- train/test channel-gain reports.

The repository also includes a runnable raw-scene example under:

- `example_data/01chicago`

This example contains the files required by the generation code:

- `scene_config.json`
- `xyz_paths.json`
- `01chicago.xml`
- `meshes/`

By default, the provided dataset-generation Slurm script uses:

- `RAW_DATA_ROOT=$REPO_ROOT/example_data`
- `RUN_ROOT=$REPO_ROOT/example_runs`

so the bundled example can be used directly.

Detailed dataset-side documentation is provided in:

- `twm/datasets/README.md`
- `example_data/README.md`

### JEPA training structure

The JEPA training code is split into three scales:

- `32x16`
  first-stage JEPA pretraining,
- `128x64`
  second-stage JEPA transfer / EMA / scratch training,
- `1024x64`
  third-stage JEPA transfer / EMA / scratch training.

The model definitions are under:

- `twm/pipelines/`

The command-line training entrypoints are under:

- `twm/run_helpers/`

The helper launchers that turn override configuration ids into concrete runs are under:

- `tools/run_jepa_32x16_from_override.py`
- `tools/run_jepa_128x64_from_override.py`
- `tools/run_jepa_1024x64_from_override.py`


### Downstream training structure

The downstream code is organized into two main groups:

- `32x16` downstream regression tasks,
- `1024x64` downstream reconstruction tasks.

The model-side implementation is under:

- `twm/downstream/`

The runnable downstream entrypoints are under:

- `twm/run_helpers/`


The downstream entrypoints are:

- `run_downstream_common.py`: shared helper utilities for downstream training.
- `run_downstream_regression32x16.py`: standard 32x16 regression training.
- `run_downstream_token_only_regression32x16.py`: 32x16 token-only training.
- `run_downstream_pilot_only_regression32x16.py`: 32x16 pilot-only baseline training.
- `run_downstream_token_only_regression32x16_test.py`: 32x16 token-only fixed-noise test.
- `run_downstream_reconstruction1024x64.py`: standard 1024x64 reconstruction training.
- `run_downstream_token_only_reconstruction1024x64.py`: 1024x64 token-only / zero-condition training.
- `run_downstream_token_only_reconstruction1024x64_proposed_test.py`: proposed-model test at 1024x64.
- `run_downstream_token_only_reconstruction1024x64_baseline_test.py`: baseline test at 1024x64.

The open-source package keeps only simple single-run Slurm templates for downstream training and testing:

- `tools/slurm_downstream_regression32x16.sh`
- `tools/slurm_downstream_token_only_regression32x16.sh`
- `tools/slurm_downstream_pilot_only_regression32x16.sh`
- `tools/slurm_downstream_token_only_regression32x16_test.sh`
- `tools/slurm_downstream_reconstruction1024x64.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64_proposed_test.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64_baseline_test.sh`

These scripts are parameterized through environment variables instead of sweep manifests.

### How to submit downstream runs

For `32x16` token-only downstream training:

```bash
export DATASET_TRAIN=/path/to/train_npy
export DATASET_TEST=/path/to/test_npy
export PRETRAINED_DIR=/path/to/jepa_32x16/best_model
export WORKING_DIR=/path/to/output/downstream_token_only_32x16
export WANDB_PROJECT=my_project
sbatch tools/slurm_downstream_token_only_regression32x16.sh
```

For `1024x64` token-only downstream training:

```bash
export DATASET_TRAIN=/path/to/train_npy
export DATASET_TEST=/path/to/test_npy
export PRETRAINED_DIR=/path/to/jepa_1024x64/best_model
export WORKING_DIR=/path/to/output/downstream_token_only_1024x64
export PILOT_STRIDE=128
export WANDB_PROJECT=my_project
sbatch tools/slurm_downstream_token_only_reconstruction1024x64.sh
```

For proposed-model evaluation:

```bash
export DATASET_TRAIN=/path/to/train_npy
export DATASET_TEST=/path/to/test_npy
export TRAINED_RUN_DIR=/path/to/output/downstream_token_only_1024x64
export OUTPUT_CSV=/path/to/output/proposed_results.csv
sbatch tools/slurm_downstream_token_only_reconstruction1024x64_proposed_test.sh
```

For baseline evaluation:

```bash
export DATASET_TRAIN=/path/to/train_npy
export DATASET_TEST=/path/to/test_npy
export TRAINED_RUN_DIR=/path/to/output/downstream_baseline_1024x64
export OUTPUT_CSV=/path/to/output/baseline_results.csv
sbatch tools/slurm_downstream_token_only_reconstruction1024x64_baseline_test.sh
```

### Override configurations

The open-source subset keeps 20 override configurations in total:

- `0-8`: the `loss9` sweep,
- `9-19`: the additional `topology12` sweep entries.

These configurations are defined in:

- `twm/pipelines/jepa_override_configs.py`

### How to choose a network configuration

To choose which network to train, specify the override configuration id:

- for `32x16`, pass `--config-id <id>` to `tools/run_jepa_32x16_from_override.py`,
- for `128x64` and `1024x64`, pass `--config-id <id>` to the corresponding helper launcher.

Example:

```bash
python tools/run_jepa_32x16_from_override.py   --config-id 1   --dataset-file /path/to/train_npy /path/to/test_npy   --working-dir-root /path/to/output/jepa_32x16   --wandb-project my_project
```

### How to choose a training method

For `32x16`, the method is fully determined by the selected override configuration.

For `128x64`, select one of:

- `transfer`
- `ema32`
- `scratch`

Example:

```bash
python tools/run_jepa_128x64_from_override.py   --method transfer   --config-id 9   --dataset-file /path/to/train_npy /path/to/test_npy   --pretrain-root /path/to/output/jepa_32x16   --working-dir-root /path/to/output/jepa_128x64_transfer   --wandb-project my_project   --epochs 200
```

For `1024x64`, select one of:

- `transfer_ref32`
- `transfer_ref128`
- `ema_ref32`
- `ema_ref128`
- `scratch`

Example:

```bash
python tools/run_jepa_1024x64_from_override.py   --method transfer_ref128   --config-id 19   --dataset-file /path/to/train_npy /path/to/test_npy   --pretrain-32-root /path/to/output/jepa_32x16   --pretrain-128-transfer-root /path/to/output/jepa_128x64_transfer   --working-dir-root /path/to/output/jepa_1024x64_transfer_ref128   --wandb-project my_project   --epochs 200
```

### How to modify training parameters

All helper launchers expose the main training arguments directly. Common examples include:

- `--batch-size`
- `--gradient-accumulation-steps`
- `--epochs`
- `--num-workers`
- `--window-length`
- `--random-windows-per-sequence`
- `--train-paths`
- `--test-paths`
- `--seed`
- `--device`
- `--save-every`

For `32x16`, the helper also exposes:

- `--lr`
- `--num-low-bands`

For EMA-based stages, the helpers also expose:

- `--teacher-momentum`

### How to modify paths

The following path groups are typically the ones you need to change:

1. dataset paths
- `--dataset-file /path/to/train_npy /path/to/test_npy`

2. output root
- `--working-dir-root /path/to/output/...`

3. pretrained model roots
- `--pretrain-root` for `128x64 transfer` and `128x64 ema32`,
- `--pretrain-32-root` for pretrained `1024x64` methods,
- `--pretrain-128-transfer-root` for `1024x64 transfer_ref32` and `transfer_ref128`,
- `--pretrain-128-ema-root` for `1024x64 ema_ref32` and `ema_ref128`.

The helper launcher resolves the exact `best_model` directory automatically from the configuration id and run name.

### How to submit with Slurm

Open-source Slurm templates are provided in `tools/`:

- `slurm_jepa_32x16.sh`
- `slurm_jepa_128x64_transfer.sh`
- `slurm_jepa_128x64_ema32.sh`
- `slurm_jepa_128x64_ema32_scratch.sh`
- `slurm_jepa_1024x64_transfer_ref32.sh`
- `slurm_jepa_1024x64_transfer_ref128.sh`
- `slurm_jepa_1024x64_ema_ref32.sh`
- `slurm_jepa_1024x64_ema_ref128.sh`
- `slurm_jepa_1024x64_scratch.sh`

Before submission, set the path-related environment variables used by the script. At minimum, you usually need to set:

- `DATA_ROOT`
- `WORKING_DIR_ROOT`
- and, for transfer or EMA stages, the corresponding pretrained roots.

Example:

```bash
export DATA_ROOT=/path/to/prepared_dataset_root/01chicago
export WORKING_DIR_ROOT=/path/to/output/jepa_32x16
export WANDB_PROJECT=my_project
sbatch --array=0-19 tools/slurm_jepa_32x16.sh
```

For second-stage transfer:

```bash
export DATA_ROOT=/path/to/prepared_dataset_root/01chicago
export PRETRAIN_ROOT=/path/to/output/jepa_32x16
export WORKING_DIR_ROOT=/path/to/output/jepa_128x64_transfer
export WANDB_PROJECT=my_project
sbatch --array=0-19 tools/slurm_jepa_128x64_transfer.sh
```

For third-stage transfer from 128x64:

```bash
export DATA_ROOT=/path/to/prepared_dataset_root/01chicago
export PRETRAIN_32_ROOT=/path/to/output/jepa_32x16
export PRETRAIN_128_TRANSFER_ROOT=/path/to/output/jepa_128x64_transfer
export WORKING_DIR_ROOT=/path/to/output/jepa_1024x64_transfer_ref128
export WANDB_PROJECT=my_project
sbatch --array=0-19 tools/slurm_jepa_1024x64_transfer_ref128.sh
```

### Where to find more detailed pipeline documentation

See:

- `twm/pipelines/README.md`

That file describes how the model-side pipeline code is organized and points to per-subdirectory READMEs.

---

## 中文

### 概述

这个仓库包含了用于 TWM 数据集流程上的 JEPA 预训练与下游信道重建实验代码。

当前开源版本主要分成三部分：

- `twm/`
  主 Python 包，包含数据集代码、JEPA pipeline，以及可直接运行的训练入口。

- `tools/`
  工具脚本目录，包含 Slurm 提交模板，以及将 override 配置编号映射到具体训练任务的辅助启动脚本。

- `tmp/`
  可选的临时 manifest 或中间记录文件。核心训练流程并不依赖这个目录。

### 数据集生成

当前提供的数据集生成主 Slurm 入口是：

- `tools/slurm_dataset_gen_01chicago_gt_only.sh`

这份脚本负责 `01chicago` 场景的端到端数据生成流程。在当前设置下，它依次完成以下步骤：

1. 使用 `twm.datasets.data_gen_sionna` 生成原始场景 snapshot；
2. 使用 `twm.datasets.split_snapshot_dataset` 将 snapshot 切分成 train/test 子集；
3. 使用 `twm.datasets.fixed_band_dataset` 构建训练 ground-truth shards；
4. 使用 `twm.datasets.fixed_band_dataset` 构建测试 ground-truth shards。

生成产物包括：

- 完整 snapshot 文件；
- train/test snapshot 文件；
- train/test ground-truth shard 目录；
- split report；
- train/test channel-gain report。

更详细的数据集说明见：

- `twm/datasets/README.md`

### JEPA 训练结构

JEPA 训练代码分成三个尺度：

- `32x16`
  第一阶段 JEPA 预训练；
- `128x64`
  第二阶段的 transfer / EMA / scratch 训练；
- `1024x64`
  第三阶段的 transfer / EMA / scratch 训练。

模型定义位于：

- `twm/pipelines/`

命令行训练入口位于：

- `twm/run_helpers/`

把 override 配置编号转换成具体训练任务的辅助启动脚本位于：

- `tools/run_jepa_32x16_from_override.py`
- `tools/run_jepa_128x64_from_override.py`
- `tools/run_jepa_1024x64_from_override.py`


### Downstream 训练结构

当前 downstream 代码分成两大类：

- `32x16` downstream regression 任务；
- `1024x64` downstream reconstruction 任务。

模型实现位于：

- `twm/downstream/`

可直接运行的 downstream 入口位于：

- `twm/run_helpers/`


具体的 downstream 入口脚本包括：

- `run_downstream_common.py`：downstream 训练的共享辅助定义。
- `run_downstream_regression32x16.py`：标准 32x16 regression 训练入口。
- `run_downstream_token_only_regression32x16.py`：32x16 token-only 训练入口。
- `run_downstream_pilot_only_regression32x16.py`：32x16 pilot-only baseline 训练入口。
- `run_downstream_token_only_regression32x16_test.py`：32x16 token-only 固定噪声测试入口。
- `run_downstream_reconstruction1024x64.py`：标准 1024x64 reconstruction 训练入口。
- `run_downstream_token_only_reconstruction1024x64.py`：1024x64 token-only / zero-condition 训练入口。
- `run_downstream_token_only_reconstruction1024x64_proposed_test.py`：1024x64 proposed 模型测试入口。
- `run_downstream_token_only_reconstruction1024x64_baseline_test.py`：1024x64 baseline 模型测试入口。

开源版只保留最基本的单任务 downstream 训练与测试 Slurm 模板：

- `tools/slurm_downstream_regression32x16.sh`
- `tools/slurm_downstream_token_only_regression32x16.sh`
- `tools/slurm_downstream_pilot_only_regression32x16.sh`
- `tools/slurm_downstream_token_only_regression32x16_test.sh`
- `tools/slurm_downstream_reconstruction1024x64.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64_proposed_test.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64_baseline_test.sh`

这些脚本统一采用环境变量传参，不再依赖 sweep manifest。

### 如何提交 downstream 任务

对于 `32x16` token-only downstream 训练：

```bash
export DATASET_TRAIN=/path/to/train_npy
export DATASET_TEST=/path/to/test_npy
export PRETRAINED_DIR=/path/to/jepa_32x16/best_model
export WORKING_DIR=/path/to/output/downstream_token_only_32x16
export WANDB_PROJECT=my_project
sbatch tools/slurm_downstream_token_only_regression32x16.sh
```

对于 `1024x64` token-only downstream 训练：

```bash
export DATASET_TRAIN=/path/to/train_npy
export DATASET_TEST=/path/to/test_npy
export PRETRAINED_DIR=/path/to/jepa_1024x64/best_model
export WORKING_DIR=/path/to/output/downstream_token_only_1024x64
export PILOT_STRIDE=128
export WANDB_PROJECT=my_project
sbatch tools/slurm_downstream_token_only_reconstruction1024x64.sh
```

对于 proposed 模型测试：

```bash
export DATASET_TRAIN=/path/to/train_npy
export DATASET_TEST=/path/to/test_npy
export TRAINED_RUN_DIR=/path/to/output/downstream_token_only_1024x64
export OUTPUT_CSV=/path/to/output/proposed_results.csv
sbatch tools/slurm_downstream_token_only_reconstruction1024x64_proposed_test.sh
```

对于 baseline 测试：

```bash
export DATASET_TRAIN=/path/to/train_npy
export DATASET_TEST=/path/to/test_npy
export TRAINED_RUN_DIR=/path/to/output/downstream_baseline_1024x64
export OUTPUT_CSV=/path/to/output/baseline_results.csv
sbatch tools/slurm_downstream_token_only_reconstruction1024x64_baseline_test.sh
```

### Override 配置

当前开源子集一共保留 20 个 override 配置：

- `0-8`：`loss9` 扫描；
- `9-19`：`topology12` 扫描中的补充配置。

这些配置定义在：

- `twm/pipelines/jepa_override_configs.py`

### 如何选择网络配置

要选择训练哪个网络，直接指定 override 配置编号即可：

- 对 `32x16`，在 `tools/run_jepa_32x16_from_override.py` 中传入 `--config-id <id>`；
- 对 `128x64` 和 `1024x64`，在对应 helper 启动脚本中传入 `--config-id <id>`。

示例：

```bash
python tools/run_jepa_32x16_from_override.py   --config-id 1   --dataset-file /path/to/train_npy /path/to/test_npy   --working-dir-root /path/to/output/jepa_32x16   --wandb-project my_project
```

### 如何选择训练方法

对于 `32x16`，训练方法完全由所选 override 配置决定。

对于 `128x64`，可以选择：

- `transfer`
- `ema32`
- `scratch`

示例：

```bash
python tools/run_jepa_128x64_from_override.py   --method transfer   --config-id 9   --dataset-file /path/to/train_npy /path/to/test_npy   --pretrain-root /path/to/output/jepa_32x16   --working-dir-root /path/to/output/jepa_128x64_transfer   --wandb-project my_project   --epochs 200
```

对于 `1024x64`，可以选择：

- `transfer_ref32`
- `transfer_ref128`
- `ema_ref32`
- `ema_ref128`
- `scratch`

示例：

```bash
python tools/run_jepa_1024x64_from_override.py   --method transfer_ref128   --config-id 19   --dataset-file /path/to/train_npy /path/to/test_npy   --pretrain-32-root /path/to/output/jepa_32x16   --pretrain-128-transfer-root /path/to/output/jepa_128x64_transfer   --working-dir-root /path/to/output/jepa_1024x64_transfer_ref128   --wandb-project my_project   --epochs 200
```

### 如何修改训练参数

所有 helper 启动脚本都直接暴露了主要训练参数。常见的包括：

- `--batch-size`
- `--gradient-accumulation-steps`
- `--epochs`
- `--num-workers`
- `--window-length`
- `--random-windows-per-sequence`
- `--train-paths`
- `--test-paths`
- `--seed`
- `--device`
- `--save-every`

对 `32x16`，helper 还额外暴露：

- `--lr`
- `--num-low-bands`

对 EMA 相关阶段，helper 还暴露：

- `--teacher-momentum`

### 如何修改路径

通常需要修改的路径分成三类：

1. 数据集路径
- `--dataset-file /path/to/train_npy /path/to/test_npy`

2. 输出根目录
- `--working-dir-root /path/to/output/...`

3. 预训练模型根目录
- `128x64 transfer` 和 `128x64 ema32` 使用 `--pretrain-root`；
- 预训练型 `1024x64` 方法使用 `--pretrain-32-root`；
- `1024x64 transfer_ref32` 与 `transfer_ref128` 使用 `--pretrain-128-transfer-root`；
- `1024x64 ema_ref32` 与 `ema_ref128` 使用 `--pretrain-128-ema-root`。

helper 启动脚本会根据配置编号和 run 名，自动解析到具体的 `best_model` 目录。

### 如何用 Slurm 提交

当前在 `tools/` 里提供了开源版 Slurm 模板：

- `slurm_jepa_32x16.sh`
- `slurm_jepa_128x64_transfer.sh`
- `slurm_jepa_128x64_ema32.sh`
- `slurm_jepa_128x64_ema32_scratch.sh`
- `slurm_jepa_1024x64_transfer_ref32.sh`
- `slurm_jepa_1024x64_transfer_ref128.sh`
- `slurm_jepa_1024x64_ema_ref32.sh`
- `slurm_jepa_1024x64_ema_ref128.sh`
- `slurm_jepa_1024x64_scratch.sh`

提交前，需要设置脚本中使用的路径类环境变量。通常至少需要设置：

- `DATA_ROOT`
- `WORKING_DIR_ROOT`
- 以及 transfer 或 EMA 阶段对应的预训练根目录

示例：

```bash
export DATA_ROOT=/path/to/prepared_dataset_root/01chicago
export WORKING_DIR_ROOT=/path/to/output/jepa_32x16
export WANDB_PROJECT=my_project
sbatch --array=0-19 tools/slurm_jepa_32x16.sh
```

第二阶段 transfer 示例：

```bash
export DATA_ROOT=/path/to/prepared_dataset_root/01chicago
export PRETRAIN_ROOT=/path/to/output/jepa_32x16
export WORKING_DIR_ROOT=/path/to/output/jepa_128x64_transfer
export WANDB_PROJECT=my_project
sbatch --array=0-19 tools/slurm_jepa_128x64_transfer.sh
```

第三阶段从 128x64 transfer 继续的示例：

```bash
export DATA_ROOT=/path/to/prepared_dataset_root/01chicago
export PRETRAIN_32_ROOT=/path/to/output/jepa_32x16
export PRETRAIN_128_TRANSFER_ROOT=/path/to/output/jepa_128x64_transfer
export WORKING_DIR_ROOT=/path/to/output/jepa_1024x64_transfer_ref128
export WANDB_PROJECT=my_project
sbatch --array=0-19 tools/slurm_jepa_1024x64_transfer_ref128.sh
```

### 更详细的 pipeline 说明位置

见：

- `twm/pipelines/README.md`

该文件说明了模型侧 pipeline 代码如何组织，并指向各个子目录的 README。
