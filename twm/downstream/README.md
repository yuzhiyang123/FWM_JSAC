# twm/downstream

## English

### Purpose

twm/downstream contains the downstream channel reconstruction tasks and their supporting components.
It provides the shared conditioning utilities, the 32x16 regression-family tasks, the 1024x64 reconstruction-family tasks, and a small set of baselines used for comparison.

### Current simplified structure

The open-source version keeps two main task packages and moves the thin wrappers into single-file modules.

#### Main task packages

- `channel_pred/`
  Shared downstream foundation. This package contains the common data loading logic, shared batch definitions, pilot processing, alignment modules, and the generic conditioning task base used by multiple downstream tasks.

- `channel_pred_reconstruction1024x64/`
  Main 1024x64 reconstruction task package. This package contains the large-scale regressor, the reconstruction task logic, and the communication-oriented evaluation utilities in `comm_eval.py`.

#### Single-file task wrappers

- `channel_pred_pointwise.py`
  Pointwise 32x16 data path used to build per-time-step training samples.

- `channel_pred_regression32x16.py`
  Consolidated 32x16 regression family. This file contains the standard regression task, the token-only variant, and the pilot-only variant.

- `channel_pred_reconstruction1024x64_variants.py`
  Consolidated 1024x64 token-only variants. This file contains the token-only and zero-condition variants, as well as the pointwise-to-shared-batch adapter used by those variants.
  In the open-source helper entrypoint, train_mode=auto keeps window training for token-only no-calibration runs, but switches zero-condition runs to pointwise training by default.

- `baselines_regression32x16.py`
  Consolidated 32x16 baseline tasks. This file contains both the direct-training baseline and the simple-prediction baseline.

- `dit32x16.py`
  Shared diffusion transformer building block required by the downstream diffusion-style models.

### How the pieces relate

- `channel_pred/` is the common base.
- `channel_pred_regression32x16.py` builds on `channel_pred/` and `channel_pred_pointwise.py`.
- `channel_pred_reconstruction1024x64/` builds on `channel_pred/` and the multiscale JEPA pipelines.
- `channel_pred_reconstruction1024x64_variants.py` adds the lightweight token-only and zero-condition variants on top of `channel_pred_reconstruction1024x64/`.
- `baselines_regression32x16.py` reuses the shared regression backbone and utilities from the 32x16 path.

### Runnable entrypoints

The runnable downstream entrypoints are kept in:

- `twm/run_helpers/run_downstream_common.py`
- `twm/run_helpers/run_downstream_regression32x16.py`
- `twm/run_helpers/run_downstream_token_only_regression32x16.py`
- `twm/run_helpers/run_downstream_pilot_only_regression32x16.py`
- `twm/run_helpers/run_downstream_token_only_regression32x16_test.py`
- `twm/run_helpers/run_downstream_reconstruction1024x64.py`
- `twm/run_helpers/run_downstream_token_only_reconstruction1024x64.py`
- `twm/run_helpers/run_downstream_token_only_reconstruction1024x64_proposed_test.py`
- `twm/run_helpers/run_downstream_token_only_reconstruction1024x64_baseline_test.py`

### Minimal Slurm templates

The open-source package keeps only simple single-run Slurm templates for downstream training and testing:

- `tools/slurm_downstream_regression32x16.sh`
- `tools/slurm_downstream_token_only_regression32x16.sh`
- `tools/slurm_downstream_pilot_only_regression32x16.sh`
- `tools/slurm_downstream_token_only_regression32x16_test.sh`
- `tools/slurm_downstream_reconstruction1024x64.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64_proposed_test.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64_baseline_test.sh`

These templates are intentionally parameterized through environment variables instead of array manifests or sweep scripts.

### Script details

#### `twm/run_helpers/`

- `run_downstream_common.py`
  Shared helper definitions used by the 32x16 downstream training entrypoints.

- `run_downstream_regression32x16.py`
  Training entrypoint for the standard 32x16 regression downstream task.

- `run_downstream_token_only_regression32x16.py`
  Training entrypoint for the 32x16 token-only downstream task.

- `run_downstream_pilot_only_regression32x16.py`
  Training entrypoint for the 32x16 pilot-only baseline.

- `run_downstream_token_only_regression32x16_test.py`
  Fixed-noise evaluation entrypoint for the 32x16 token-only downstream task.

- `run_downstream_reconstruction1024x64.py`
  Training entrypoint for the main 1024x64 reconstruction downstream task.

- `run_downstream_token_only_reconstruction1024x64.py`
  Training entrypoint for the 1024x64 token-only / zero-condition downstream variants.

- `run_downstream_token_only_reconstruction1024x64_proposed_test.py`
  Evaluation entrypoint for the proposed 1024x64 downstream models.

- `run_downstream_token_only_reconstruction1024x64_baseline_test.py`
  Evaluation entrypoint for the 1024x64 baseline models.

#### `tools/`

- `slurm_downstream_regression32x16.sh`
  Minimal single-run Slurm template for standard 32x16 regression training.

- `slurm_downstream_token_only_regression32x16.sh`
  Minimal single-run Slurm template for 32x16 token-only training.

- `slurm_downstream_pilot_only_regression32x16.sh`
  Minimal single-run Slurm template for the 32x16 pilot-only baseline.

- `slurm_downstream_token_only_regression32x16_test.sh`
  Minimal single-run Slurm template for 32x16 token-only testing.

- `slurm_downstream_reconstruction1024x64.sh`
  Minimal single-run Slurm template for the main 1024x64 reconstruction task.

- `slurm_downstream_token_only_reconstruction1024x64.sh`
  Minimal single-run Slurm template for the 1024x64 token-only / zero-condition variants.

- `slurm_downstream_token_only_reconstruction1024x64_proposed_test.sh`
  Minimal single-run Slurm template for proposed-model testing at 1024x64.

- `slurm_downstream_token_only_reconstruction1024x64_baseline_test.sh`
  Minimal single-run Slurm template for baseline testing at 1024x64.

### Script details

#### `twm/run_helpers/`

- `run_downstream_common.py`
  Shared helper definitions used by the 32x16 downstream training entrypoints.

- `run_downstream_regression32x16.py`
  Training entrypoint for the standard 32x16 regression downstream task.

- `run_downstream_token_only_regression32x16.py`
  Training entrypoint for the 32x16 token-only downstream task.

- `run_downstream_pilot_only_regression32x16.py`
  Training entrypoint for the 32x16 pilot-only baseline.

- `run_downstream_token_only_regression32x16_test.py`
  Fixed-noise evaluation entrypoint for the 32x16 token-only downstream task.

- `run_downstream_reconstruction1024x64.py`
  Training entrypoint for the main 1024x64 reconstruction downstream task.

- `run_downstream_token_only_reconstruction1024x64.py`
  Training entrypoint for the 1024x64 token-only / zero-condition downstream variants.

- `run_downstream_token_only_reconstruction1024x64_proposed_test.py`
  Evaluation entrypoint for the proposed 1024x64 downstream models.

- `run_downstream_token_only_reconstruction1024x64_baseline_test.py`
  Evaluation entrypoint for the 1024x64 baseline models.

#### `tools/`

- `slurm_downstream_regression32x16.sh`
  Minimal single-run Slurm template for standard 32x16 regression training.

- `slurm_downstream_token_only_regression32x16.sh`
  Minimal single-run Slurm template for 32x16 token-only training.

- `slurm_downstream_pilot_only_regression32x16.sh`
  Minimal single-run Slurm template for the 32x16 pilot-only baseline.

- `slurm_downstream_token_only_regression32x16_test.sh`
  Minimal single-run Slurm template for 32x16 token-only testing.

- `slurm_downstream_reconstruction1024x64.sh`
  Minimal single-run Slurm template for the main 1024x64 reconstruction task.

- `slurm_downstream_token_only_reconstruction1024x64.sh`
  Minimal single-run Slurm template for the 1024x64 token-only / zero-condition variants.

- `slurm_downstream_token_only_reconstruction1024x64_proposed_test.sh`
  Minimal single-run Slurm template for proposed-model testing at 1024x64.

- `slurm_downstream_token_only_reconstruction1024x64_baseline_test.sh`
  Minimal single-run Slurm template for baseline testing at 1024x64.

---

## 中文

### 作用

twm/downstream 包含下游信道重建任务及其配套组件。
它提供共享的条件构造工具、32x16 regression 系列任务、1024x64 reconstruction 系列任务，以及少量用于对比的 baseline。

### 当前的精简结构

当前开源版本保留两个主体任务 package，并把原来只是薄包装的部分压缩成单文件模块。

#### 主体任务 package

- `channel_pred/`
  下游任务的共享底座。这个 package 包含通用的数据加载逻辑、共享 batch 定义、pilot 处理、alignment 模块，以及多个下游任务会复用的通用条件构造 task base。

- `channel_pred_reconstruction1024x64/`
  主体的 1024x64 reconstruction 任务 package。这个 package 包含大尺度 regressor、reconstruction 任务逻辑，以及 `comm_eval.py` 中的通信评估工具。

#### 单文件任务包装层

- `channel_pred_pointwise.py`
  32x16 的 pointwise 数据路径，用于把每个时间点切成单独训练样本。

- `channel_pred_regression32x16.py`
  合并后的 32x16 regression 系列文件。里面包含标准 regression task、token-only 变体和 pilot-only 变体。

- `channel_pred_reconstruction1024x64_variants.py`
  合并后的 1024x64 token-only 变体文件。里面包含 token-only、zero-condition 变体，以及这些变体所需的 pointwise-to-shared-batch 适配逻辑。
  在开源版 helper 入口中，train_mode=auto 会让 token-only 且无 calibration 的训练保持 window 方式，而 zero-condition 默认切换为 pointwise 训练。

- `baselines_regression32x16.py`
  合并后的 32x16 baseline 文件。里面同时包含 direct-training baseline 和 simple-prediction baseline。

- `dit32x16.py`
  共享的 diffusion transformer 基础模块，供下游 diffusion 风格模型使用。

### 各部分之间的关系

- `channel_pred/` 是共享底座。
- `channel_pred_regression32x16.py` 建立在 `channel_pred/` 和 `channel_pred_pointwise.py` 之上。
- `channel_pred_reconstruction1024x64/` 建立在 `channel_pred/` 和多尺度 JEPA pipeline 之上。
- `channel_pred_reconstruction1024x64_variants.py` 在 `channel_pred_reconstruction1024x64/` 的基础上增加了轻量的 token-only 和 zero-condition 变体。
- `baselines_regression32x16.py` 复用了 32x16 路径中的共享 regression 骨干和工具函数。

### 可运行入口

可直接运行的 downstream 入口集中放在：

- `twm/run_helpers/run_downstream_common.py`
- `twm/run_helpers/run_downstream_regression32x16.py`
- `twm/run_helpers/run_downstream_token_only_regression32x16.py`
- `twm/run_helpers/run_downstream_pilot_only_regression32x16.py`
- `twm/run_helpers/run_downstream_token_only_regression32x16_test.py`
- `twm/run_helpers/run_downstream_reconstruction1024x64.py`
- `twm/run_helpers/run_downstream_token_only_reconstruction1024x64.py`
- `twm/run_helpers/run_downstream_token_only_reconstruction1024x64_proposed_test.py`
- `twm/run_helpers/run_downstream_token_only_reconstruction1024x64_baseline_test.py`

### 最简 Slurm 模板

开源版只保留最基本的单任务 downstream 训练与测试模板：

- `tools/slurm_downstream_regression32x16.sh`
- `tools/slurm_downstream_token_only_regression32x16.sh`
- `tools/slurm_downstream_pilot_only_regression32x16.sh`
- `tools/slurm_downstream_token_only_regression32x16_test.sh`
- `tools/slurm_downstream_reconstruction1024x64.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64_proposed_test.sh`
- `tools/slurm_downstream_token_only_reconstruction1024x64_baseline_test.sh`

这些模板统一采用环境变量传参，不再依赖 array manifest 或 sweep 脚本。

### 脚本说明

#### `twm/run_helpers/`

- `run_downstream_common.py`
  供 32x16 downstream 训练入口复用的共享辅助定义。

- `run_downstream_regression32x16.py`
  标准 32x16 regression downstream 任务的训练入口。

- `run_downstream_token_only_regression32x16.py`
  32x16 token-only downstream 任务的训练入口。

- `run_downstream_pilot_only_regression32x16.py`
  32x16 pilot-only baseline 的训练入口。

- `run_downstream_token_only_regression32x16_test.py`
  32x16 token-only downstream 的固定噪声测试入口。

- `run_downstream_reconstruction1024x64.py`
  主体 1024x64 reconstruction downstream 任务的训练入口。

- `run_downstream_token_only_reconstruction1024x64.py`
  1024x64 token-only / zero-condition downstream 变体的训练入口。

- `run_downstream_token_only_reconstruction1024x64_proposed_test.py`
  1024x64 proposed 模型的测试入口。

- `run_downstream_token_only_reconstruction1024x64_baseline_test.py`
  1024x64 baseline 模型的测试入口。

#### `tools/`

- `slurm_downstream_regression32x16.sh`
  标准 32x16 regression 训练的最简单任务 Slurm 模板。

- `slurm_downstream_token_only_regression32x16.sh`
  32x16 token-only 训练的最简单任务 Slurm 模板。

- `slurm_downstream_pilot_only_regression32x16.sh`
  32x16 pilot-only baseline 的最简单任务 Slurm 模板。

- `slurm_downstream_token_only_regression32x16_test.sh`
  32x16 token-only 测试的最简单任务 Slurm 模板。

- `slurm_downstream_reconstruction1024x64.sh`
  主体 1024x64 reconstruction 任务的最简单任务 Slurm 模板。

- `slurm_downstream_token_only_reconstruction1024x64.sh`
  1024x64 token-only / zero-condition 变体的最简单任务 Slurm 模板。

- `slurm_downstream_token_only_reconstruction1024x64_proposed_test.sh`
  1024x64 proposed 模型测试的最简单任务 Slurm 模板。

- `slurm_downstream_token_only_reconstruction1024x64_baseline_test.sh`
  1024x64 baseline 模型测试的最简单任务 Slurm 模板。

