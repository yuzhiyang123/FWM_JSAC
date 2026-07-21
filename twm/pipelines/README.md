# twm/pipelines

## English

### Purpose

twm/pipelines contains the model-side pipeline code for JEPA pretraining and multiscale latent transfer.
It defines how inputs are preprocessed, tokenized, encoded, masked, predicted, and trained at the 32x16, 128x64, and 1024x64 scales.

### Main components

- jepa_32x16/
  Core 32x16 JEPA pipeline package. See `jepa_32x16/README.md` for file-level details.

- jepa_128x64/
  Second-stage JEPA package built on top of the 32x16 stage. See `jepa_128x64/README.md`.

- jepa_1024x64/
  Third-stage JEPA package for the largest scale. See `jepa_1024x64/README.md`.

- tokenizer/
  Shared tokenizer package used by the JEPA pipelines at different scales. See `tokenizer/README.md`.

### Training entrypoints and launch helpers

The runnable command-line entrypoints are placed under:

- `twm/run_helpers/`

These files are thin wrappers around the pipeline training modules. They expose standard command-line arguments and are intended to be called directly, or through the helper launchers in `tools/`.

The override-based helper launchers are:

- `tools/run_jepa_32x16_from_override.py`
- `tools/run_jepa_128x64_from_override.py`
- `tools/run_jepa_1024x64_from_override.py`

These helper launchers map a configuration id to a concrete model configuration and assemble the correct training run for each scale and training method.

The corresponding Slurm templates are stored in:

- `tools/slurm_jepa_32x16.sh`
- `tools/slurm_jepa_128x64_transfer.sh`
- `tools/slurm_jepa_128x64_ema32.sh`
- `tools/slurm_jepa_128x64_ema32_scratch.sh`
- `tools/slurm_jepa_1024x64_transfer_ref32.sh`
- `tools/slurm_jepa_1024x64_transfer_ref128.sh`
- `tools/slurm_jepa_1024x64_ema_ref32.sh`
- `tools/slurm_jepa_1024x64_ema_ref128.sh`
- `tools/slurm_jepa_1024x64_scratch.sh`

### Confirmed external dependencies

This directory is mostly self-contained, but it currently imports a few modules from outside `twm/pipelines`:

- `twm.datasets`
  Used by the training entrypoints to build dataloaders.

- `twm.noise_generators`
  Used by `jepa_32x16/utils.py`.

### Kept override configuration subset

The open-source subset keeps 20 override configurations in total.
They are organized as:

- `0-8`: loss9 sweep
- `9-19`: topology12 sweep additions

The corresponding file is:

- `jepa_override_configs.py`

---

## 中文

### 作用

twm/pipelines 包含 JEPA 预训练与多尺度 latent 迁移所需的模型侧 pipeline 代码。
它定义了在 32x16、128x64 和 1024x64 三个尺度下，输入如何被预处理、tokenize、编码、mask、预测以及训练。

### 主要组成

- jepa_32x16/
  核心的 32x16 JEPA package。各文件的详细说明见 `jepa_32x16/README.md`。

- jepa_128x64/
  建立在 32x16 阶段之上的第二阶段 JEPA package。详细说明见 `jepa_128x64/README.md`。

- jepa_1024x64/
  面向最大尺度的第三阶段 JEPA package。详细说明见 `jepa_1024x64/README.md`。

- tokenizer/
  不同尺度 JEPA pipeline 共用的 tokenizer package。详细说明见 `tokenizer/README.md`。

### 训练入口与启动辅助脚本

可直接运行的命令行入口放在：

- `twm/run_helpers/`

这些文件是对各个 pipeline 训练模块的薄包装，暴露统一的命令行参数，可以直接调用，也可以通过 `tools/` 下的辅助启动脚本来调用。

基于 override 配置的辅助启动脚本包括：

- `tools/run_jepa_32x16_from_override.py`
- `tools/run_jepa_128x64_from_override.py`
- `tools/run_jepa_1024x64_from_override.py`

这些 helper 会把配置编号映射成具体的模型配置，并为不同尺度、不同训练方法拼出正确的训练任务。

对应的 Slurm 模板位于：

- `tools/slurm_jepa_32x16.sh`
- `tools/slurm_jepa_128x64_transfer.sh`
- `tools/slurm_jepa_128x64_ema32.sh`
- `tools/slurm_jepa_128x64_ema32_scratch.sh`
- `tools/slurm_jepa_1024x64_transfer_ref32.sh`
- `tools/slurm_jepa_1024x64_transfer_ref128.sh`
- `tools/slurm_jepa_1024x64_ema_ref32.sh`
- `tools/slurm_jepa_1024x64_ema_ref128.sh`
- `tools/slurm_jepa_1024x64_scratch.sh`

### 已确认的外部依赖

这个目录整体上比较独立，但目前仍然会从 `twm/pipelines` 外部导入少量模块：

- `twm.datasets`
  训练入口用它来构建 dataloader。

- `twm.noise_generators`
  被 `jepa_32x16/utils.py` 使用。

### 保留的 override 配置子集

当前开源子集一共保留了 20 个 override 配置。
它们的组织方式是：

- `0-8`：loss9 扫描
- `9-19`：topology12 扫描中的补充配置

对应文件是：

- `jepa_override_configs.py`
