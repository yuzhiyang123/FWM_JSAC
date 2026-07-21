# tools

## English

### Purpose

This directory contains helper launchers and Slurm submission templates for the open-source release. The helper launchers translate override configuration ids into concrete JEPA training runs. The Slurm scripts provide minimal examples for dataset generation, JEPA training, downstream training, and downstream evaluation.

### JEPA helper launchers

- run_jepa_32x16_from_override.py: selects a configuration id from twm/pipelines/jepa_override_configs.py and launches the corresponding 32x16 JEPA run.
- run_jepa_128x64_from_override.py: selects a configuration id and launches the corresponding 128x64 JEPA run with the requested training method.
- run_jepa_1024x64_from_override.py: selects a configuration id and launches the corresponding 1024x64 JEPA run with the requested training method.

### Dataset-generation Slurm template

- slurm_dataset_gen_01chicago_gt_only.sh: example Slurm script for generating the 01chicago dataset used in the experiments.

### JEPA Slurm templates

- slurm_jepa_32x16.sh: first-stage 32x16 JEPA training.
- slurm_jepa_128x64_transfer.sh: second-stage 128x64 transfer training.
- slurm_jepa_128x64_ema32.sh: second-stage 128x64 EMA training.
- slurm_jepa_128x64_ema32_scratch.sh: second-stage 128x64 scratch training.
- slurm_jepa_1024x64_transfer_ref32.sh: third-stage 1024x64 transfer training from a 32x16 reference.
- slurm_jepa_1024x64_transfer_ref128.sh: third-stage 1024x64 transfer training from a 128x64 reference.
- slurm_jepa_1024x64_ema_ref32.sh: third-stage 1024x64 EMA training from a 32x16 reference.
- slurm_jepa_1024x64_ema_ref128.sh: third-stage 1024x64 EMA training from a 128x64 reference.
- slurm_jepa_1024x64_scratch.sh: third-stage 1024x64 scratch training.

### Downstream Slurm templates

- slurm_downstream_regression32x16.sh: standard 32x16 downstream regression training.
- slurm_downstream_token_only_regression32x16.sh: 32x16 token-only downstream training.
- slurm_downstream_pilot_only_regression32x16.sh: 32x16 pilot-only downstream training.
- slurm_downstream_token_only_regression32x16_test.sh: 32x16 token-only downstream testing.
- slurm_downstream_reconstruction1024x64.sh: full 1024x64 downstream reconstruction training.
- slurm_downstream_token_only_reconstruction1024x64.sh: 1024x64 token-only downstream training.
- slurm_downstream_token_only_reconstruction1024x64_proposed_test.sh: 1024x64 proposed downstream testing.
- slurm_downstream_token_only_reconstruction1024x64_baseline_test.sh: 1024x64 baseline downstream testing.
- test_simple.sh: small shell example kept as a minimal local test helper.

---

## 中文

### 作用

本目录包含开源版本所需的辅助启动脚本和 Slurm 提交模板。其中辅助启动脚本负责把 override 配置编号映射到具体的 JEPA 训练任务；Slurm 脚本则提供数据集生成、JEPA 训练、downstream 训练与 downstream 测试的最简提交示例。

### JEPA 辅助启动脚本

- run_jepa_32x16_from_override.py：从 twm/pipelines/jepa_override_configs.py 中选择一个配置编号，并启动对应的 32x16 JEPA 训练。
- run_jepa_128x64_from_override.py：选择一个配置编号，并按指定训练方法启动对应的 128x64 JEPA 训练。
- run_jepa_1024x64_from_override.py：选择一个配置编号，并按指定训练方法启动对应的 1024x64 JEPA 训练。

### 数据集生成 Slurm 模板

- slurm_dataset_gen_01chicago_gt_only.sh：用于生成实验中 01chicago 数据集的 Slurm 示例脚本。

### JEPA Slurm 模板

- slurm_jepa_32x16.sh：第一阶段 32x16 JEPA 训练。
- slurm_jepa_128x64_transfer.sh：第二阶段 128x64 transfer 训练。
- slurm_jepa_128x64_ema32.sh：第二阶段 128x64 EMA 训练。
- slurm_jepa_128x64_ema32_scratch.sh：第二阶段 128x64 scratch 训练。
- slurm_jepa_1024x64_transfer_ref32.sh：以 32x16 为参考的第三阶段 1024x64 transfer 训练。
- slurm_jepa_1024x64_transfer_ref128.sh：以 128x64 为参考的第三阶段 1024x64 transfer 训练。
- slurm_jepa_1024x64_ema_ref32.sh：以 32x16 为参考的第三阶段 1024x64 EMA 训练。
- slurm_jepa_1024x64_ema_ref128.sh：以 128x64 为参考的第三阶段 1024x64 EMA 训练。
- slurm_jepa_1024x64_scratch.sh：第三阶段 1024x64 scratch 训练。

### Downstream Slurm 模板

- slurm_downstream_regression32x16.sh：标准 32x16 downstream regression 训练。
- slurm_downstream_token_only_regression32x16.sh：32x16 token-only downstream 训练。
- slurm_downstream_pilot_only_regression32x16.sh：32x16 pilot-only downstream 训练。
- slurm_downstream_token_only_regression32x16_test.sh：32x16 token-only downstream 测试。
- slurm_downstream_reconstruction1024x64.sh：完整 1024x64 downstream reconstruction 训练。
- slurm_downstream_token_only_reconstruction1024x64.sh：1024x64 token-only downstream 训练。
- slurm_downstream_token_only_reconstruction1024x64_proposed_test.sh：1024x64 proposed downstream 测试。
- slurm_downstream_token_only_reconstruction1024x64_baseline_test.sh：1024x64 baseline downstream 测试。
- test_simple.sh：保留下来的一个最小本地测试 shell 示例。
