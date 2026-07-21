# twm/run_helpers

## English

### Purpose

This directory contains the runnable command-line entrypoints used by the open-source release. These files are thin wrappers around the model and training code in twm/pipelines and twm/downstream. They expose the arguments needed for single-run training or evaluation, while the Slurm templates in tools provide cluster-side submission examples.

### JEPA training entrypoints

- run_jepa_32x16_training.py: first-stage 32x16 JEPA pretraining.
- run_jepa_128x64_training.py: second-stage 128x64 transfer training from a 32x16 pretrained model.
- run_jepa_128x64_ema32_training.py: second-stage 128x64 EMA-based training.
- run_jepa_128x64_ema32_scratch_training.py: second-stage 128x64 scratch training.
- run_jepa_1024x64_training.py: third-stage 1024x64 transfer training.
- run_jepa_1024x64_ema_training.py: third-stage 1024x64 EMA-based training.
- run_jepa_1024x64_ema_scratch_training.py: third-stage 1024x64 scratch training.

### Downstream training entrypoints

- run_downstream_common.py: shared configuration and utility layer used by the downstream entrypoints.
- run_downstream_regression32x16.py: standard 32x16 downstream regression task.
- run_downstream_token_only_regression32x16.py: 32x16 token-only downstream variant.
- run_downstream_pilot_only_regression32x16.py: 32x16 pilot-only downstream variant.
- run_downstream_reconstruction1024x64.py: full 1024x64 downstream reconstruction task.
- run_downstream_token_only_reconstruction1024x64.py: 1024x64 token-only downstream training variants, including the proposed and baseline settings used in the paper.

### Downstream evaluation entrypoints

- run_downstream_token_only_regression32x16_test.py: 32x16 token-only downstream evaluation.
- run_downstream_token_only_reconstruction1024x64_proposed_test.py: 1024x64 proposed-model evaluation. This entrypoint supports the proposed NoCal and Cal settings and reports reconstruction NMSE together with the corresponding communication metrics.
- run_downstream_token_only_reconstruction1024x64_baseline_test.py: 1024x64 baseline evaluation for zero-condition and scratch baselines under both NoCal and Cal settings.

---

## 中文

### 作用

本目录包含开源版本中可直接运行的命令行入口。这些文件是 twm/pipelines 与 twm/downstream 中模型和训练代码的轻量封装，用来暴露单次训练或评测所需的参数；集群提交示例则由 tools 中的 Slurm 模板提供。

### JEPA 训练入口

- run_jepa_32x16_training.py：第一阶段 32x16 JEPA 预训练。
- run_jepa_128x64_training.py：从 32x16 预训练模型启动第二阶段 128x64 transfer 训练。
- run_jepa_128x64_ema32_training.py：第二阶段 128x64 EMA 训练。
- run_jepa_128x64_ema32_scratch_training.py：第二阶段 128x64 scratch 训练。
- run_jepa_1024x64_training.py：第三阶段 1024x64 transfer 训练。
- run_jepa_1024x64_ema_training.py：第三阶段 1024x64 EMA 训练。
- run_jepa_1024x64_ema_scratch_training.py：第三阶段 1024x64 scratch 训练。

### Downstream 训练入口

- run_downstream_common.py：downstream 入口共用的配置与辅助逻辑。
- run_downstream_regression32x16.py：标准 32x16 downstream regression 任务。
- run_downstream_token_only_regression32x16.py：32x16 token-only downstream 变体。
- run_downstream_pilot_only_regression32x16.py：32x16 pilot-only downstream 变体。
- run_downstream_reconstruction1024x64.py：完整 1024x64 downstream reconstruction 任务。
- run_downstream_token_only_reconstruction1024x64.py：1024x64 token-only downstream 训练变体，包含论文中的 proposed 与 baseline 设置。

### Downstream 测试入口

- run_downstream_token_only_regression32x16_test.py：32x16 token-only downstream 测试。
- run_downstream_token_only_reconstruction1024x64_proposed_test.py：1024x64 proposed 模型测试，支持 NoCal 与 Cal，并输出 reconstruction NMSE 及对应通信指标。
- run_downstream_token_only_reconstruction1024x64_baseline_test.py：1024x64 baseline 测试，用于 zero-condition 与 scratch baseline，并覆盖 NoCal 与 Cal 两类设置。
