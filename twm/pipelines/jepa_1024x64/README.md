# jepa_1024x64

## English

### Purpose

This directory contains the third-stage JEPA models for the 1024x64 scale. These models combine information from lower scales and define the largest-scale latent prediction stage in the project.

### File overview

- `__init__.py`
  Re-exports the main 1024x64 pipeline and training interfaces.

- `pipeline.py`
  Defines the standard tri-scale 1024x64 JEPA pipeline and mask split logic.

- `pipeline_ema.py`
  Defines the EMA-based 1024x64 pipeline variant.

- `training.py`
  Implements the standard 1024x64 transfer training configuration and training loop.

- `training_ema.py`
  Implements the EMA-based 1024x64 training entrypoint.

- `training_ema_scratch.py`
  Implements the scratch-style 1024x64 training entrypoint with the EMA pipeline structure.

---

## 中文

### 作用

这个目录包含 1024x64 尺度的第三阶段 JEPA 模型。这些模型组合更低尺度的信息，并定义了项目中最大尺度的 latent 预测阶段。

### 文件说明

- `__init__.py`
  对外导出 1024x64 的主要 pipeline 和训练接口。

- `pipeline.py`
  定义标准的三尺度 1024x64 JEPA pipeline 及其 mask split 逻辑。

- `pipeline_ema.py`
  定义带 EMA teacher 的 1024x64 pipeline 变体。

- `training.py`
  实现标准 1024x64 transfer 训练配置和训练循环。

- `training_ema.py`
  实现 EMA 版 1024x64 训练入口。

- `training_ema_scratch.py`
  实现基于 EMA pipeline 结构的 scratch 风格 1024x64 训练入口。
