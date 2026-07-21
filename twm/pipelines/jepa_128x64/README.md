# jepa_128x64

## English

### Purpose

This directory contains the second-stage JEPA models for the 128x64 scale. These models are built on top of the 32x16 stage and support transfer, EMA, and scratch-style training variants.

### File overview

- `__init__.py`
  Re-exports the main 128x64 pipeline and training interfaces.

- `pipeline.py`
  Defines the standard dual-scale 128x64 JEPA pipeline and mask split logic.

- `pipeline_ema32.py`
  Defines the EMA-based 128x64 pipeline variant.

- `training.py`
  Implements the standard 128x64 transfer training configuration and training loop.

- `training_ema32.py`
  Implements the EMA-based 128x64 training entrypoint.

- `training_ema32_scratch.py`
  Implements the scratch-style 128x64 training entrypoint with the EMA pipeline structure.

---

## 中文

### 作用

这个目录包含 128x64 尺度的第二阶段 JEPA 模型。这些模型建立在 32x16 阶段之上，并支持 transfer、EMA 和 scratch 风格的训练变体。

### 文件说明

- `__init__.py`
  对外导出 128x64 的主要 pipeline 和训练接口。

- `pipeline.py`
  定义标准的双尺度 128x64 JEPA pipeline 及其 mask split 逻辑。

- `pipeline_ema32.py`
  定义带 EMA teacher 的 128x64 pipeline 变体。

- `training.py`
  实现标准 128x64 transfer 训练配置和训练循环。

- `training_ema32.py`
  实现 EMA 版 128x64 训练入口。

- `training_ema32_scratch.py`
  实现基于 EMA pipeline 结构的 scratch 风格 128x64 训练入口。
