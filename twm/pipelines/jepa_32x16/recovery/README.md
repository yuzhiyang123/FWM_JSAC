# jepa_32x16/recovery

## English

### Purpose

This directory contains the explicit reconstruction auxiliary head used by 32x16 JEPA when recovery supervision is enabled.

### File overview

- `__init__.py`
  Re-exports the recovery model and training helpers.

- `models.py`
  Defines the recovery decoder architectures that map latent tokens back to channel-like outputs.

- `training.py`
  Defines probe-style helper tasks and batch-level train/eval functions for the recovery head.

---

## 中文

### 作用

这个目录包含 32x16 JEPA 在启用 recovery supervision 时使用的显式重建 auxiliary head。

### 文件说明

- `__init__.py`
  对外导出 recovery 模型和训练辅助接口。

- `models.py`
  定义将 latent tokens 映射回类信道输出的 recovery decoder 结构。

- `training.py`
  定义 recovery head 的 probe 风格辅助 task，以及 batch 级的训练/评估函数。
