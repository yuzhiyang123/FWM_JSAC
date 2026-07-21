# jepa_32x16/recovery_diffusion

## English

### Purpose

This directory contains the diffusion-based auxiliary reconstruction branch for 32x16 JEPA.

### File overview

- `__init__.py`
  Re-exports the diffusion models and training helpers.

- `models.py`
  Defines conditional and unconditional diffusion recovery models built on the local DiT components.

- `training.py`
  Defines helper tasks and batch-level train/eval functions for diffusion-based recovery supervision.

---

## 中文

### 作用

这个目录包含 32x16 JEPA 的 diffusion 型 auxiliary reconstruction 分支。

### 文件说明

- `__init__.py`
  对外导出 diffusion 模型和训练辅助接口。

- `models.py`
  定义建立在本地 DiT 组件上的 conditional / unconditional diffusion recovery 模型。

- `training.py`
  定义 diffusion recovery supervision 的辅助 task，以及 batch 级训练/评估函数。
