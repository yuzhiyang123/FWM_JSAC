# jepa_32x16

## English

### Purpose

This directory contains the core 32x16 JEPA implementation. It is the first-stage model family in the project and provides the main latent prediction pipeline used by later multiscale stages.

### File overview

- `__init__.py`
  Re-exports the main public interfaces of the 32x16 JEPA package.

- `default_pipeline_configs.py`
  Defines the preset topology/config builders used to instantiate standard 32x16 model variants.

- `pipeline.py`
  Defines the main `TimeBandJEPAPipeline`, including tokenization, masking, student-teacher prediction, optional auxiliary heads, save/load, and latent encoding.

- `training.py`
  Implements the training configuration, loss components, scheduler helpers, and the main training/evaluation entrypoints for 32x16 JEPA.

- `time_freq_band.py`
  Defines the mask generators used by 32x16 JEPA, including random and curriculum-based masking.

- `vit.py`
  Defines the transformer backbone used for latent encoding and prediction.

- `rope.py`
  Implements rotary positional embedding utilities used by the transformer layers.

- `utils.py`
  Provides helper builders for preprocessors and tokenizers.

- `dit32x16.py`
  Contains DiT-style diffusion modules used by the optional diffusion auxiliary branch.

- `recovery/`
  Contains the auxiliary recovery decoder and its probe-style training helpers.

- `recovery_diffusion/`
  Contains the auxiliary diffusion recovery model and its training helpers.

---

## 中文

### 作用

这个目录包含核心的 32x16 JEPA 实现。它是本项目的第一阶段模型族，并为后续多尺度阶段提供主要的 latent 预测 pipeline。

### 文件说明

- `__init__.py`
  对外导出 32x16 JEPA 的主要公共接口。

- `default_pipeline_configs.py`
  定义标准 32x16 模型变体所使用的拓扑/配置预设构造函数。

- `pipeline.py`
  定义主 `TimeBandJEPAPipeline`，包括 tokenization、mask、student-teacher 预测、可选 auxiliary head、save/load 和 latent 编码逻辑。

- `training.py`
  实现 32x16 JEPA 的训练配置、loss 组件、scheduler 辅助函数以及主训练/评估入口。

- `time_freq_band.py`
  定义 32x16 JEPA 使用的 mask generator，包括随机 mask 和 curriculum mask。

- `vit.py`
  定义用于 latent 编码与预测的 transformer 主体。

- `rope.py`
  实现 transformer 层使用的 rotary positional embedding 工具。

- `utils.py`
  提供 preprocessor 和 tokenizer 的构造辅助函数。

- `dit32x16.py`
  包含可选 diffusion auxiliary 分支使用的 DiT 风格模块。

- `recovery/`
  包含 auxiliary recovery decoder 以及配套的 probe 风格训练辅助代码。

- `recovery_diffusion/`
  包含 auxiliary diffusion recovery 模型以及配套训练辅助代码。
