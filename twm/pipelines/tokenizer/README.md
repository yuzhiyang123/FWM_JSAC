# tokenizer

## English

### Purpose

This directory contains the shared tokenizer implementations used by the JEPA pipelines at different scales.

### File overview

- `__init__.py`
  Re-exports tokenizer builders and base interfaces.

- `base.py`
  Defines the abstract tokenizer interface and common validation/output-norm behavior.

- `cnn_blocks.py`
  Defines reusable CNN building blocks used by convolutional tokenizer variants and recovery decoders.

- `pure_cnn_32x16.py`
  Defines the 32x16 CNN tokenizer family.

- `vit_128x64.py`
  Defines the ViT-style tokenizer for the 128x64 scale.

- `vit_1024x64.py`
  Defines the ViT-style tokenizer for the 1024x64 scale.

---

## 中文

### 作用

这个目录包含 JEPA 各尺度 pipeline 共用的 tokenizer 实现。

### 文件说明

- `__init__.py`
  对外导出 tokenizer 构造器和基础接口。

- `base.py`
  定义抽象 tokenizer 接口，以及通用的输入校验和输出归一化行为。

- `cnn_blocks.py`
  定义卷积式 tokenizer 变体和 recovery decoder 会复用的 CNN 基础模块。

- `pure_cnn_32x16.py`
  定义 32x16 的 CNN tokenizer 系列。

- `vit_128x64.py`
  定义 128x64 尺度使用的 ViT 风格 tokenizer。

- `vit_1024x64.py`
  定义 1024x64 尺度使用的 ViT 风格 tokenizer。
