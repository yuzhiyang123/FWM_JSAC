from __future__ import annotations

import torch
from torch import nn

from twm.noise_generators import (
    UniformGrid32x16QuadrantSampler,
    UniformGrid128x64Sampler,
    UniformGridSampleAndNoise,
    UniformGridSampler,
)
from ..tokenizer import build_32x16, build_128x64, build_1024x64



class PerPathRMSNormalize(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            return x
        reduce_dims = tuple(range(1, x.ndim))
        rms = x.pow(2).mean(dim=reduce_dims, keepdim=True).sqrt().clamp_min(self.eps)
        return x / rms


def _wrap_with_rms_normalization(module: nn.Module) -> nn.Module:
    return nn.Sequential(PerPathRMSNormalize(), module)


def build_preprocessor(name: str, kwargs: dict | None = None) -> nn.Module:
    key = str(name).strip().lower()
    params = dict(kwargs or {})
    if key in {'', 'none', 'identity'}:
        return _wrap_with_rms_normalization(nn.Identity())
    if key == 'uniform_grid':
        return _wrap_with_rms_normalization(
            UniformGridSampler(
                stride_c=int(params.get('stride_c', 2)),
                stride_n=int(params.get('stride_n', 2)),
            )
        )
    if key == 'uniform_grid_noise':
        return _wrap_with_rms_normalization(
            UniformGridSampleAndNoise(
                stride_c=int(params.get('stride_c', 2)),
                stride_n=int(params.get('stride_n', 2)),
                noise_std=float(params.get('noise_std', 0.05)),
            )
        )
    if key == 'uniform_grid_32x16_quadrant':
        return _wrap_with_rms_normalization(UniformGrid32x16QuadrantSampler())
    if key == 'uniform_grid_128x64':
        return _wrap_with_rms_normalization(UniformGrid128x64Sampler())
    if key in {'identity_1024x64', 'uniform_grid_1024x64_identity', 'normalized_identity_1024x64'}:
        return _wrap_with_rms_normalization(nn.Identity())
    raise ValueError('Unknown preprocessor_name. Expected one of: empty/none/identity, uniform_grid, uniform_grid_noise, uniform_grid_32x16_quadrant, uniform_grid_128x64, identity_1024x64')


def build_tokenizer(tokenizer_name: str, num_tokens: int, token_dim: int, output_norm: str = 'batchnorm') -> nn.Module:
    key = str(tokenizer_name).strip().lower()
    if key.startswith('32x16_'):
        variant = key.split('_', 1)[1]
        return build_32x16(num_tokens=num_tokens, token_dim=token_dim, config_name=variant, output_norm=output_norm)
    if key.startswith('128x64_'):
        variant = key.split('_', 1)[1]
        return build_128x64(num_tokens=num_tokens, token_dim=token_dim, config_name=variant, output_norm=output_norm)
    if key.startswith('1024x64_'):
        variant = key.split('_', 1)[1]
        return build_1024x64(num_tokens=num_tokens, token_dim=token_dim, config_name=variant, output_norm=output_norm)
    raise ValueError('Unknown tokenizer_name. Expected one of: 32x16_s, 32x16_m, 32x16_a, 32x16_l, 128x64_s, 128x64_m, 128x64_l, 1024x64_xl')
