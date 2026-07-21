from __future__ import annotations

import torch
from torch import Tensor, nn


class AbstractSamplePreprocessor(nn.Module):
    """Base interface for per-cell preprocessing before tokenization."""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


def uniform_grid_sample(x: Tensor, stride_c: int = 2, stride_n: int = 2) -> tuple[Tensor, Tensor]:
    stride_c = max(1, int(stride_c))
    stride_n = max(1, int(stride_n))
    c_dim = int(x.shape[-2])
    n_dim = int(x.shape[-1])

    start_c = int(torch.randint(0, stride_c, (1,), device=x.device).item())
    start_n = int(torch.randint(0, stride_n, (1,), device=x.device).item())
    idx_c = torch.arange(start_c, c_dim, stride_c, device=x.device)
    idx_n = torch.arange(start_n, n_dim, stride_n, device=x.device)

    mask_2d = torch.zeros((c_dim, n_dim), device=x.device, dtype=torch.bool)
    mask_2d[idx_c.unsqueeze(1), idx_n.unsqueeze(0)] = True
    mask = mask_2d.view(*([1] * (x.ndim - 2)), c_dim, n_dim)
    sampled = x * mask.to(dtype=x.dtype)
    return sampled, mask


def _build_quadrant_indices_8x8(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    grid = torch.arange(64, device=device, dtype=torch.long).view(8, 8)
    return (
        grid[:4, :4].reshape(-1),
        grid[:4, 4:].reshape(-1),
        grid[4:, :4].reshape(-1),
        grid[4:, 4:].reshape(-1),
    )


class UniformGrid32x16QuadrantSampler(AbstractSamplePreprocessor):
    """Sampler that maps an input with trailing shape [C, 64] to [32, 16]."""

    def __init__(self) -> None:
        super().__init__()
        self.stride_c = 32
        self.stride_n = 1

    def forward(self, x: Tensor) -> Tensor:
        start_c = int(torch.randint(0, self.stride_c, (1,), device=x.device).item())
        idx_c = torch.arange(start_c, x.shape[-2], self.stride_c, device=x.device)

        quadrants = _build_quadrant_indices_8x8(x.device)
        quadrant_idx = int(torch.randint(0, 4, (1,), device=x.device).item())
        idx_n = quadrants[quadrant_idx]
        return x.index_select(-2, idx_c).index_select(-1, idx_n)


class UniformGrid128x64Sampler(AbstractSamplePreprocessor):
    """Simple uniform-grid downsampler that maps trailing [1024, 64] to [128, 64]."""

    def __init__(self) -> None:
        super().__init__()
        self.stride_c = 8
        self.stride_n = 1

    def forward(self, x: Tensor) -> Tensor:
        start_c = int(torch.randint(0, self.stride_c, (1,), device=x.device).item())
        idx_c = torch.arange(start_c, x.shape[-2], self.stride_c, device=x.device)
        idx_n = torch.arange(0, x.shape[-1], self.stride_n, device=x.device)
        return x.index_select(-2, idx_c).index_select(-1, idx_n)


class UniformGridSampler(AbstractSamplePreprocessor):
    def __init__(self, stride_c: int = 2, stride_n: int = 2) -> None:
        super().__init__()
        self.stride_c = max(1, int(stride_c))
        self.stride_n = max(1, int(stride_n))

    def forward(self, x: Tensor) -> Tensor:
        sampled, _ = uniform_grid_sample(x, stride_c=self.stride_c, stride_n=self.stride_n)
        return sampled


class GaussianNoiseOnly(AbstractSamplePreprocessor):
    def __init__(self, noise_std: float = 0.05) -> None:
        super().__init__()
        self.noise_std = float(max(0.0, noise_std))

    def forward(self, x: Tensor) -> Tensor:
        if self.noise_std <= 0.0:
            return x
        return x + torch.randn_like(x) * self.noise_std


class UniformGridSampleAndNoise(AbstractSamplePreprocessor):
    def __init__(self, stride_c: int = 2, stride_n: int = 2, noise_std: float = 0.05) -> None:
        super().__init__()
        self.sampler = UniformGridSampler(stride_c=stride_c, stride_n=stride_n)
        self.noise_std = float(max(0.0, noise_std))

    def forward(self, x: Tensor) -> Tensor:
        sampled, mask = uniform_grid_sample(
            x,
            stride_c=self.sampler.stride_c,
            stride_n=self.sampler.stride_n,
        )
        if self.noise_std <= 0.0:
            return sampled
        noise = torch.randn_like(sampled) * self.noise_std
        return sampled + noise * mask.to(dtype=sampled.dtype)


class RandomStdGaussianNoise(AbstractSamplePreprocessor):
    def __init__(self, max_noise_std: float = 6e-5) -> None:
        super().__init__()
        self.max_noise_std = float(max(0.0, max_noise_std))

    def forward(self, x: Tensor) -> Tensor:
        if self.max_noise_std <= 0.0:
            return x
        if x.ndim == 0:
            std = torch.rand((), device=x.device, dtype=x.dtype) * self.max_noise_std
        else:
            std_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            std = torch.rand(std_shape, device=x.device, dtype=x.dtype) * self.max_noise_std
        return x + torch.randn_like(x) * std
