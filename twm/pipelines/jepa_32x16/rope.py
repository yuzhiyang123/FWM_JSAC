from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f'RoPE dim must be even, got {dim}')
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, positions: Tensor, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
        freqs = torch.outer(positions.to(device=device, dtype=self.inv_freq.dtype), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


class TwoDimensionalRotaryEmbedding(nn.Module):
    """Axial 2D RoPE over time and band coordinates.

    The input head dimension is split evenly into a time half and a band half.
    Each half uses an independent 1D RoPE.
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f'2D RoPE dim must be divisible by 4, got {dim}')
        axis_dim = dim // 2
        self.time_rope = RotaryEmbedding(axis_dim, base=base)
        self.band_rope = RotaryEmbedding(axis_dim, base=base)
        self.dim = int(dim)
        self.axis_dim = int(axis_dim)

    def forward(self, positions: Tensor, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if positions.ndim == 2 and positions.shape[1] == 2:
            time_pos = positions[:, 0]
            band_pos = positions[:, 1]
            cos_t, sin_t = self.time_rope(time_pos, dtype=dtype, device=device)
            cos_b, sin_b = self.band_rope(band_pos, dtype=dtype, device=device)
            return cos_t, sin_t, cos_b, sin_b
        if positions.ndim == 3 and positions.shape[-1] == 2:
            batch_size, seq_len, _ = positions.shape
            flat_positions = positions.reshape(batch_size * seq_len, 2)
            cos_t, sin_t, cos_b, sin_b = self.forward(flat_positions, dtype=dtype, device=device)
            return (
                cos_t.view(batch_size, seq_len, -1),
                sin_t.view(batch_size, seq_len, -1),
                cos_b.view(batch_size, seq_len, -1),
                sin_b.view(batch_size, seq_len, -1),
            )
        raise ValueError(f'Expected positions with shape [S, 2] or [B, S, 2], got {tuple(positions.shape)}')


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack([-x2, x1], dim=-1)
    return rotated.flatten(start_dim=-2)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        raise ValueError(f'Expected cos/sin with shape [S, D] or [B, S, D], got {tuple(cos.shape)}')
    return (x * cos) + (_rotate_half(x) * sin)


def apply_2d_rope(x: Tensor, cos_t: Tensor, sin_t: Tensor, cos_b: Tensor, sin_b: Tensor) -> Tensor:
    time_half, band_half = torch.chunk(x, 2, dim=-1)
    time_half = apply_rope(time_half, cos_t, sin_t)
    band_half = apply_rope(band_half, cos_b, sin_b)
    return torch.cat([time_half, band_half], dim=-1)
