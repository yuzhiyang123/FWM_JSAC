from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from .base import AbstractCellTokenizer

__all__ = ["build_1024x64"]


@dataclass
class ViTCompressor1024x64Config:
    num_tokens: int
    token_dim: int
    output_norm: str = 'batchnorm'
    depth: int = 6
    hidden_dim: int = 1024
    stem_channels: int = 2
    patch_size: tuple[int, int] = (32, 16)
    mlp_ratio: float = 2.0
    dropout: float = 0.0


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTCompressor1024x64(AbstractCellTokenizer):
    input_hw = (1024, 64)
    num_patch_tokens = 128

    def __init__(self, config: ViTCompressor1024x64Config) -> None:
        super().__init__(num_tokens=config.num_tokens, token_dim=config.token_dim, output_norm=config.output_norm)
        self.config = config
        self.hidden_dim = int(config.hidden_dim)
        self.stem_channels = int(config.stem_channels)
        self.patch_size = tuple(int(v) for v in config.patch_size)
        if self.hidden_dim % self.stem_channels != 0:
            raise ValueError('hidden_dim must be divisible by stem_channels')
        patch_area = self.patch_size[0] * self.patch_size[1]
        if self.stem_channels * patch_area != self.hidden_dim:
            raise ValueError(
                f'stem_channels * patch_area must equal hidden_dim, got {self.stem_channels} * {patch_area} != {self.hidden_dim}'
            )
        if self.hidden_dim % self.Nt != 0:
            raise ValueError(f'For this tokenizer, hidden_dim={self.hidden_dim} must be divisible by Nt={self.Nt}')

        self.stem = nn.Sequential(
            nn.Conv2d(2, self.stem_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.stem_channels, self.stem_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.patch_embed = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, self.num_patch_tokens, self.hidden_dim))
        self.query_tokens = nn.Parameter(torch.zeros(1, self.Nt, self.hidden_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.Nt,
                mlp_ratio=float(config.mlp_ratio),
                dropout=float(config.dropout),
            )
            for _ in range(int(config.depth))
        ])
        self.norm_out = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.Lt)
        nn.init.normal_(self.patch_pos_embed, std=0.02)
        nn.init.normal_(self.query_tokens, std=0.02)

    def validate_realization_input(self, x: Tensor) -> None:
        if tuple(x.shape[-2:]) != self.input_hw:
            raise ValueError(f'ViTCompressor1024x64 expects H,W={self.input_hw}, got {tuple(x.shape[-2:])}')

    def forward_tokens(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        stem = self.stem(x)
        patches = self.patch_embed(stem).transpose(1, 2)
        if patches.shape[1] != self.num_patch_tokens or patches.shape[2] != self.hidden_dim:
            raise ValueError(
                f'Expected patch tensor [B, {self.num_patch_tokens}, {self.hidden_dim}], got {tuple(patches.shape)}'
            )
        patches = patches + self.patch_pos_embed
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        out = torch.cat([patches, query_tokens], dim=1)
        for block in self.blocks:
            out = block(out)
        out = self.norm_out(out)
        query_out = out[:, -self.Nt:]
        return self.output_proj(query_out)


config_XL = ViTCompressor1024x64Config(num_tokens=1, token_dim=1)


def build_1024x64(num_tokens: int, token_dim: int, config_name: str, output_norm: str = 'batchnorm') -> ViTCompressor1024x64:
    key = str(config_name).strip().lower()
    if key not in {'xl', '1024', '1024x64', 'default'}:
        raise ValueError('Unknown config_name. Expected one of: xl, 1024, 1024x64, default')
    return ViTCompressor1024x64(replace(config_XL, num_tokens=int(num_tokens), token_dim=int(token_dim), output_norm=str(output_norm)))
