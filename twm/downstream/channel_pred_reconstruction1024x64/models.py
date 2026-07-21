from __future__ import annotations

import torch
from torch import Tensor, nn


class _RegressorBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=max(1, int(num_heads)), batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=max(1, int(num_heads)), batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

    def forward(self, x: Tensor, token_ctx: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x)
        x = x + self.cross_attn(h, token_ctx, token_ctx, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class PilotCrossAttentionPatchRegressor1024x64(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        pilot_channels: int = 2,
        out_channels: int = 2,
        height: int = 1024,
        width: int = 64,
        patch_size: tuple[int, int] = (16, 16),
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.pilot_channels = int(pilot_channels)
        self.out_channels = int(out_channels)
        self.height = int(height)
        self.width = int(width)
        self.patch_h = int(patch_size[0])
        self.patch_w = int(patch_size[1])
        if self.height % self.patch_h != 0 or self.width % self.patch_w != 0:
            raise ValueError(f'Patch size {patch_size} must divide {(self.height, self.width)}')
        self.grid_h = self.height // self.patch_h
        self.grid_w = self.width // self.patch_w
        self.num_patches = self.grid_h * self.grid_w
        self.hidden_dim = int(hidden_dim)
        self.pilot_embed = nn.Conv2d(self.pilot_channels, self.hidden_dim, kernel_size=(self.patch_h, self.patch_w), stride=(self.patch_h, self.patch_w), bias=True)
        self.token_proj = nn.Linear(self.token_dim, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_dim))
        self.blocks = nn.ModuleList([
            _RegressorBlock(self.hidden_dim, num_heads=int(num_heads), mlp_ratio=float(mlp_ratio))
            for _ in range(max(1, int(depth)))
        ])
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.final = nn.Linear(self.hidden_dim, self.out_channels * self.patch_h * self.patch_w)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, pilot_estimate: Tensor, tokens: Tensor) -> Tensor:
        expected = (self.pilot_channels, self.height, self.width)
        if pilot_estimate.ndim != 4 or tuple(pilot_estimate.shape[1:]) != expected:
            raise ValueError(f'Expected pilot_estimate [B, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(pilot_estimate.shape)}')
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_dim:
            raise ValueError(f'Expected tokens [B, Nt, {self.token_dim}], got {tuple(tokens.shape)}')
        x = self.pilot_embed(pilot_estimate).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        token_ctx = self.token_proj(tokens)
        for block in self.blocks:
            x = block(x, token_ctx)
        patch_values = self.final(self.final_norm(x))
        batch_size = pilot_estimate.shape[0]
        patch_values = patch_values.view(batch_size, self.grid_h, self.grid_w, self.out_channels, self.patch_h, self.patch_w)
        patch_values = patch_values.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patch_values.view(batch_size, self.out_channels, self.height, self.width)
