from __future__ import annotations

import torch
from torch import Tensor, nn

from .rope import TwoDimensionalRotaryEmbedding, apply_2d_rope


class RopeSelfAttention(nn.Module):
    """Self-attention over sparse visible time-band cells.

    Expected input:
    - x: [B, M, Nt, E]
    - positions_2d: [M, 2]

    Here M is the number of visible time-band cells kept for attention.
    The Nt axis remains inside each cell and is flattened internally only for
    the attention computation.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim={embed_dim} must be divisible by num_heads={num_heads}')
        head_dim = embed_dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError(f'head_dim={head_dim} must be divisible by 4 for 2D RoPE')

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = TwoDimensionalRotaryEmbedding(head_dim)

    def forward(self, x: Tensor, positions_2d: Tensor, valid_cell_mask: Tensor | None = None) -> Tensor:
        batch_size, num_cells, num_tokens, embed_dim = x.shape
        seq_len = num_cells * num_tokens
        flat_x = x.reshape(batch_size, seq_len, embed_dim)

        qkv = self.qkv(flat_x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if positions_2d.ndim == 2:
            expanded_positions = positions_2d[:, None, :].expand(num_cells, num_tokens, 2)
            flat_positions = expanded_positions.reshape(seq_len, 2)
        elif positions_2d.ndim == 3:
            expanded_positions = positions_2d[:, :, None, :].expand(batch_size, num_cells, num_tokens, 2)
            flat_positions = expanded_positions.reshape(batch_size, seq_len, 2)
        else:
            raise ValueError(f'Expected positions_2d shape [M, 2] or [B, M, 2], got {tuple(positions_2d.shape)}')
        cos_t, sin_t, cos_b, sin_b = self.rope(flat_positions, dtype=q.dtype, device=q.device)
        q = apply_2d_rope(q, cos_t, sin_t, cos_b, sin_b)
        k = apply_2d_rope(k, cos_t, sin_t, cos_b, sin_b)

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if valid_cell_mask is not None:
            token_valid_mask = valid_cell_mask[:, :, None].expand(batch_size, num_cells, num_tokens).reshape(batch_size, seq_len)
            attn_scores = attn_scores.masked_fill(~token_valid_mask[:, None, None, :], float('-inf'))
            all_invalid = ~token_valid_mask.any(dim=1)
            if all_invalid.any():
                attn_scores[all_invalid] = 0.0
        attn = torch.softmax(attn_scores, dim=-1)
        if valid_cell_mask is not None:
            attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        return out.view(batch_size, num_cells, num_tokens, embed_dim)


class RopeTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RopeSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, positions_2d: Tensor, valid_cell_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), positions_2d=positions_2d, valid_cell_mask=valid_cell_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TimeBandViT(nn.Module):
    """Transformer over sparse visible channel cells.

    Input contract:
    - x: [B, M, Nt, Lt]
      B: batch size
      M: number of visible time-band cells
      Nt: number of token slots per cell
      Lt: token embedding dimension
    - positions_2d: [M, 2]
      each row stores [time_index, band_index] for one visible cell

    Positional handling:
    - 2D RoPE is applied only to the time/band coordinates
    - a learnable additive embedding is used for the sequential Nt axis
    """

    def __init__(
        self,
        embed_dim: int,
        num_cell_tokens: int,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_cell_tokens <= 0:
            raise ValueError(f'num_cell_tokens must be positive, got {num_cell_tokens}')
        self.num_cell_tokens = int(num_cell_tokens)
        self.token_pos_embed = nn.Parameter(torch.zeros(self.num_cell_tokens, embed_dim))
        nn.init.normal_(self.token_pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            RopeTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, positions_2d: Tensor, valid_cell_mask: Tensor | None = None) -> Tensor:
        # if x.ndim != 4:
        #     raise ValueError(f'Expected input shape [B, M, Nt, E], got {tuple(x.shape)}')
        # if positions_2d.ndim != 2 or positions_2d.shape[-1] != 2:
        #     raise ValueError(f'Expected positions_2d shape [M, 2], got {tuple(positions_2d.shape)}')
        # if x.shape[2] != self.num_cell_tokens:
        #     raise ValueError(f'Expected Nt={self.num_cell_tokens}, got {x.shape[2]}')
        # if x.shape[1] != positions_2d.shape[0]:
        #     raise ValueError(f'M mismatch between x and positions_2d: {x.shape[1]} vs {positions_2d.shape[0]}')

        x = x + self.token_pos_embed.view(1, 1, self.num_cell_tokens, -1)
        for block in self.blocks:
            x = block(x, positions_2d=positions_2d, valid_cell_mask=valid_cell_mask)
        return self.norm(x)
