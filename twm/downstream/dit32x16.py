from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn


@dataclass
class DiT32x16Output:
    predicted_x0: Tensor

    @property
    def predicted_noise(self) -> Tensor:
        return self.predicted_x0


class _SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f'Expected timesteps [B], got {tuple(timesteps.shape)}')
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(1, half - 1))
        args = timesteps.to(dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class _AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear = self.modulation[1]
        nn.init.constant_(linear.weight, 0)
        nn.init.constant_(linear.bias, 0)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _ContextCrossBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, num_contexts: int) -> None:
        super().__init__()
        self.norm1 = _AdaLayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=max(1, int(num_heads)), batch_first=True)
        self.cross_norms = nn.ModuleList([_AdaLayerNorm(hidden_dim) for _ in range(num_contexts)])
        self.cross_attns = nn.ModuleList([nn.MultiheadAttention(hidden_dim, num_heads=max(1, int(num_heads)), batch_first=True) for _ in range(num_contexts)])
        self.norm_out = _AdaLayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * float(mlp_ratio))
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, hidden_dim))

    def forward(self, x: Tensor, cond: Tensor, *contexts: Tensor) -> Tensor:
        h = self.norm1(x, cond)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        for norm, attn, ctx in zip(self.cross_norms, self.cross_attns, contexts):
            q = norm(x, cond)
            x = x + attn(q, ctx, ctx, need_weights=False)[0]
        x = x + self.mlp(self.norm_out(x, cond))
        return x


class _FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, out_channels: int) -> None:
        super().__init__()
        self.norm = _AdaLayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.linear(self.norm(x, cond))


class _BaseDiTDiffusionModel32x16(nn.Module):
    def __init__(self, *, num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(num_train_timesteps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            beta_schedule='linear',
            prediction_type='epsilon',
            clip_sample=False,
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(0, int(self.noise_scheduler.config.num_train_timesteps), (int(batch_size),), device=device, dtype=torch.long)

    def add_noise(self, target: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.noise_scheduler.add_noise(target, noise, timesteps)

    def predict_x0(self, noisy_target: Tensor, predicted_noise: Tensor, timesteps: Tensor) -> Tensor:
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=noisy_target.device, dtype=noisy_target.dtype)
        alpha_bar = alphas_cumprod[timesteps].view(noisy_target.shape[0], 1, 1, 1)
        return (noisy_target - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)

    def predict_noise(self, noisy_target: Tensor, predicted_x0: Tensor, timesteps: Tensor) -> Tensor:
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=noisy_target.device, dtype=noisy_target.dtype)
        alpha_bar = alphas_cumprod[timesteps].view(noisy_target.shape[0], 1, 1, 1)
        return (noisy_target - torch.sqrt(alpha_bar) * predicted_x0) / torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))

    def predict_velocity_target(self, target: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=target.device, dtype=target.dtype)
        alpha_bar = alphas_cumprod[timesteps].view(target.shape[0], 1, 1, 1)
        return torch.sqrt(alpha_bar) * noise - torch.sqrt(1.0 - alpha_bar) * target

    def predict_x0_from_velocity(self, noisy_target: Tensor, predicted_v: Tensor, timesteps: Tensor) -> Tensor:
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=noisy_target.device, dtype=noisy_target.dtype)
        alpha_bar = alphas_cumprod[timesteps].view(noisy_target.shape[0], 1, 1, 1)
        return torch.sqrt(alpha_bar) * noisy_target - torch.sqrt(1.0 - alpha_bar) * predicted_v

    def predict_noise_from_velocity(self, noisy_target: Tensor, predicted_v: Tensor, timesteps: Tensor) -> Tensor:
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=noisy_target.device, dtype=noisy_target.dtype)
        alpha_bar = alphas_cumprod[timesteps].view(noisy_target.shape[0], 1, 1, 1)
        return torch.sqrt(alpha_bar) * predicted_v + torch.sqrt(1.0 - alpha_bar) * noisy_target


class TokenConditionalDiT32x16(_BaseDiTDiffusionModel32x16):
    def __init__(self, *, token_dim: int, sample_channels: int = 2, hidden_dim: int = 256, depth: int = 9, num_heads: int = 8, mlp_ratio: float = 4.0, num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
        self.token_dim = int(token_dim)
        self.sample_channels = int(sample_channels)
        self.height = 32
        self.width = 16
        self.hidden_dim = int(hidden_dim)
        self.x_embed = nn.Conv2d(self.sample_channels, self.hidden_dim, kernel_size=1, bias=True)
        self.token_proj = nn.Linear(self.token_dim, self.hidden_dim)
        self.time_embed = nn.Sequential(_SinusoidalTimeEmbedding(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.height * self.width, self.hidden_dim))
        self.blocks = nn.ModuleList([_ContextCrossBlock(self.hidden_dim, num_heads=int(num_heads), mlp_ratio=float(mlp_ratio), num_contexts=1) for _ in range(max(1, int(depth)))])
        self.final_layer = _FinalLayer(self.hidden_dim, self.sample_channels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        for block in self.blocks:
            for norm in [block.norm1, *block.cross_norms, block.norm_out]:
                norm.reset_parameters()
        self.final_layer.reset_parameters()

    def forward(self, noisy_target: Tensor, tokens: Tensor, timesteps: Tensor) -> DiT32x16Output:
        if noisy_target.ndim != 4 or tuple(noisy_target.shape[1:]) != (self.sample_channels, self.height, self.width):
            raise ValueError(f'Expected noisy_target [B, {self.sample_channels}, {self.height}, {self.width}], got {tuple(noisy_target.shape)}')
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_dim:
            raise ValueError(f'Expected token shape [B, Nt, {self.token_dim}], got {tuple(tokens.shape)}')
        x_tokens = self.x_embed(noisy_target).flatten(2).transpose(1, 2) + self.pos_embed
        token_ctx = self.token_proj(tokens)
        cond = self.time_embed(timesteps)
        for block in self.blocks:
            x_tokens = block(x_tokens, cond, token_ctx)
        predicted_x0 = self.final_layer(x_tokens, cond).transpose(1, 2).reshape(noisy_target.shape[0], self.sample_channels, self.height, self.width)
        return DiT32x16Output(predicted_x0=predicted_x0)


class PilotTokenConditionalDiT32x16(_BaseDiTDiffusionModel32x16):
    def __init__(self, *, token_dim: int, sample_channels: int = 2, pilot_channels: int = 3, hidden_dim: int = 256, depth: int = 9, num_heads: int = 8, mlp_ratio: float = 4.0, num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
        self.token_dim = int(token_dim)
        self.sample_channels = int(sample_channels)
        self.pilot_channels = int(pilot_channels)
        self.height = 32
        self.width = 16
        self.hidden_dim = int(hidden_dim)
        self.x_embed = nn.Conv2d(self.sample_channels, self.hidden_dim, kernel_size=1, bias=True)
        self.pilot_embed = nn.Conv2d(self.pilot_channels, self.hidden_dim, kernel_size=1, bias=True)
        self.token_proj = nn.Linear(self.token_dim, self.hidden_dim)
        self.time_embed = nn.Sequential(_SinusoidalTimeEmbedding(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.pilot_summary = nn.Sequential(nn.Linear(self.pilot_channels * self.height * self.width, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.height * self.width, self.hidden_dim))
        self.blocks = nn.ModuleList([_ContextCrossBlock(self.hidden_dim, num_heads=int(num_heads), mlp_ratio=float(mlp_ratio), num_contexts=2) for _ in range(max(1, int(depth)))])
        self.final_layer = _FinalLayer(self.hidden_dim, self.sample_channels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        for block in self.blocks:
            for norm in [block.norm1, *block.cross_norms, block.norm_out]:
                norm.reset_parameters()
        self.final_layer.reset_parameters()

    def forward(self, noisy_target: Tensor, pilot_estimate: Tensor, tokens: Tensor, timesteps: Tensor) -> DiT32x16Output:
        if noisy_target.ndim != 4 or tuple(noisy_target.shape[1:]) != (self.sample_channels, self.height, self.width):
            raise ValueError(f'Expected noisy_target [B, {self.sample_channels}, {self.height}, {self.width}], got {tuple(noisy_target.shape)}')
        if pilot_estimate.ndim != 4 or tuple(pilot_estimate.shape[1:]) != (self.pilot_channels, self.height, self.width):
            raise ValueError(f'Expected pilot_estimate [B, {self.pilot_channels}, {self.height}, {self.width}], got {tuple(pilot_estimate.shape)}')
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_dim:
            raise ValueError(f'Expected token shape [B, Nt, {self.token_dim}], got {tuple(tokens.shape)}')
        x_tokens = self.x_embed(noisy_target).flatten(2).transpose(1, 2) + self.pos_embed
        pilot_tokens = self.pilot_embed(pilot_estimate).flatten(2).transpose(1, 2) + self.pos_embed
        token_ctx = self.token_proj(tokens)
        cond = self.time_embed(timesteps) + self.pilot_summary(pilot_estimate.flatten(1))
        for block in self.blocks:
            x_tokens = block(x_tokens, cond, pilot_tokens, token_ctx)
        predicted_x0 = self.final_layer(x_tokens, cond).transpose(1, 2).reshape(noisy_target.shape[0], self.sample_channels, self.height, self.width)
        return DiT32x16Output(predicted_x0=predicted_x0)
