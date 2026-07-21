from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from twm.downstream.dit32x16 import DiT32x16Output, PilotTokenConditionalDiT32x16


@dataclass
class ChannelPredictionOutput:
    predicted_x0: Tensor

    @property
    def predicted_noise(self) -> Tensor:
        return self.predicted_x0


class LatentHistoryPredictor(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.gru = nn.GRU(self.feature_dim, self.feature_dim, num_layers=max(1, int(num_layers)), batch_first=True)

    def forward(self, history_latents: Tensor, history_mask: Tensor) -> Tensor:
        if history_latents.ndim != 4:
            raise ValueError(f'Expected history_latents [B, T, N, D], got {tuple(history_latents.shape)}')
        if history_mask.shape != history_latents.shape[:3]:
            raise ValueError(f'Expected history_mask shape {tuple(history_latents.shape[:3])}, got {tuple(history_mask.shape)}')
        batch_size, time_steps, num_bands, feature_dim = history_latents.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f'Expected feature_dim={self.feature_dim}, got {feature_dim}')
        flat_history = history_latents.permute(0, 2, 1, 3).reshape(batch_size * num_bands, time_steps, feature_dim)
        flat_mask = history_mask.permute(0, 2, 1).reshape(batch_size * num_bands, time_steps)
        lengths = flat_mask.sum(dim=1).to(dtype=history_mask.dtype)
        output = history_latents.new_zeros(batch_size * num_bands, feature_dim)
        valid = lengths > 0
        if valid.any():
            packed = nn.utils.rnn.pack_padded_sequence(flat_history[valid], lengths[valid].cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
            output[valid] = hidden[-1]
        return output.view(batch_size, num_bands, feature_dim)


class _PilotSequenceFusionBlock(nn.Module):
    def __init__(self, feature_dim: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads=max(1, int(num_heads)), dropout=float(dropout), batch_first=True)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 4 * feature_dim),
            nn.GELU(),
            nn.Linear(4 * feature_dim, feature_dim),
        )

    def forward(self, x: Tensor, *, key_padding_mask: Tensor | None) -> Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class _PilotTokenizerInputEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        input_hw: tuple[int, int],
        num_tokens: int,
        token_dim: int,
        output_norm: str = 'layernorm',
        hidden_dim: int = 1024,
        depth: int = 6,
        patch_size: tuple[int, int] = (32, 16),
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.input_hw = tuple(int(v) for v in input_hw)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.output_norm = str(output_norm).strip().lower()
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        patch_h, patch_w = self.patch_size
        height, width = self.input_hw
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f'Pilot tokenizer expects input_hw divisible by patch_size, got input_hw={self.input_hw}, patch_size={self.patch_size}'
            )
        self.num_patch_tokens = (height // patch_h) * (width // patch_w)
        patch_area = patch_h * patch_w
        if self.hidden_dim % patch_area != 0:
            raise ValueError(
                f'hidden_dim must be divisible by patch_area, got hidden_dim={self.hidden_dim}, patch_area={patch_area}'
            )
        self.stem_channels = self.hidden_dim // patch_area
        if self.hidden_dim % self.num_tokens != 0:
            raise ValueError(f'hidden_dim={self.hidden_dim} must be divisible by num_tokens={self.num_tokens}')
        if self.output_norm not in {'batchnorm', 'layernorm', 'none'}:
            raise ValueError(f'Unsupported pilot tokenizer output_norm: {output_norm!r}')

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, self.stem_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.stem_channels, self.stem_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.patch_embed = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, self.num_patch_tokens, self.hidden_dim))
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_tokens, self.hidden_dim))
        self.blocks = nn.ModuleList(
            _PilotSequenceFusionBlock(
                self.hidden_dim,
                num_heads=max(1, self.num_tokens),
                dropout=self.dropout,
            )
            for _ in range(self.depth)
        )
        self.norm_out = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.token_dim)
        if self.output_norm == 'batchnorm':
            self.output_batch_norm = nn.BatchNorm1d(self.num_tokens * self.token_dim, affine=False)
            self.output_layer_norm = None
        elif self.output_norm == 'layernorm':
            self.output_batch_norm = None
            self.output_layer_norm = nn.LayerNorm(self.token_dim, elementwise_affine=False)
        else:
            self.output_batch_norm = None
            self.output_layer_norm = None
        nn.init.normal_(self.patch_pos_embed, std=0.02)
        nn.init.normal_(self.query_tokens, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f'Expected pilot tokenizer input [M, C, H, W], got {tuple(x.shape)}')
        if x.shape[1] != self.input_channels:
            raise ValueError(f'Expected input_channels={self.input_channels}, got {x.shape[1]}')
        if tuple(x.shape[-2:]) != self.input_hw:
            raise ValueError(f'Expected input_hw={self.input_hw}, got {tuple(x.shape[-2:])}')
        batch_size = x.shape[0]
        stem = self.stem(x)
        patches = self.patch_embed(stem).transpose(1, 2)
        expected_patch_shape = (batch_size, self.num_patch_tokens, self.hidden_dim)
        if tuple(patches.shape) != expected_patch_shape:
            raise ValueError(f'Expected patch tensor {expected_patch_shape}, got {tuple(patches.shape)}')
        patches = patches + self.patch_pos_embed
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        out = torch.cat([patches, query_tokens], dim=1)
        for block in self.blocks:
            out = block(out, key_padding_mask=None)
        out = self.norm_out(out)
        tokens = self.output_proj(out[:, -self.num_tokens:])
        if self.output_batch_norm is not None:
            tokens = self.output_batch_norm(tokens.reshape(batch_size, -1)).view_as(tokens)
        elif self.output_layer_norm is not None:
            tokens = self.output_layer_norm(tokens)
        return tokens


class PilotAwareMLPAlignHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        feature_dim: int,
        num_tokens: int,
        token_dim: int,
        input_hw: tuple[int, int],
        input_mode: str = 'tok',
        num_heads: int = 2,
        dropout: float = 0.0,
        max_positions: int = 8,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.input_hw = tuple(int(v) for v in input_hw)
        self.max_positions = int(max_positions)
        self.input_mode = str(input_mode).strip().lower()
        if self.num_tokens * self.token_dim != self.feature_dim:
            raise ValueError(
                f'Expected feature_dim == num_tokens * token_dim, got feature_dim={self.feature_dim}, num_tokens={self.num_tokens}, token_dim={self.token_dim}'
            )
        if self.input_mode == 'linear':
            self.input_projector = nn.Sequential(
                nn.Linear(self.input_dim, self.feature_dim),
                nn.GELU(),
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.GELU(),
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.GELU(),
                nn.Linear(self.feature_dim, self.feature_dim),
            )
        elif self.input_mode == 'tok':
            # Keep the tokenizer-style inductive bias, but use a much smaller
            # encoder than the full 1024x64 tokenizer to reduce calibration cost.
            self.input_projector = _PilotTokenizerInputEncoder(
                input_channels=2,
                input_hw=self.input_hw,
                num_tokens=self.num_tokens,
                token_dim=self.token_dim,
                output_norm='layernorm',
                hidden_dim=512,
                depth=2,
                patch_size=(32, 16),
                mlp_ratio=2.0,
                dropout=float(dropout),
            )
        else:
            raise ValueError(f'Unsupported pilot align input_mode: {input_mode!r}')
        self.position_embed = nn.Parameter(torch.zeros(1, self.max_positions, self.feature_dim))
        self.type_embed_pred = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.type_embed_pilot = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        # Keep the fusion stage lightweight: a shallow attention stack is
        # enough for pilot/latent conditioning while avoiding the previous cost.
        self.blocks = nn.ModuleList(
            _PilotSequenceFusionBlock(self.feature_dim, num_heads=max(1, int(num_heads)), dropout=float(dropout))
            for _ in range(max(1, int(depth)))
        )
        self.out_norm = nn.LayerNorm(self.feature_dim)
        nn.init.normal_(self.position_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embed_pred, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embed_pilot, mean=0.0, std=0.02)

    def forward(self, predicted_latents: Tensor, preprocessed_current: Tensor, subband_mask: Tensor) -> tuple[Tensor, Tensor]:
        if predicted_latents.ndim != 3:
            raise ValueError(f'Expected predicted_latents [B, N, D], got {tuple(predicted_latents.shape)}')
        if preprocessed_current.ndim != 5:
            raise ValueError(f'Expected preprocessed_current [B, N, C, H, W], got {tuple(preprocessed_current.shape)}')
        if subband_mask.shape != predicted_latents.shape[:2]:
            raise ValueError(f'Expected subband_mask shape {tuple(predicted_latents.shape[:2])}, got {tuple(subband_mask.shape)}')
        batch_size, num_bands, feature_dim = predicted_latents.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f'Expected feature_dim={self.feature_dim}, got {feature_dim}')
        if num_bands > self.max_positions:
            raise ValueError(f'Expected at most {self.max_positions} subbands, got {num_bands}')

        flat_input = preprocessed_current.reshape(batch_size * num_bands, *preprocessed_current.shape[-3:])
        if self.input_mode == 'linear':
            pilot_features = self.input_projector(flat_input.reshape(batch_size * num_bands, -1)).view(batch_size, num_bands, self.feature_dim)
        else:
            pilot_features = self.input_projector(flat_input).flatten(start_dim=1).view(batch_size, num_bands, self.feature_dim)
        pos_embed = self.position_embed[:, :num_bands]
        pred_tokens = predicted_latents + pos_embed + self.type_embed_pred
        pilot_tokens = pilot_features + pos_embed + self.type_embed_pilot

        available_mask = subband_mask.to(dtype=torch.bool)
        obs_mask = torch.zeros_like(available_mask)
        for batch_index in range(batch_size):
            available_idx = available_mask[batch_index].nonzero(as_tuple=False).flatten()
            num_available = int(available_idx.numel())
            if num_available <= 0:
                continue
            max_drop = min(7, num_available - 1)
            num_drop = int(torch.randint(low=0, high=max_drop + 1, size=(1,), device=predicted_latents.device).item())
            if num_drop == 0:
                kept_idx = available_idx
            else:
                perm = torch.randperm(num_available, device=predicted_latents.device)
                keep_mask = torch.ones(num_available, device=predicted_latents.device, dtype=torch.bool)
                keep_mask[perm[:num_drop]] = False
                kept_idx = available_idx[keep_mask]
            obs_mask[batch_index, kept_idx] = True

        observed_counts = obs_mask.sum(dim=1)
        max_joint_tokens = int(num_bands + observed_counts.max().item())
        joint = predicted_latents.new_zeros(batch_size, max_joint_tokens, self.feature_dim)
        key_padding_mask = torch.ones(batch_size, max_joint_tokens, device=predicted_latents.device, dtype=torch.bool)

        for batch_index in range(batch_size):
            joint[batch_index, :num_bands] = pred_tokens[batch_index]
            key_padding_mask[batch_index, :num_bands] = False
            observed_idx = obs_mask[batch_index].nonzero(as_tuple=False).flatten()
            num_observed = int(observed_idx.numel())
            if num_observed > 0:
                joint[batch_index, num_bands:num_bands + num_observed] = pilot_tokens[batch_index, observed_idx]
                key_padding_mask[batch_index, num_bands:num_bands + num_observed] = False

        for block in self.blocks:
            joint = block(joint, key_padding_mask=key_padding_mask)

        calibrated = self.out_norm(joint[:, :num_bands])
        return calibrated, pilot_features


class ChannelPredConditionalDiffusionModel32x16(nn.Module):
    def __init__(self, *, token_dim: int, sample_channels: int = 2, down_block_out_channels: tuple[int, ...] = (64, 128, 128), layers_per_block: int = 1, num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, fusion_mode: str = "split_cross_attn") -> None:
        super().__init__()
        hidden_dim = int(max(down_block_out_channels)) if down_block_out_channels else 128
        depth = max(3, len(tuple(down_block_out_channels)) * max(1, int(layers_per_block)))
        self.fusion_mode = str(fusion_mode)
        self.base = PilotTokenConditionalDiT32x16(token_dim=int(token_dim), sample_channels=int(sample_channels), pilot_channels=3, hidden_dim=hidden_dim, depth=depth, num_heads=8, mlp_ratio=4.0, num_train_timesteps=int(num_train_timesteps), beta_start=float(beta_start), beta_end=float(beta_end))
        self.noise_scheduler = self.base.noise_scheduler
        self.sample_channels = int(sample_channels)
        self.token_dim = int(token_dim)
        self.height = 32
        self.width = 16
        if self.fusion_mode == "gated_fusion":
            self.token_gate = nn.Parameter(torch.tensor(0.0))
            self.pilot_gate = nn.Parameter(torch.tensor(0.0))
        elif self.fusion_mode == "latent_residual":
            self.latent_to_residual = nn.Linear(self.token_dim, self.sample_channels * self.height * self.width)
            nn.init.zeros_(self.latent_to_residual.weight)
            nn.init.zeros_(self.latent_to_residual.bias)
        elif self.fusion_mode == "late_fusion":
            self.latent_to_pilot = nn.Linear(self.token_dim, 2 * self.height * self.width)
            nn.init.zeros_(self.latent_to_pilot.weight)
            nn.init.zeros_(self.latent_to_pilot.bias)
        elif self.fusion_mode not in {"pilot_only", "split_cross_attn"}:
            raise ValueError(f'Unsupported fusion_mode: {self.fusion_mode}')

    def sample_timesteps(self, batch_size: int, device) -> Tensor:
        return self.base.sample_timesteps(batch_size, device)

    def add_noise(self, target: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.base.add_noise(target, noise, timesteps)

    def predict_noise(self, noisy_target: Tensor, predicted_x0: Tensor, timesteps: Tensor) -> Tensor:
        return self.base.predict_noise(noisy_target, predicted_x0, timesteps)

    def predict_x0(self, noisy_target: Tensor, predicted_noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.base.predict_x0(noisy_target, predicted_noise, timesteps)

    def predict_velocity_target(self, target: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.base.predict_velocity_target(target, noise, timesteps)

    def predict_x0_from_velocity(self, noisy_target: Tensor, predicted_v: Tensor, timesteps: Tensor) -> Tensor:
        return self.base.predict_x0_from_velocity(noisy_target, predicted_v, timesteps)

    def predict_noise_from_velocity(self, noisy_target: Tensor, predicted_v: Tensor, timesteps: Tensor) -> Tensor:
        return self.base.predict_noise_from_velocity(noisy_target, predicted_v, timesteps)

    def forward(self, noisy_target: Tensor, pilot_estimate: Tensor, tokens: Tensor, timesteps: Tensor) -> ChannelPredictionOutput:
        pilot_in = pilot_estimate
        tokens_in = tokens
        noisy_in = noisy_target
        if self.fusion_mode == "pilot_only":
            tokens_in = tokens.new_zeros(tokens.shape)
        elif self.fusion_mode == "gated_fusion":
            tg = self.token_gate.sigmoid().to(dtype=tokens.dtype)
            pg = self.pilot_gate.sigmoid().to(dtype=pilot_estimate.dtype)
            tokens_in = tokens * tg
            pilot_in = torch.cat([pilot_estimate[:, :2] * pg, pilot_estimate[:, 2:3]], dim=1)
        elif self.fusion_mode == "latent_residual":
            summary = tokens.mean(dim=1)
            residual = self.latent_to_residual(summary).view(noisy_target.shape[0], self.sample_channels, self.height, self.width)
            noisy_in = noisy_target + residual
        elif self.fusion_mode == "late_fusion":
            summary = tokens.mean(dim=1)
            pilot_delta = self.latent_to_pilot(summary).view(noisy_target.shape[0], 2, self.height, self.width)
            pilot_in = torch.cat([pilot_estimate[:, :2] + pilot_delta, pilot_estimate[:, 2:3]], dim=1)
        out: DiT32x16Output = self.base.forward(noisy_in, pilot_in, tokens_in, timesteps)
        return ChannelPredictionOutput(predicted_x0=out.predicted_x0)
