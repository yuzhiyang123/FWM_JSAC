from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from twm.pipelines.jepa_32x16.vit import TimeBandViT
from twm.pipelines.jepa_32x16.utils import build_preprocessor, build_tokenizer


@dataclass
class TimeBandViTConfig:
    embed_dim: int | None = None
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0



@dataclass
class DualScaleMaskSplit:
    known_32x16: Tensor
    known_128x64: Tensor
    unknown: Tensor


class DualScaleMaskGenerator(nn.Module):
    def __init__(
        self,
        *,
        final_min_mask_rate: float = 0.1,
        final_max_mask_rate: float = 0.9,
        final_beta_alpha: float = 4.0,
        final_beta_beta: float = 1.2,
        known_split_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.final_min_mask_rate = float(final_min_mask_rate)
        self.final_max_mask_rate = float(final_max_mask_rate)
        self.final_beta_alpha = float(final_beta_alpha)
        self.final_beta_beta = float(final_beta_beta)
        self.known_split_prob = float(known_split_prob)
        if not (0.0 <= self.final_min_mask_rate <= self.final_max_mask_rate <= 1.0):
            raise ValueError('final mask rate range must satisfy 0 <= min <= max <= 1')
        if self.final_beta_alpha <= 0.0 or self.final_beta_beta <= 0.0:
            raise ValueError('beta parameters must be positive')
        if not (0.0 < self.known_split_prob < 1.0):
            raise ValueError('known_split_prob must be in (0, 1)')

    def forward(self, x: Tensor) -> DualScaleMaskSplit:
        if x.ndim < 3:
            raise ValueError(f'Expected input with at least 3 dims, got {tuple(x.shape)}')
        batch_size, time_steps, num_bands = x.shape[:3]
        if time_steps < 2:
            raise ValueError('Dual-scale JEPA requires at least 2 time steps so the final step can be masked')

        beta_dist = torch.distributions.Beta(self.final_beta_alpha, self.final_beta_beta)
        sampled = beta_dist.sample((batch_size,)).to(device=x.device)
        per_sample_rates = self.final_min_mask_rate + sampled * (self.final_max_mask_rate - self.final_min_mask_rate)

        unknown = torch.zeros(batch_size, time_steps, num_bands, device=x.device, dtype=torch.bool)
        if time_steps > 1:
            random_values = torch.rand(batch_size, time_steps - 1, num_bands, device=x.device)
            unknown[:, :-1, :] = random_values < per_sample_rates[:, None, None]
        unknown[:, -1, :] = True

        for batch_idx in range(batch_size):
            if not (~unknown[batch_idx]).any():
                time_idx = int(torch.randint(0, time_steps - 1, (1,), device=x.device).item())
                band_idx = int(torch.randint(0, num_bands, (1,), device=x.device).item())
                unknown[batch_idx, time_idx, band_idx] = False

        retained = ~unknown
        split_random = torch.rand(batch_size, time_steps, num_bands, device=x.device)
        known_32x16 = retained & (split_random < self.known_split_prob)
        known_128x64 = retained & ~known_32x16
        return DualScaleMaskSplit(known_32x16=known_32x16, known_128x64=known_128x64, unknown=unknown)


@dataclass
class TimeBand128x64JEPAConfig:
    num_low_bands: int
    predictor_dim: int
    query_init_std: float = 0.02
    preprocessor_name_32x16: str = 'uniform_grid_32x16_quadrant'
    preprocessor_name_128x64: str = 'uniform_grid_128x64'
    preprocessor_kwargs_32x16: dict[str, Any] = field(default_factory=dict)
    preprocessor_kwargs_128x64: dict[str, Any] = field(default_factory=dict)
    tokenizer_name_32x16_3p5: str = '32x16_m'
    tokenizer_name_32x16_28: str = '32x16_m'
    tokenizer_name_128x64_3p5: str = '128x64_m'
    tokenizer_name_128x64_28: str = '128x64_m'
    num_tokens: int = 2
    token_dim: int = 192
    tokenizer_output_norm_32x16: str = 'batchnorm'
    tokenizer_output_norm_128x64: str = 'batchnorm'
    encoder: TimeBandViTConfig = field(default_factory=TimeBandViTConfig)
    predictor: TimeBandViTConfig = field(default_factory=lambda: TimeBandViTConfig(depth=2, num_heads=4))


@dataclass
class TimeBand128x64JEPAOutput:
    latent_visible: Tensor
    latent_masked: Tensor
    latent_predicted: Tensor
    latent_all: Tensor
    latent_visible_mask: Tensor
    latent_masked_mask: Tensor
    latent_all_mask: Tensor
    latent_masked_positions: Tensor
    latent_32x16_all: Tensor
    latent_128x64_all: Tensor


class TimeBand128x64JEPAPipeline(nn.Module):
    def __init__(self, mask_generator: DualScaleMaskGenerator, config: TimeBand128x64JEPAConfig) -> None:
        super().__init__()
        self.mask_generator = mask_generator
        self.config = config
        self.num_low_bands = int(config.num_low_bands)
        self.preprocessor_32x16 = build_preprocessor(config.preprocessor_name_32x16, config.preprocessor_kwargs_32x16)
        self.preprocessor_128x64 = build_preprocessor(config.preprocessor_name_128x64, config.preprocessor_kwargs_128x64)

        self.tokenizer_32x16_3p5 = build_tokenizer(config.tokenizer_name_32x16_3p5, config.num_tokens, config.token_dim, output_norm=config.tokenizer_output_norm_32x16)
        self.tokenizer_32x16_28 = build_tokenizer(config.tokenizer_name_32x16_28, config.num_tokens, config.token_dim, output_norm=config.tokenizer_output_norm_32x16)
        self.tokenizer_128x64_3p5 = build_tokenizer(config.tokenizer_name_128x64_3p5, config.num_tokens, config.token_dim, output_norm=config.tokenizer_output_norm_128x64)
        self.tokenizer_128x64_28 = build_tokenizer(config.tokenizer_name_128x64_28, config.num_tokens, config.token_dim, output_norm=config.tokenizer_output_norm_128x64)
        self._freeze_module(self.tokenizer_32x16_3p5)
        self._freeze_module(self.tokenizer_32x16_28)

        self.num_cell_tokens = int(config.num_tokens)
        self.embed_dim = int(config.token_dim)
        self.predictor_dim = int(config.predictor_dim)
        self.vit_encoder = TimeBandViT(
            embed_dim=self.embed_dim,
            num_cell_tokens=self.num_cell_tokens,
            depth=config.encoder.depth,
            num_heads=config.encoder.num_heads,
            mlp_ratio=config.encoder.mlp_ratio,
            dropout=config.encoder.dropout,
        )
        self.encoder_to_predictor = nn.Linear(self.embed_dim, self.predictor_dim)
        self.mask_query_tokens = nn.Parameter(torch.zeros(1, self.num_cell_tokens, self.predictor_dim))
        nn.init.normal_(self.mask_query_tokens, std=config.query_init_std)
        self.vit_predictor = TimeBandViT(
            embed_dim=self.predictor_dim,
            num_cell_tokens=self.num_cell_tokens,
            depth=config.predictor.depth,
            num_heads=config.predictor.num_heads,
            mlp_ratio=config.predictor.mlp_ratio,
            dropout=config.predictor.dropout,
        )
        self.predictor_to_latent = nn.Sequential(
            nn.Linear(self.predictor_dim, self.predictor_dim),
            nn.GELU(),
            nn.Linear(self.predictor_dim, self.embed_dim),
        )

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def freeze_32x16_branch(self) -> None:
        self._freeze_module(self.tokenizer_32x16_3p5)
        self._freeze_module(self.tokenizer_32x16_28)

    def load_pretrained_32x16_core(self, path: str | Path) -> None:
        path = Path(path)
        state_map = {
            'student_tokenizer_3p5.pt': self.tokenizer_32x16_3p5,
            'student_tokenizer_28.pt': self.tokenizer_32x16_28,
            'vit_encoder.pt': self.vit_encoder,
            'vit_predictor.pt': self.vit_predictor,
            'encoder_to_predictor.pt': self.encoder_to_predictor,
            'predictor_to_latent.pt': self.predictor_to_latent,
        }
        for name, module in state_map.items():
            module.load_state_dict(torch.load(path / name, map_location='cpu'))
        self.mask_query_tokens.data.copy_(torch.load(path / 'mask_query_tokens.pt', map_location='cpu').to(self.mask_query_tokens.device))
        self.freeze_32x16_branch()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_map = {
            'student_tokenizer_32x16_3p5.pt': self.tokenizer_32x16_3p5.state_dict(),
            'student_tokenizer_32x16_28.pt': self.tokenizer_32x16_28.state_dict(),
            'student_tokenizer_128x64_3p5.pt': self.tokenizer_128x64_3p5.state_dict(),
            'student_tokenizer_128x64_28.pt': self.tokenizer_128x64_28.state_dict(),
            'vit_encoder.pt': self.vit_encoder.state_dict(),
            'vit_predictor.pt': self.vit_predictor.state_dict(),
            'encoder_to_predictor.pt': self.encoder_to_predictor.state_dict(),
            'predictor_to_latent.pt': self.predictor_to_latent.state_dict(),
            'mask_query_tokens.pt': self.mask_query_tokens.detach().cpu(),
        }
        for name, state in state_map.items():
            torch.save(state, path / name)

    def _build_cell_positions(self, time_steps: int, num_bands: int, device: torch.device) -> Tensor:
        time_ids = torch.arange(time_steps, device=device, dtype=torch.long).view(time_steps, 1)
        band_ids = torch.arange(num_bands, device=device, dtype=torch.long).view(1, num_bands)
        return torch.stack([time_ids.expand(time_steps, num_bands), band_ids.expand(time_steps, num_bands)], dim=-1)

    def _build_band_masks(self, num_bands: int, flat_band_ids: Tensor) -> tuple[Tensor, Tensor]:
        low_mask = flat_band_ids < self.num_low_bands
        return low_mask, ~low_mask

    def _encode_group(self, group_x: Tensor, *, preprocessor: nn.Module, tokenizer: nn.Module) -> Tensor:
        batch_size, num_cells = group_x.shape[:2]
        if num_cells == 0:
            return group_x.new_zeros((batch_size, 0, self.num_cell_tokens, self.embed_dim))
        flat_group = group_x.reshape(batch_size * num_cells, *group_x.shape[2:])
        flat_preprocessed = preprocessor(flat_group)
        flat_tokens = tokenizer(flat_preprocessed)
        return flat_tokens.view(batch_size, num_cells, self.num_cell_tokens, self.embed_dim)

    def _merge_group_outputs(self, *, low_values: Tensor, high_values: Tensor, low_mask: Tensor, high_mask: Tensor) -> Tensor:
        num_cells = int(low_mask.numel())
        ref = low_values if (low_values.numel() > 0 or high_values.numel() == 0) else high_values
        merged = ref.new_zeros((ref.shape[0], num_cells, *ref.shape[2:]))
        if low_mask.any():
            merged[:, low_mask] = low_values
        if high_mask.any():
            merged[:, high_mask] = high_values
        return merged

    def _encode_all_branch(self, flat_x: Tensor, *, low_mask: Tensor, high_mask: Tensor, preprocessor: nn.Module, tokenizer_low: nn.Module, tokenizer_high: nn.Module) -> Tensor:
        low_x = flat_x[:, low_mask]
        high_x = flat_x[:, high_mask]
        low_latent = self._encode_group(low_x, preprocessor=preprocessor, tokenizer=tokenizer_low)
        high_latent = self._encode_group(high_x, preprocessor=preprocessor, tokenizer=tokenizer_high)
        return self._merge_group_outputs(low_values=low_latent, high_values=high_latent, low_mask=low_mask, high_mask=high_mask)

    def _gather_cells(self, values: Tensor, mask: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_cells = mask.shape
        counts = mask.sum(dim=1)
        max_cells = int(counts.max().item()) if counts.numel() > 0 else 0
        gathered_values = values.new_zeros((batch_size, max_cells, *values.shape[2:]))
        gathered_positions = positions.new_zeros((batch_size, max_cells, 2))
        valid_mask = torch.zeros(batch_size, max_cells, device=mask.device, dtype=torch.bool)
        if max_cells == 0:
            return gathered_values, valid_mask, gathered_positions
        slot_ids = mask.long().cumsum(dim=1) - 1
        batch_ids, cell_ids = torch.nonzero(mask, as_tuple=True)
        target_slots = slot_ids[batch_ids, cell_ids]
        gathered_values[batch_ids, target_slots] = values[batch_ids, cell_ids]
        gathered_positions[batch_ids, target_slots] = positions[cell_ids]
        valid_mask[batch_ids, target_slots] = True
        return gathered_values, valid_mask, gathered_positions

    def _build_query_tokens(self, batch_size: int, num_masked_cells: int) -> Tensor:
        if num_masked_cells == 0:
            return self.mask_query_tokens.new_zeros(batch_size, 0, self.num_cell_tokens, self.predictor_dim)
        return self.mask_query_tokens.unsqueeze(1).expand(batch_size, num_masked_cells, -1, -1)

    def forward(self, x: Tensor, *, mask_split: DualScaleMaskSplit | None = None) -> TimeBand128x64JEPAOutput:
        if x.ndim != 6:
            raise ValueError(f'Expected input shape [B, T, N, 2, car, ant], got {tuple(x.shape)}')
        batch_size, max_time, num_bands = x.shape[:3]
        if mask_split is None:
            mask_split = self.mask_generator(x)
        known_32x16 = mask_split.known_32x16.to(device=x.device, dtype=torch.bool)
        known_128x64 = mask_split.known_128x64.to(device=x.device, dtype=torch.bool)
        unknown = mask_split.unknown.to(device=x.device, dtype=torch.bool)
        if not torch.equal(known_32x16 | known_128x64 | unknown, torch.ones_like(unknown, dtype=torch.bool)):
            raise ValueError('known_32x16, known_128x64, unknown must cover all cells')
        if (known_32x16 & known_128x64).any() or (known_32x16 & unknown).any() or (known_128x64 & unknown).any():
            raise ValueError('known_32x16, known_128x64, unknown must be non-overlapping')

        flat_x = x.view(batch_size, max_time * num_bands, *x.shape[3:])
        cell_positions = self._build_cell_positions(max_time, num_bands, x.device).view(max_time * num_bands, 2)
        flat_band_ids = cell_positions[:, 1]
        flat_low_band_mask, flat_high_band_mask = self._build_band_masks(num_bands, flat_band_ids)

        with torch.no_grad():
            latent_32x16_all = self._encode_all_branch(
                flat_x,
                low_mask=flat_low_band_mask,
                high_mask=flat_high_band_mask,
                preprocessor=self.preprocessor_32x16,
                tokenizer_low=self.tokenizer_32x16_3p5,
                tokenizer_high=self.tokenizer_32x16_28,
            )
        latent_128x64_all = self._encode_all_branch(
            flat_x,
            low_mask=flat_low_band_mask,
            high_mask=flat_high_band_mask,
            preprocessor=self.preprocessor_128x64,
            tokenizer_low=self.tokenizer_128x64_3p5,
            tokenizer_high=self.tokenizer_128x64_28,
        )

        flat_known_32x16 = known_32x16.view(batch_size, -1)
        flat_known_128x64 = known_128x64.view(batch_size, -1)
        flat_unknown = unknown.view(batch_size, -1)
        flat_visible = flat_known_32x16 | flat_known_128x64
        if not flat_visible.any(dim=1).all():
            raise ValueError('At least one visible cell is required for every sample')

        combined_all = torch.where(flat_known_32x16[:, :, None, None], latent_32x16_all, latent_128x64_all)
        combined_all = torch.where(flat_unknown[:, :, None, None], latent_32x16_all, combined_all)

        latent_visible, visible_valid_mask, visible_positions = self._gather_cells(combined_all, flat_visible, cell_positions)
        latent_masked, masked_valid_mask, masked_positions = self._gather_cells(latent_32x16_all, flat_unknown, cell_positions)
        latent_all_mask = torch.ones(batch_size, max_time * num_bands, device=x.device, dtype=torch.bool)

        encoded_visible = self.vit_encoder(latent_visible, positions_2d=visible_positions, valid_cell_mask=visible_valid_mask)
        predictor_tokens = self.encoder_to_predictor(encoded_visible)
        query_tokens = self._build_query_tokens(batch_size, latent_masked.shape[1])
        predictor_input = torch.cat([predictor_tokens, query_tokens], dim=1)
        predictor_positions = torch.cat([visible_positions, masked_positions], dim=1)
        predictor_valid_mask = torch.cat([visible_valid_mask, masked_valid_mask], dim=1)
        predictor_output = self.vit_predictor(predictor_input, positions_2d=predictor_positions, valid_cell_mask=predictor_valid_mask)
        masked_predictor_tokens = predictor_output[:, -latent_masked.shape[1]:] if latent_masked.shape[1] > 0 else predictor_output[:, 0:0]
        latent_predicted = self.predictor_to_latent(masked_predictor_tokens)

        return TimeBand128x64JEPAOutput(
            latent_visible=latent_visible,
            latent_masked=latent_masked,
            latent_predicted=latent_predicted,
            latent_all=combined_all,
            latent_visible_mask=visible_valid_mask,
            latent_masked_mask=masked_valid_mask,
            latent_all_mask=latent_all_mask,
            latent_masked_positions=masked_positions,
            latent_32x16_all=latent_32x16_all,
            latent_128x64_all=latent_128x64_all,
        )
