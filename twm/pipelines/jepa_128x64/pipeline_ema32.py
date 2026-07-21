from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .pipeline import DualScaleMaskGenerator, DualScaleMaskSplit, TimeBand128x64JEPAConfig
from twm.pipelines.jepa_32x16.utils import build_preprocessor, build_tokenizer
from twm.pipelines.jepa_32x16.vit import TimeBandViT


@dataclass
class TimeBand128x64EMA32JEPAOutput:
    latent_visible: Tensor
    latent_masked: Tensor
    latent_predicted: Tensor
    latent_all: Tensor
    latent_visible_mask: Tensor
    latent_masked_mask: Tensor
    latent_all_mask: Tensor
    latent_masked_positions: Tensor
    latent_32x16_student_all: Tensor
    latent_32x16_teacher_all: Tensor
    latent_128x64_all: Tensor


class TimeBand128x64EMA32JEPAPipeline(nn.Module):
    def __init__(self, mask_generator: DualScaleMaskGenerator, config: TimeBand128x64JEPAConfig) -> None:
        super().__init__()
        self.mask_generator = mask_generator
        self.config = config
        self.num_low_bands = int(config.num_low_bands)
        self.preprocessor_32x16 = build_preprocessor(config.preprocessor_name_32x16, config.preprocessor_kwargs_32x16)
        self.preprocessor_128x64 = build_preprocessor(config.preprocessor_name_128x64, config.preprocessor_kwargs_128x64)

        self.tokenizer_32x16_3p5 = build_tokenizer(
            config.tokenizer_name_32x16_3p5,
            config.num_tokens,
            config.token_dim,
            output_norm=config.tokenizer_output_norm_32x16,
        )
        self.tokenizer_32x16_28 = build_tokenizer(
            config.tokenizer_name_32x16_28,
            config.num_tokens,
            config.token_dim,
            output_norm=config.tokenizer_output_norm_32x16,
        )
        self.teacher_tokenizer_32x16_3p5 = copy.deepcopy(self.tokenizer_32x16_3p5)
        self.teacher_tokenizer_32x16_28 = copy.deepcopy(self.tokenizer_32x16_28)
        self._freeze_module(self.teacher_tokenizer_32x16_3p5)
        self._freeze_module(self.teacher_tokenizer_32x16_28)

        self.tokenizer_128x64_3p5 = build_tokenizer(
            config.tokenizer_name_128x64_3p5,
            config.num_tokens,
            config.token_dim,
            output_norm=config.tokenizer_output_norm_128x64,
        )
        self.tokenizer_128x64_28 = build_tokenizer(
            config.tokenizer_name_128x64_28,
            config.num_tokens,
            config.token_dim,
            output_norm=config.tokenizer_output_norm_128x64,
        )

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

    def _refresh_teacher_32x16(self) -> None:
        self.teacher_tokenizer_32x16_3p5.load_state_dict(self.tokenizer_32x16_3p5.state_dict())
        self.teacher_tokenizer_32x16_28.load_state_dict(self.tokenizer_32x16_28.state_dict())
        self._freeze_module(self.teacher_tokenizer_32x16_3p5)
        self._freeze_module(self.teacher_tokenizer_32x16_28)

    def synchronize_teacher(self, momentum: float = 0.95) -> None:
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f'momentum must be in [0, 1), got {momentum}')
        with torch.no_grad():
            for student_module, teacher_module in [
                (self.tokenizer_32x16_3p5, self.teacher_tokenizer_32x16_3p5),
                (self.tokenizer_32x16_28, self.teacher_tokenizer_32x16_28),
            ]:
                for teacher_param, student_param in zip(teacher_module.parameters(), student_module.parameters()):
                    teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
                for teacher_buf, student_buf in zip(teacher_module.buffers(), student_module.buffers()):
                    teacher_buf.copy_(student_buf)

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
        self._refresh_teacher_32x16()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_map = {
            'student_tokenizer_32x16_3p5.pt': self.tokenizer_32x16_3p5.state_dict(),
            'student_tokenizer_32x16_28.pt': self.tokenizer_32x16_28.state_dict(),
            'teacher_tokenizer_32x16_3p5.pt': self.teacher_tokenizer_32x16_3p5.state_dict(),
            'teacher_tokenizer_32x16_28.pt': self.teacher_tokenizer_32x16_28.state_dict(),
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

    def _encode_all_branch(
        self,
        flat_x: Tensor,
        *,
        low_mask: Tensor,
        high_mask: Tensor,
        preprocessor: nn.Module,
        tokenizer_low: nn.Module,
        tokenizer_high: nn.Module,
    ) -> Tensor:
        low_x = flat_x[:, low_mask]
        high_x = flat_x[:, high_mask]
        low_latent = self._encode_group(low_x, preprocessor=preprocessor, tokenizer=tokenizer_low)
        high_latent = self._encode_group(high_x, preprocessor=preprocessor, tokenizer=tokenizer_high)
        return self._merge_group_outputs(low_values=low_latent, high_values=high_latent, low_mask=low_mask, high_mask=high_mask)

    def _gather_cells(self, values: Tensor, mask: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, _ = mask.shape
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

    def forward(self, x: Tensor, *, mask_split: DualScaleMaskSplit | None = None) -> TimeBand128x64EMA32JEPAOutput:
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

        latent_32x16_student_all = self._encode_all_branch(
            flat_x,
            low_mask=flat_low_band_mask,
            high_mask=flat_high_band_mask,
            preprocessor=self.preprocessor_32x16,
            tokenizer_low=self.tokenizer_32x16_3p5,
            tokenizer_high=self.tokenizer_32x16_28,
        )
        with torch.no_grad():
            latent_32x16_teacher_all = self._encode_all_branch(
                flat_x,
                low_mask=flat_low_band_mask,
                high_mask=flat_high_band_mask,
                preprocessor=self.preprocessor_32x16,
                tokenizer_low=self.teacher_tokenizer_32x16_3p5,
                tokenizer_high=self.teacher_tokenizer_32x16_28,
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

        combined_all = torch.where(flat_known_32x16[:, :, None, None], latent_32x16_student_all, latent_128x64_all)
        combined_all = torch.where(flat_unknown[:, :, None, None], latent_32x16_teacher_all, combined_all)

        latent_visible, visible_valid_mask, visible_positions = self._gather_cells(combined_all, flat_visible, cell_positions)
        latent_masked, masked_valid_mask, masked_positions = self._gather_cells(latent_32x16_teacher_all, flat_unknown, cell_positions)
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

        return TimeBand128x64EMA32JEPAOutput(
            latent_visible=latent_visible,
            latent_masked=latent_masked,
            latent_predicted=latent_predicted,
            latent_all=combined_all,
            latent_visible_mask=visible_valid_mask,
            latent_masked_mask=masked_valid_mask,
            latent_all_mask=latent_all_mask,
            latent_masked_positions=masked_positions,
            latent_32x16_student_all=latent_32x16_student_all,
            latent_32x16_teacher_all=latent_32x16_teacher_all,
            latent_128x64_all=latent_128x64_all,
        )
