from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .time_freq_band import AbstractMaskGenerator
from .utils import build_preprocessor, build_tokenizer
from .vit import TimeBandViT


@dataclass
class TimeBandViTConfig:
    embed_dim: int | None = None
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class TimeBandJEPAConfig:
    num_low_bands: int
    predictor_dim: int | None = None
    query_init_std: float = 0.02
    preprocessor_name: str = 'identity'
    preprocessor_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_name_3p5: str = '32x16_m'
    tokenizer_name_28: str = '32x16_m'
    num_tokens: int = 8
    token_dim: int = 128
    tokenizer_output_norm: str = 'batchnorm'
    enable_recovery_head: bool = True
    enable_cond_diffusion_head: bool = True
    masked_target_source: str = 'teacher_tokenizer'
    encoder: TimeBandViTConfig = field(default_factory=TimeBandViTConfig)
    predictor: TimeBandViTConfig = field(default_factory=lambda: TimeBandViTConfig(depth=4, num_heads=8))


@dataclass
class TimeBandLatentBatch:
    latents: Tensor
    valid_mask: Tensor
    positions_2d: Tensor | None = None
    preprocessed: Tensor | None = None


@dataclass
class MaskedBandAuxiliaryBatch:
    tokens: Tensor
    target: Tensor
    valid_mask: Tensor | None
    recovered: Tensor | None = None
    diffusion_predicted_noise: Tensor | None = None
    diffusion_target_noise: Tensor | None = None


@dataclass
class TimeBandJEPAOutput:
    latent_remained: Tensor
    latent_masked: Tensor
    latent_predicted: Tensor
    latent_all: Tensor | None = None
    latent_remained_mask: Tensor | None = None
    latent_masked_mask: Tensor | None = None
    latent_all_mask: Tensor | None = None
    latent_masked_positions: Tensor | None = None
    masked_aux_3p5: MaskedBandAuxiliaryBatch | None = None
    masked_aux_28: MaskedBandAuxiliaryBatch | None = None


class TimeBandJEPAPipeline(nn.Module):
    def __init__(
        self,
        mask_generator: AbstractMaskGenerator,
        config: TimeBandJEPAConfig,
    ) -> None:
        super().__init__()
        self.mask_generator = mask_generator
        self.config = config
        self.num_low_bands = int(config.num_low_bands)

        self._validate_32x16_config(config)
        self.preprocessor = build_preprocessor(config.preprocessor_name, config.preprocessor_kwargs)
        self.tokenizer_3p5 = build_tokenizer(
            tokenizer_name=config.tokenizer_name_3p5,
            num_tokens=config.num_tokens,
            token_dim=config.token_dim,
            output_norm=config.tokenizer_output_norm,
        )
        self.tokenizer_28 = build_tokenizer(
            tokenizer_name=config.tokenizer_name_28,
            num_tokens=config.num_tokens,
            token_dim=config.token_dim,
            output_norm=config.tokenizer_output_norm,
        )
        self.teacher_tokenizer_3p5 = copy.deepcopy(self.tokenizer_3p5)
        self.teacher_tokenizer_28 = copy.deepcopy(self.tokenizer_28)
        self._freeze_module(self.teacher_tokenizer_3p5)
        self._freeze_module(self.teacher_tokenizer_28)

        inferred_preprocessed_shape, inferred_num_tokens, inferred_embed_dim = self._load_or_infer_token_shape()
        self.preprocessed_shape = inferred_preprocessed_shape
        self.num_cell_tokens = inferred_num_tokens
        self.embed_dim = int(config.token_dim)
        if self.embed_dim != inferred_embed_dim:
            raise ValueError(f'token_dim must match tokenizer output dim {inferred_embed_dim}, got {self.embed_dim}')
        if config.encoder.embed_dim is not None and int(config.encoder.embed_dim) != self.embed_dim:
            raise ValueError(
                f'encoder.embed_dim must follow token_dim={self.embed_dim}, got {int(config.encoder.embed_dim)}'
            )

        predictor_dim = config.predictor_dim if config.predictor_dim is not None else self.embed_dim
        self.predictor_dim = int(predictor_dim)
        if config.predictor.embed_dim is not None and int(config.predictor.embed_dim) != self.predictor_dim:
            raise ValueError(
                f'predictor.embed_dim must follow predictor_dim={self.predictor_dim}, got {int(config.predictor.embed_dim)}'
            )

        self.vit_encoder = TimeBandViT(
            embed_dim=self.embed_dim,
            num_cell_tokens=self.num_cell_tokens,
            depth=config.encoder.depth,
            num_heads=config.encoder.num_heads,
            mlp_ratio=config.encoder.mlp_ratio,
            dropout=config.encoder.dropout,
        )
        self.teacher_vit_encoder = copy.deepcopy(self.vit_encoder)
        self._freeze_module(self.teacher_vit_encoder)
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
        self.recovery_head_3p5: nn.Module | None = None
        self.recovery_head_28: nn.Module | None = None
        if config.enable_recovery_head:
            from .recovery.models import SymmetricRecoveryDecoder32x16Large
            self.recovery_head_3p5 = SymmetricRecoveryDecoder32x16Large(
                num_tokens=self.num_cell_tokens,
                token_dim=self.embed_dim,
            )
            self.recovery_head_28 = SymmetricRecoveryDecoder32x16Large(
                num_tokens=self.num_cell_tokens,
                token_dim=self.embed_dim,
            )
        self.diffusion_model_3p5: DiffusersConditionalRecoveryModel32x16 | None = None
        self.diffusion_model_28: DiffusersConditionalRecoveryModel32x16 | None = None
        if config.enable_cond_diffusion_head:
            from .recovery_diffusion.models import DiffusersConditionalRecoveryModel32x16
            self.diffusion_model_3p5 = DiffusersConditionalRecoveryModel32x16(token_dim=self.embed_dim)
            self.diffusion_model_28 = DiffusersConditionalRecoveryModel32x16(token_dim=self.embed_dim)

    @classmethod
    def from_config(
        cls,
        mask_generator: AbstractMaskGenerator,
        config: TimeBandJEPAConfig,
    ) -> TimeBandJEPAPipeline:
        return cls(mask_generator=mask_generator, config=config)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _validate_32x16_config(config: TimeBandJEPAConfig) -> None:
        preprocessor_name = str(config.preprocessor_name).strip().lower()
        if preprocessor_name != 'uniform_grid_32x16_quadrant':
            raise ValueError(
                'TimeBandJEPAPipeline is fixed to the 32x16 path and only supports '
                "preprocessor_name='uniform_grid_32x16_quadrant'"
            )
        for attr_name in ['tokenizer_name_3p5', 'tokenizer_name_28']:
            tokenizer_name = str(getattr(config, attr_name)).strip().lower()
            if not tokenizer_name.startswith('32x16_'):
                raise ValueError(
                    'TimeBandJEPAPipeline is fixed to the 32x16 path and only supports '
                    f'32x16 tokenizers, got {attr_name}={tokenizer_name!r}'
                )

    def _cache_root(self) -> Path:
        return Path(__file__).resolve().parents[3] / '.cache' / 'jepa_token_shape'

    def _token_shape_cache_payload(self) -> dict[str, Any]:
        return {
            'preprocessor_name': str(self.config.preprocessor_name),
            'preprocessor_kwargs': dict(self.config.preprocessor_kwargs),
            'tokenizer_name_3p5': str(self.config.tokenizer_name_3p5),
            'tokenizer_name_28': str(self.config.tokenizer_name_28),
            'num_tokens': int(self.config.num_tokens),
            'token_dim': int(self.config.token_dim),
            'num_low_bands': int(self.config.num_low_bands),
        }

    def _token_shape_cache_path(self) -> Path:
        payload = json.dumps(self._token_shape_cache_payload(), sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha1(payload.encode('utf-8')).hexdigest()
        return self._cache_root() / f'{digest}.json'

    def _read_token_shape_cache(self) -> tuple[tuple[int, ...], int, int] | None:
        path = self._token_shape_cache_path()
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
        preprocessed_shape = tuple(int(dim) for dim in data['preprocessed_shape'])
        num_cell_tokens = int(data['num_cell_tokens'])
        embed_dim = int(data['embed_dim'])
        return preprocessed_shape, num_cell_tokens, embed_dim

    def _write_token_shape_cache(self, preprocessed_shape: tuple[int, ...], num_cell_tokens: int, embed_dim: int) -> None:
        path = self._token_shape_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'config': self._token_shape_cache_payload(),
            'preprocessed_shape': list(preprocessed_shape),
            'num_cell_tokens': int(num_cell_tokens),
            'embed_dim': int(embed_dim),
        }
        tmp_path = path.with_suffix('.tmp')
        tmp_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        tmp_path.replace(path)

    def _load_or_infer_token_shape(self) -> tuple[tuple[int, ...], int, int]:
        cached = self._read_token_shape_cache()
        if cached is not None:
            return cached
        inferred = self._infer_token_shape()
        self._write_token_shape_cache(*inferred)
        return inferred

    def _infer_token_shape(self) -> tuple[tuple[int, ...], int, int]:
        preprocessor_training = self.preprocessor.training
        tokenizer_training = self.tokenizer_3p5.training
        try:
            self.preprocessor.eval()
            self.tokenizer_3p5.eval()
            with torch.no_grad():
                probe = self._make_inference_probe()
                preprocessed = self.preprocessor(probe)
                probe_out = self.tokenizer_3p5(preprocessed)
        finally:
            self.preprocessor.train(preprocessor_training)
            self.tokenizer_3p5.train(tokenizer_training)
        if probe_out.ndim != 3:
            raise ValueError(f'Tokenizer must output [B, Nt, Lt], got shape {tuple(probe_out.shape)}')
        preprocessed_shape = tuple(int(dim) for dim in preprocessed.shape[1:])
        return preprocessed_shape, int(probe_out.shape[1]), int(probe_out.shape[2])

    def _make_inference_probe(self) -> Tensor:
        return torch.zeros(1, 2, 1024, 64)

    def synchronize_teacher(self, momentum: float = 0.99) -> None:
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f'momentum must be in [0, 1), got {momentum}')
        with torch.no_grad():
            for student_module, teacher_module in [
                (self.tokenizer_3p5, self.teacher_tokenizer_3p5),
                (self.tokenizer_28, self.teacher_tokenizer_28),
                (self.vit_encoder, self.teacher_vit_encoder),
            ]:
                for teacher_param, student_param in zip(teacher_module.parameters(), student_module.parameters()):
                    teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
                for teacher_buf, student_buf in zip(teacher_module.buffers(), student_module.buffers()):
                    teacher_buf.copy_(student_buf)
    def _build_cell_positions(self, time_steps: int, num_bands: int, device: torch.device) -> Tensor:
        time_ids = torch.arange(time_steps, device=device, dtype=torch.long).view(time_steps, 1)
        band_ids = torch.arange(num_bands, device=device, dtype=torch.long).view(1, num_bands)
        return torch.stack(
            [
                time_ids.expand(time_steps, num_bands),
                band_ids.expand(time_steps, num_bands),
            ],
            dim=-1,
        )

    def _build_band_masks(self, num_bands: int, flat_band_ids: Tensor) -> tuple[Tensor, Tensor]:
        if num_bands == 1:
            low_mask = torch.ones_like(flat_band_ids, dtype=torch.bool)
            high_mask = torch.zeros_like(flat_band_ids, dtype=torch.bool)
            return low_mask, high_mask
        if self.num_low_bands <= 0 or self.num_low_bands >= num_bands:
            raise ValueError(f'num_low_bands must be in (0, N), got {self.num_low_bands} for N={num_bands}')
        low_mask = flat_band_ids < self.num_low_bands
        return low_mask, ~low_mask

    def _encode_group(
        self,
        group_x: Tensor,
        *,
        tokenizer: nn.Module,
        latent_dim: int,
        return_preprocessed: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch_size, num_cells = group_x.shape[:2]
        if num_cells == 0:
            tokens = torch.zeros(
                batch_size,
                0,
                self.num_cell_tokens,
                latent_dim,
                device=group_x.device,
                dtype=group_x.dtype,
            )
            if not return_preprocessed:
                return tokens
            preprocessed = torch.zeros(
                batch_size,
                0,
                *self.preprocessed_shape,
                device=group_x.device,
                dtype=group_x.dtype,
            )
            return tokens, preprocessed
        flat_group = group_x.reshape(batch_size * num_cells, *group_x.shape[2:])
        flat_preprocessed = self.preprocessor(flat_group)
        flat_tokens = tokenizer(flat_preprocessed)
        tokens = flat_tokens.view(batch_size, num_cells, self.num_cell_tokens, latent_dim)
        if not return_preprocessed:
            return tokens
        preprocessed = flat_preprocessed.view(batch_size, num_cells, *flat_preprocessed.shape[1:])
        return tokens, preprocessed

    def _merge_group_outputs(
        self,
        *,
        low_values: Tensor,
        high_values: Tensor,
        low_mask: Tensor,
        high_mask: Tensor,
    ) -> Tensor:
        num_cells = int(low_mask.numel())
        ref = low_values if (low_values.numel() > 0 or high_values.numel() == 0) else high_values
        merged = ref.new_zeros((ref.shape[0], num_cells, *ref.shape[2:]))
        if low_mask.any():
            merged[:, low_mask] = low_values
        if high_mask.any():
            merged[:, high_mask] = high_values
        return merged

    def _build_query_tokens(self, batch_size: int, num_masked_cells: int) -> Tensor:
        if num_masked_cells == 0:
            return self.mask_query_tokens.new_zeros(batch_size, 0, self.num_cell_tokens, self.predictor_dim)
        return self.mask_query_tokens.unsqueeze(1).expand(batch_size, num_masked_cells, -1, -1)

    def _gather_masked_cells(self, values: Tensor, mask: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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

    def _pad_group_list(
        self,
        value_list: list[Tensor],
        *,
        positions_list: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        batch_size = len(value_list)
        max_cells = max((value.shape[1] for value in value_list), default=0)
        device = self.mask_query_tokens.device
        if value_list:
            tail_shape = value_list[0].shape[2:]
            dtype = value_list[0].dtype
        else:
            tail_shape = ()
            dtype = self.mask_query_tokens.dtype
        padded = torch.zeros((batch_size, max_cells, *tail_shape), device=device, dtype=dtype)
        valid_mask = torch.zeros(batch_size, max_cells, device=device, dtype=torch.bool)
        padded_positions = None
        if positions_list is not None:
            padded_positions = torch.zeros(batch_size, max_cells, 2, device=device, dtype=torch.long)
        for batch_index, value in enumerate(value_list):
            num_cells = value.shape[1]
            if num_cells == 0:
                continue
            padded[batch_index, :num_cells] = value[0]
            valid_mask[batch_index, :num_cells] = True
            if padded_positions is not None:
                padded_positions[batch_index, :num_cells] = positions_list[batch_index]
        return padded, valid_mask, padded_positions

    def encode_visible_cells(self, x: Tensor) -> TimeBandLatentBatch:
        if x.ndim == 5:
            x = x.unsqueeze(1)
        elif x.ndim != 6:
            raise ValueError(f'Expected input shape [B, T, N, 2, car, ant] or [B, N, 2, car, ant], got {tuple(x.shape)}')

        batch_size, max_time, num_bands = x.shape[:3]

        flat_x = x.view(batch_size, max_time * num_bands, *x.shape[3:])
        cell_positions = self._build_cell_positions(max_time, num_bands, x.device).view(max_time * num_bands, 2)
        flat_band_ids = cell_positions[:, 1]
        flat_low_band_mask, flat_high_band_mask = self._build_band_masks(num_bands, flat_band_ids)

        low_x, low_valid_mask, low_positions = self._gather_masked_cells(
            flat_x,
            flat_low_band_mask.unsqueeze(0).expand(batch_size, -1),
            cell_positions,
        )
        high_x, high_valid_mask, high_positions = self._gather_masked_cells(
            flat_x,
            flat_high_band_mask.unsqueeze(0).expand(batch_size, -1),
            cell_positions,
        )
        low_latent, low_preprocessed = self._encode_group(
            low_x,
            tokenizer=self.tokenizer_3p5,
            latent_dim=self.embed_dim,
            return_preprocessed=True,
        )
        high_latent, high_preprocessed = self._encode_group(
            high_x,
            tokenizer=self.tokenizer_28,
            latent_dim=self.embed_dim,
            return_preprocessed=True,
        )
        latent_visible = torch.cat([low_latent, high_latent], dim=1)
        preprocessed_visible = torch.cat([low_preprocessed, high_preprocessed], dim=1)
        visible_positions = torch.cat([low_positions, high_positions], dim=1)
        visible_valid_mask = torch.cat([low_valid_mask, high_valid_mask], dim=1)
        encoded_visible = self.vit_encoder(
            latent_visible,
            positions_2d=visible_positions,
            valid_cell_mask=visible_valid_mask,
        )
        return TimeBandLatentBatch(
            latents=encoded_visible,
            valid_mask=visible_valid_mask,
            positions_2d=visible_positions,
            preprocessed=preprocessed_visible,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_map = {
            'student_tokenizer_3p5.pt': self.tokenizer_3p5.state_dict(),
            'student_tokenizer_28.pt': self.tokenizer_28.state_dict(),
            'vit_encoder.pt': self.vit_encoder.state_dict(),
            'vit_predictor.pt': self.vit_predictor.state_dict(),
            'encoder_to_predictor.pt': self.encoder_to_predictor.state_dict(),
            'predictor_to_latent.pt': self.predictor_to_latent.state_dict(),
            'mask_query_tokens.pt': self.mask_query_tokens.detach().cpu(),
        }
        if self.recovery_head_3p5 is not None:
            state_map['recovery_head_3p5.pt'] = self.recovery_head_3p5.state_dict()
        if self.recovery_head_28 is not None:
            state_map['recovery_head_28.pt'] = self.recovery_head_28.state_dict()
        if self.diffusion_model_3p5 is not None:
            state_map['diffusion_model_3p5.pt'] = self.diffusion_model_3p5.state_dict()
        if self.diffusion_model_28 is not None:
            state_map['diffusion_model_28.pt'] = self.diffusion_model_28.state_dict()
        for name, state_dict in state_map.items():
            torch.save(state_dict, path / name)

    def _refresh_teacher_tokenizers(self) -> None:
        self.teacher_tokenizer_3p5.load_state_dict(self.tokenizer_3p5.state_dict())
        self.teacher_tokenizer_28.load_state_dict(self.tokenizer_28.state_dict())
        self.teacher_vit_encoder.load_state_dict(self.vit_encoder.state_dict())
        self._freeze_module(self.teacher_tokenizer_3p5)
        self._freeze_module(self.teacher_tokenizer_28)
        self._freeze_module(self.teacher_vit_encoder)

    def freeze_tokenizers(self) -> None:
        self._freeze_module(self.tokenizer_3p5)
        self._freeze_module(self.tokenizer_28)
        self._freeze_module(self.teacher_tokenizer_3p5)
        self._freeze_module(self.teacher_tokenizer_28)

    def load_predictor_core(self, path: str | Path) -> None:
        path = Path(path)
        module_map = {
            'student_tokenizer_3p5.pt': self.tokenizer_3p5,
            'student_tokenizer_28.pt': self.tokenizer_28,
            'vit_encoder.pt': self.vit_encoder,
            'vit_predictor.pt': self.vit_predictor,
            'encoder_to_predictor.pt': self.encoder_to_predictor,
            'predictor_to_latent.pt': self.predictor_to_latent,
        }
        for name, module in module_map.items():
            file_path = path / name
            if not file_path.exists():
                raise FileNotFoundError(f'Missing required checkpoint file: {file_path}')
            state_dict = torch.load(file_path, map_location='cpu')
            if name.startswith('student_tokenizer_') and getattr(module, 'output_norm', None) == 'batchnorm':
                incompatible = module.load_state_dict(state_dict, strict=False)
                allowed_missing = {
                    'output_batch_norm.running_mean',
                    'output_batch_norm.running_var',
                }
                missing = set(incompatible.missing_keys)
                unexpected = set(incompatible.unexpected_keys)
                if unexpected or not missing.issubset(allowed_missing):
                    raise RuntimeError(
                        f'Unexpected tokenizer checkpoint mismatch for {file_path}: '
                        f'missing={sorted(missing)}, unexpected={sorted(unexpected)}'
                    )
                continue
            module.load_state_dict(state_dict)

        query_path = path / 'mask_query_tokens.pt'
        if not query_path.exists():
            raise FileNotFoundError(f'Missing required checkpoint file: {query_path}')
        self.mask_query_tokens.data.copy_(torch.load(query_path, map_location='cpu').to(self.mask_query_tokens.device))
        self._refresh_teacher_tokenizers()

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.load_predictor_core(path)
        module_map = {}
        if self.recovery_head_3p5 is not None:
            module_map['recovery_head_3p5.pt'] = self.recovery_head_3p5
        if self.recovery_head_28 is not None:
            module_map['recovery_head_28.pt'] = self.recovery_head_28
        if self.diffusion_model_3p5 is not None:
            module_map['diffusion_model_3p5.pt'] = self.diffusion_model_3p5
        if self.diffusion_model_28 is not None:
            module_map['diffusion_model_28.pt'] = self.diffusion_model_28
        for name, module in module_map.items():
            file_path = path / name
            if file_path.exists():
                module.load_state_dict(torch.load(file_path, map_location='cpu'))

    def forward(self, x: Tensor, *, mask: Tensor | None = None) -> TimeBandJEPAOutput:
        if x.ndim != 6:
            raise ValueError(f'Expected input shape [B, T, N, 2, car, ant], got {tuple(x.shape)}')

        batch_size, max_time, num_bands = x.shape[:3]
        if self.num_low_bands <= 0 or self.num_low_bands >= num_bands:
            raise ValueError(f'num_low_bands must be in (0, N), got {self.num_low_bands} for N={num_bands}')

        if mask is None:
            mask = self.mask_generator(x)
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)
        if tuple(mask.shape) != (batch_size, max_time, num_bands):
            raise ValueError(f'Mask must have shape [B, T, N], got {tuple(mask.shape)}')
        remained_mask = ~mask

        flat_x = x.view(batch_size, max_time * num_bands, *x.shape[3:])
        cell_positions = self._build_cell_positions(max_time, num_bands, x.device).view(max_time * num_bands, 2)
        flat_band_ids = cell_positions[:, 1]
        flat_low_band_mask, flat_high_band_mask = self._build_band_masks(num_bands, flat_band_ids)

        flat_remained_mask = remained_mask.reshape(batch_size, max_time * num_bands)
        flat_masked_mask = mask.reshape(batch_size, max_time * num_bands)
        if not flat_remained_mask.any(dim=1).all():
            raise ValueError('At least one remained cell is required for every sample.')

        remained_low_x, remained_low_valid, remained_low_positions = self._gather_masked_cells(
            flat_x,
            flat_remained_mask & flat_low_band_mask.unsqueeze(0),
            cell_positions,
        )
        remained_high_x, remained_high_valid, remained_high_positions = self._gather_masked_cells(
            flat_x,
            flat_remained_mask & flat_high_band_mask.unsqueeze(0),
            cell_positions,
        )
        masked_low_x, masked_low_valid, masked_low_positions = self._gather_masked_cells(
            flat_x,
            flat_masked_mask & flat_low_band_mask.unsqueeze(0),
            cell_positions,
        )
        masked_high_x, masked_high_valid, masked_high_positions = self._gather_masked_cells(
            flat_x,
            flat_masked_mask & flat_high_band_mask.unsqueeze(0),
            cell_positions,
        )

        low_remained_tokens, low_remained_target = self._encode_group(
            remained_low_x,
            tokenizer=self.tokenizer_3p5,
            latent_dim=self.embed_dim,
            return_preprocessed=True,
        )
        high_remained_tokens, high_remained_target = self._encode_group(
            remained_high_x,
            tokenizer=self.tokenizer_28,
            latent_dim=self.embed_dim,
            return_preprocessed=True,
        )
        low_remained_latent = low_remained_tokens
        high_remained_latent = high_remained_tokens
        low_masked_tokens, low_masked_target = self._encode_group(
            masked_low_x,
            tokenizer=self.tokenizer_3p5,
            latent_dim=self.embed_dim,
            return_preprocessed=True,
        )
        high_masked_tokens, high_masked_target = self._encode_group(
            masked_high_x,
            tokenizer=self.tokenizer_28,
            latent_dim=self.embed_dim,
            return_preprocessed=True,
        )

        masked_target_source = str(self.config.masked_target_source).strip().lower()
        with torch.no_grad():
            if masked_target_source == 'teacher_tokenizer' or masked_target_source == 'teacher_encoder':
                low_target_flat = self.teacher_tokenizer_3p5(low_masked_target.reshape(-1, *low_masked_target.shape[2:]))
                high_target_flat = self.teacher_tokenizer_28(high_masked_target.reshape(-1, *high_masked_target.shape[2:]))
            elif masked_target_source == 'student_tokenizer':
                low_target_flat = self.tokenizer_3p5(low_masked_target.reshape(-1, *low_masked_target.shape[2:]))
                high_target_flat = self.tokenizer_28(high_masked_target.reshape(-1, *high_masked_target.shape[2:]))
            else:
                raise ValueError(f'Unsupported masked_target_source: {self.config.masked_target_source!r}')
        low_masked_latent = low_target_flat.view(batch_size, low_masked_target.shape[1], self.num_cell_tokens, self.embed_dim)
        high_masked_latent = high_target_flat.view(batch_size, high_masked_target.shape[1], self.num_cell_tokens, self.embed_dim)

        low_recovered = None
        low_predicted_noise = None
        low_target_noise = None
        if self.recovery_head_3p5 is not None:
            low_recovered_flat = self.recovery_head_3p5(low_remained_tokens.reshape(-1, self.num_cell_tokens, self.embed_dim)).recovered
            low_recovered = low_recovered_flat.view(batch_size, low_remained_tokens.shape[1], *self.preprocessed_shape)
        if self.diffusion_model_3p5 is not None:
            low_flat_target = low_remained_target.reshape(-1, *self.preprocessed_shape)
            low_target_noise_flat = torch.randn_like(low_flat_target)
            low_timesteps = self.diffusion_model_3p5.sample_timesteps(low_flat_target.shape[0], low_flat_target.device)
            low_noisy_target = self.diffusion_model_3p5.add_noise(low_flat_target, low_target_noise_flat, low_timesteps)
            low_predicted_noise_flat = self.diffusion_model_3p5(
                low_noisy_target,
                low_remained_tokens.reshape(-1, self.num_cell_tokens, self.embed_dim),
                low_timesteps,
            ).predicted_noise
            low_predicted_noise = low_predicted_noise_flat.view(batch_size, low_remained_tokens.shape[1], *self.preprocessed_shape)
            low_target_noise = low_target_noise_flat.view(batch_size, low_remained_tokens.shape[1], *self.preprocessed_shape)

        high_recovered = None
        high_predicted_noise = None
        high_target_noise = None
        if self.recovery_head_28 is not None:
            high_recovered_flat = self.recovery_head_28(high_remained_tokens.reshape(-1, self.num_cell_tokens, self.embed_dim)).recovered
            high_recovered = high_recovered_flat.view(batch_size, high_remained_tokens.shape[1], *self.preprocessed_shape)
        if self.diffusion_model_28 is not None:
            high_flat_target = high_remained_target.reshape(-1, *self.preprocessed_shape)
            high_target_noise_flat = torch.randn_like(high_flat_target)
            high_timesteps = self.diffusion_model_28.sample_timesteps(high_flat_target.shape[0], high_flat_target.device)
            high_noisy_target = self.diffusion_model_28.add_noise(high_flat_target, high_target_noise_flat, high_timesteps)
            high_predicted_noise_flat = self.diffusion_model_28(
                high_noisy_target,
                high_remained_tokens.reshape(-1, self.num_cell_tokens, self.embed_dim),
                high_timesteps,
            ).predicted_noise
            high_predicted_noise = high_predicted_noise_flat.view(batch_size, high_remained_tokens.shape[1], *self.preprocessed_shape)
            high_target_noise = high_target_noise_flat.view(batch_size, high_remained_tokens.shape[1], *self.preprocessed_shape)

        latent_remained = torch.cat([low_remained_latent, high_remained_latent], dim=1)
        remained_positions = torch.cat([remained_low_positions, remained_high_positions], dim=1)
        remained_valid_mask = torch.cat([remained_low_valid, remained_high_valid], dim=1)
        latent_masked = torch.cat([low_masked_latent, high_masked_latent], dim=1)
        masked_positions = torch.cat([masked_low_positions, masked_high_positions], dim=1)
        masked_valid_mask = torch.cat([masked_low_valid, masked_high_valid], dim=1)
        if masked_target_source == 'teacher_encoder':
            latent_masked = self.teacher_vit_encoder(
                latent_masked,
                positions_2d=masked_positions,
                valid_cell_mask=masked_valid_mask,
            )

        all_latent_unsorted = torch.cat([latent_remained, latent_masked], dim=1)
        all_positions_unsorted = torch.cat([remained_positions, masked_positions], dim=1)
        all_valid_mask_unsorted = torch.cat([remained_valid_mask, masked_valid_mask], dim=1)
        all_flat_indices = all_positions_unsorted[..., 0] * num_bands + all_positions_unsorted[..., 1]
        invalid_fill = torch.full_like(all_flat_indices, fill_value=max_time * num_bands)
        all_sort_keys = torch.where(all_valid_mask_unsorted, all_flat_indices, invalid_fill)
        all_sort_order = all_sort_keys.argsort(dim=1)
        gather_index = all_sort_order[:, :, None, None].expand(-1, -1, self.num_cell_tokens, self.embed_dim)
        latent_all = torch.gather(all_latent_unsorted, dim=1, index=gather_index)
        all_valid_mask = torch.gather(all_valid_mask_unsorted, dim=1, index=all_sort_order)

        encoded_remained = self.vit_encoder(
            latent_remained,
            positions_2d=remained_positions,
            valid_cell_mask=remained_valid_mask,
        )
        predictor_tokens = self.encoder_to_predictor(encoded_remained)

        query_tokens = self._build_query_tokens(batch_size, latent_masked.shape[1])
        predictor_input = torch.cat([predictor_tokens, query_tokens], dim=1)
        predictor_positions = torch.cat([remained_positions, masked_positions], dim=1)
        predictor_valid_mask = torch.cat([remained_valid_mask, masked_valid_mask], dim=1)
        predictor_output = self.vit_predictor(
            predictor_input,
            positions_2d=predictor_positions,
            valid_cell_mask=predictor_valid_mask,
        )
        if latent_masked.shape[1] == 0:
            masked_predictor_tokens = predictor_output[:, 0:0]
        else:
            masked_predictor_tokens = predictor_output[:, -latent_masked.shape[1]:]
        latent_predicted = self.predictor_to_latent(masked_predictor_tokens)

        padded_remained = latent_remained
        padded_masked = latent_masked
        padded_predicted = latent_predicted

        padded_tokens_3p5, mask_3p5 = low_remained_tokens, remained_low_valid
        padded_targets_3p5 = low_remained_target
        padded_recovered_3p5 = low_recovered
        padded_pred_noise_3p5 = low_predicted_noise
        padded_target_noise_3p5 = low_target_noise

        padded_tokens_28, mask_28 = high_remained_tokens, remained_high_valid
        padded_targets_28 = high_remained_target
        padded_recovered_28 = high_recovered
        padded_pred_noise_28 = high_predicted_noise
        padded_target_noise_28 = high_target_noise

        return TimeBandJEPAOutput(
            latent_remained=padded_remained,
            latent_masked=padded_masked,
            latent_predicted=padded_predicted,
            latent_all=latent_all,
            latent_remained_mask=remained_valid_mask,
            latent_masked_mask=masked_valid_mask,
            latent_all_mask=all_valid_mask,
            latent_masked_positions=masked_positions,
            masked_aux_3p5=MaskedBandAuxiliaryBatch(
                tokens=padded_tokens_3p5,
                target=padded_targets_3p5,
                valid_mask=mask_3p5,
                recovered=padded_recovered_3p5,
                diffusion_predicted_noise=padded_pred_noise_3p5,
                diffusion_target_noise=padded_target_noise_3p5,
            ),
            masked_aux_28=MaskedBandAuxiliaryBatch(
                tokens=padded_tokens_28,
                target=padded_targets_28,
                valid_mask=mask_28,
                recovered=padded_recovered_28,
                diffusion_predicted_noise=padded_pred_noise_28,
                diffusion_target_noise=padded_target_noise_28,
            ),
        )
