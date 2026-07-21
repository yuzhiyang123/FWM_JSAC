from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from twm.downstream.channel_pred.models import PilotAwareMLPAlignHead
from twm.downstream.channel_pred.training import (
    ChannelPredBatch,
    ChannelPredConfig,
    ChannelPredTrainOutput,
    _latent_metric_triplet,
    _nmse_loss,
    _pilot_mask_from_preprocessed,
    _sample_final_jepa_history_mask,
    _select_current_band_predictions,
    _subband_mask_from_channel,
)
from twm.pipelines.jepa_32x16 import RandomMaskGenerator, TimeBandJEPAConfig, TimeBandJEPAPipeline, build_lazy_config
from twm.pipelines.jepa_32x16.pipeline import TimeBandViTConfig as TimeBand32ViTConfig
from twm.pipelines.jepa_32x16.utils import build_preprocessor
from twm.pipelines.jepa_128x64 import DualScaleMaskGenerator, DualScaleMaskSplit, TimeBand128x64JEPAConfig, TimeBand128x64JEPAPipeline
from twm.pipelines.jepa_128x64.pipeline import TimeBandViTConfig as TimeBand128ViTConfig
from twm.pipelines.jepa_1024x64 import TriScaleMaskGenerator, TriScaleMaskSplit, TimeBand1024x64JEPAConfig, TimeBand1024x64JEPAPipeline
from twm.pipelines.jepa_1024x64.pipeline import TimeBandViTConfig as TimeBand1024ViTConfig

from .models import PilotCrossAttentionPatchRegressor1024x64


@dataclass
class ChannelPredReconstruction1024x64Config:
    pilot_align_mode: str = 'none'
    pilot_align_heads: int = 4
    pilot_align_dropout: float = 0.0
    pilot_align_input_mode: str = 'tok'
    pilot_est_subcarrier_strides: tuple[int, ...] = (8,)
    pilot_est_noise_mode: str = 'uniform_gaussian'
    pilot_est_max_noise_std: float = 1.224744871391589
    pilot_input_noise_var: float = 0.5
    pilot_total_noise_var_target: float = 0.5
    pilot_fixed_nmse: float | None = None
    eval_history_mask_rate: float | None = None
    output_height: int = 1024
    output_width: int = 64
    regressor_patch_size: tuple[int, int] = (32, 16)
    regressor_hidden_dim: int = 1024
    regressor_depth: int = 6
    regressor_num_heads: int = 2
    regressor_mlp_ratio: float = 2.0


def _freeze_backbone_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def _backbone_feature_dim(model: nn.Module) -> int:
    return int(model.num_cell_tokens * model.embed_dim)


def _backbone_encode_reference_latents(model: nn.Module, kind: str, reference_domain: str, x: Tensor, *, freeze_backbone: bool) -> Tensor:
    if x.ndim != 5:
        raise ValueError(f'Expected x [B, N, 2, H, W], got {tuple(x.shape)}')
    batch_size, num_bands = x.shape[:2]
    if kind == '32x16':
        if reference_domain != '32x16':
            raise ValueError(f'Unsupported reference_domain for 32x16 backbone: {reference_domain!r}')
        flat = x.reshape(-1, *x.shape[-3:])
        with torch.no_grad() if freeze_backbone else torch.enable_grad():
            preprocessed = model.preprocessor(flat)
            preprocessed = preprocessed.view(*x.shape[:-3], *preprocessed.shape[1:])
            low_pre = preprocessed[..., : model.num_low_bands, :, :, :]
            high_pre = preprocessed[..., model.num_low_bands :, :, :, :]
            low_features = model.tokenizer_3p5(low_pre.reshape(-1, *low_pre.shape[-3:]))
            high_features = model.tokenizer_28(high_pre.reshape(-1, *high_pre.shape[-3:]))
        low_features = low_features.flatten(start_dim=1).view(*low_pre.shape[:-3], -1)
        high_features = high_features.flatten(start_dim=1).view(*high_pre.shape[:-3], -1)
        return torch.cat([low_features, high_features], dim=-2)
    flat_x = x.reshape(batch_size, num_bands, *x.shape[2:])
    cell_positions = model._build_cell_positions(1, num_bands, x.device).view(num_bands, 2)
    flat_band_ids = cell_positions[:, 1]
    low_mask, high_mask = model._build_band_masks(num_bands, flat_band_ids)
    if kind == '128x64':
        if reference_domain != '32x16':
            raise ValueError(f'Unsupported reference_domain for 128x64 backbone: {reference_domain!r}')
        preprocessor = model.preprocessor_32x16
        tokenizer_low = model.tokenizer_32x16_3p5
        tokenizer_high = model.tokenizer_32x16_28
    elif kind == '1024x64':
        if reference_domain == '32x16':
            preprocessor = model.preprocessor_32x16
            tokenizer_low = model.tokenizer_32x16_3p5
            tokenizer_high = model.tokenizer_32x16_28
        elif reference_domain == '128x64':
            preprocessor = model.preprocessor_128x64
            tokenizer_low = model.tokenizer_128x64_3p5
            tokenizer_high = model.tokenizer_128x64_28
        else:
            raise ValueError(f'Unsupported reference_domain for 1024x64 backbone: {reference_domain!r}')
    else:
        raise ValueError(f'Unsupported backbone kind: {kind!r}')
    with torch.no_grad() if freeze_backbone else torch.enable_grad():
        latent_all = model._encode_all_branch(
            flat_x,
            low_mask=low_mask,
            high_mask=high_mask,
            preprocessor=preprocessor,
            tokenizer_low=tokenizer_low,
            tokenizer_high=tokenizer_high,
        )
    return latent_all.flatten(start_dim=2)


def _backbone_predict_current_latents(model: nn.Module, kind: str, reference_domain: str, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor, *, freeze_backbone: bool, eval_history_mask_rate: float | None) -> Tensor:
    if history_channel.ndim != 6:
        raise ValueError(f'Expected history_channel [B, T, N, 2, H, W], got {tuple(history_channel.shape)}')
    if history_mask.shape != history_channel.shape[:3]:
        raise ValueError(f'Expected history_mask shape {tuple(history_channel.shape[:3])}, got {tuple(history_mask.shape)}')
    if current_channel.shape != history_channel[:, 0].shape:
        raise ValueError(f'Expected current_channel shape {tuple(history_channel[:, 0].shape)}, got {tuple(current_channel.shape)}')

    batch_size, max_history, num_bands = history_channel.shape[:3]
    history_lengths = history_mask.any(dim=2).sum(dim=1).to(dtype=torch.long)
    sampled_history_mask = _sample_final_jepa_history_mask(history_mask.to(device=history_channel.device, dtype=torch.bool), fixed_rate=eval_history_mask_rate)
    full_channel = history_channel.new_zeros(batch_size, max_history + 1, num_bands, *history_channel.shape[3:])

    if kind == '32x16':
        mask = torch.ones(batch_size, max_history + 1, num_bands, device=history_channel.device, dtype=torch.bool)
        for batch_index in range(batch_size):
            history_length = int(history_lengths[batch_index].item())
            if history_length > 0:
                full_channel[batch_index, :history_length] = history_channel[batch_index, :history_length]
                mask[batch_index, :history_length] = sampled_history_mask[batch_index, :history_length]
            full_channel[batch_index, history_length] = current_channel[batch_index]
            mask[batch_index, history_length] = True
        with torch.no_grad() if freeze_backbone else torch.enable_grad():
            output = model(full_channel, mask=mask)
    elif kind == '128x64':
        known_32x16 = torch.zeros(batch_size, max_history + 1, num_bands, device=history_channel.device, dtype=torch.bool)
        known_128x64 = torch.zeros_like(known_32x16)
        unknown = torch.ones_like(known_32x16)
        for batch_index in range(batch_size):
            history_length = int(history_lengths[batch_index].item())
            if history_length > 0:
                full_channel[batch_index, :history_length] = history_channel[batch_index, :history_length]
                visible = ~sampled_history_mask[batch_index, :history_length]
                split_random = torch.rand(history_length, num_bands, device=history_channel.device)
                known_32x16[batch_index, :history_length] = visible & (split_random < 0.5)
                known_128x64[batch_index, :history_length] = visible & ~known_32x16[batch_index, :history_length]
                unknown[batch_index, :history_length] = ~visible
            full_channel[batch_index, history_length] = current_channel[batch_index]
            unknown[batch_index, history_length] = True
        mask_split = DualScaleMaskSplit(known_32x16=known_32x16, known_128x64=known_128x64, unknown=unknown)
        with torch.no_grad() if freeze_backbone else torch.enable_grad():
            output = model(full_channel, mask_split=mask_split)
    elif kind == '1024x64':
        known_32x16 = torch.zeros(batch_size, max_history + 1, num_bands, device=history_channel.device, dtype=torch.bool)
        known_128x64 = torch.zeros_like(known_32x16)
        known_1024x64 = torch.zeros_like(known_32x16)
        unknown = torch.ones_like(known_32x16)
        for batch_index in range(batch_size):
            history_length = int(history_lengths[batch_index].item())
            if history_length > 0:
                full_channel[batch_index, :history_length] = history_channel[batch_index, :history_length]
                visible = ~sampled_history_mask[batch_index, :history_length]
                split_random = torch.rand(history_length, num_bands, device=history_channel.device)
                known_32x16[batch_index, :history_length] = visible & (split_random < (1.0 / 3.0))
                known_128x64[batch_index, :history_length] = visible & (split_random >= (1.0 / 3.0)) & (split_random < (2.0 / 3.0))
                known_1024x64[batch_index, :history_length] = visible & (split_random >= (2.0 / 3.0))
                unknown[batch_index, :history_length] = ~visible
            full_channel[batch_index, history_length] = current_channel[batch_index]
            unknown[batch_index, history_length] = True
        mask_split = TriScaleMaskSplit(
            known_32x16=known_32x16,
            known_128x64=known_128x64,
            known_1024x64=known_1024x64,
            unknown=unknown,
        )
        with torch.no_grad() if freeze_backbone else torch.enable_grad():
            output = model(full_channel, mask_split=mask_split)
    else:
        raise ValueError(f'Unsupported backbone kind: {kind!r}')

    predicted_latents = _select_current_band_predictions(
        output.latent_predicted,
        output.latent_masked_mask,
        output.latent_masked_positions,
        history_lengths,
        num_bands,
    ).flatten(start_dim=2)
    if predicted_latents.shape[1] != num_bands:
        raise ValueError(f'Expected {num_bands} predicted current-band latents, got {predicted_latents.shape[1]}')
    return predicted_latents


def _checkpoint_dir(pretrained_dir: str | Path) -> Path:
    path = Path(pretrained_dir)
    if path.is_dir() and path.name in {'best_model', 'last_model'}:
        return path
    if path.is_dir() and (path / 'best_model').is_dir():
        return path / 'best_model'
    return path


def _metadata_dir(pretrained_dir: str | Path) -> Path:
    path = Path(pretrained_dir)
    if path.is_dir() and path.name in {'best_model', 'last_model'}:
        return path.parent
    return path


def _load_state(module: nn.Module, path: Path) -> None:
    module.load_state_dict(torch.load(path, map_location='cpu'))


def _infer_backbone_kind(checkpoint_dir: Path) -> str:
    if (checkpoint_dir / 'student_tokenizer_1024x64_3p5.pt').exists():
        return '1024x64'
    if (checkpoint_dir / 'student_tokenizer_128x64_3p5.pt').exists():
        return '128x64'
    if (checkpoint_dir / 'student_tokenizer_3p5.pt').exists() or (checkpoint_dir / 'student_tokenizer_32x16_3p5.pt').exists():
        return '32x16'
    raise ValueError(f'Unsupported pretrained downstream backbone at {checkpoint_dir}')


def _build_32x16_backbone(pretrained_dir: str | Path, *, load_weights: bool = True) -> tuple[TimeBandJEPAPipeline, str]:
    checkpoint_dir = _checkpoint_dir(pretrained_dir)
    metadata_dir = _metadata_dir(pretrained_dir)
    payload = json.loads((metadata_dir / 'run_config.json').read_text(encoding='utf-8'))
    jepa_cfg = payload.get('jepa_config')
    if not isinstance(jepa_cfg, dict):
        raise KeyError('Missing jepa_config in old 32x16 run_config.json')
    tokenizer_output_norm = str(payload.get('tokenizer_output_norm', jepa_cfg.get('tokenizer_output_norm', 'batchnorm')))
    model_config = build_lazy_config(
        preset_name=str(jepa_cfg['preset_name']),
        num_low_bands=int(jepa_cfg['num_low_bands']),
        num_tokens=int(jepa_cfg['num_tokens']),
        token_dim=int(jepa_cfg['token_dim']),
        predictor_dim=int(jepa_cfg['predictor_dim']),
        tokenizer_output_norm=tokenizer_output_norm,
    )
    model_config.enable_recovery_head = False
    model_config.enable_cond_diffusion_head = False
    model_config.masked_target_source = str(payload.get('masked_target_source', 'teacher_tokenizer'))
    model = TimeBandJEPAPipeline(mask_generator=RandomMaskGenerator(mask_rate=0.3), config=model_config)
    if load_weights:
        model.load_predictor_core(checkpoint_dir)
    return model, '32x16'


def _build_128x64_backbone(pretrained_dir: str | Path, *, load_weights: bool = True) -> tuple[TimeBand128x64JEPAPipeline, str]:
    checkpoint_dir = _checkpoint_dir(pretrained_dir)
    metadata_dir = _metadata_dir(pretrained_dir)
    payload = json.loads((metadata_dir / 'run_config.json').read_text(encoding='utf-8'))
    resolved = payload['resolved_model_config']
    model_config = TimeBand128x64JEPAConfig(
        num_low_bands=int(resolved['num_low_bands']),
        predictor_dim=int(resolved['predictor_dim']),
        preprocessor_name_32x16=str(resolved.get('preprocessor_name_32x16', 'uniform_grid_32x16_quadrant')),
        preprocessor_name_128x64=str(resolved.get('preprocessor_name_128x64', 'uniform_grid_128x64')),
        tokenizer_name_32x16_3p5=str(resolved['tokenizer_name_32x16_3p5']),
        tokenizer_name_32x16_28=str(resolved['tokenizer_name_32x16_28']),
        tokenizer_name_128x64_3p5=str(resolved['tokenizer_name_128x64_3p5']),
        tokenizer_name_128x64_28=str(resolved['tokenizer_name_128x64_28']),
        num_tokens=int(resolved['num_tokens']),
        token_dim=int(resolved['token_dim']),
        tokenizer_output_norm_32x16=str(resolved.get('tokenizer_output_norm_32x16', 'layernorm')),
        tokenizer_output_norm_128x64=str(resolved.get('tokenizer_output_norm_128x64', payload.get('tokenizer_output_norm_128x64', 'layernorm'))),
        encoder=TimeBand128ViTConfig(**resolved['encoder']),
        predictor=TimeBand128ViTConfig(**resolved['predictor']),
    )
    model = TimeBand128x64JEPAPipeline(mask_generator=DualScaleMaskGenerator(), config=model_config)
    if load_weights:
        _load_state(model.tokenizer_32x16_3p5, checkpoint_dir / 'student_tokenizer_32x16_3p5.pt')
        _load_state(model.tokenizer_32x16_28, checkpoint_dir / 'student_tokenizer_32x16_28.pt')
        _load_state(model.tokenizer_128x64_3p5, checkpoint_dir / 'student_tokenizer_128x64_3p5.pt')
        _load_state(model.tokenizer_128x64_28, checkpoint_dir / 'student_tokenizer_128x64_28.pt')
        _load_state(model.vit_encoder, checkpoint_dir / 'vit_encoder.pt')
        _load_state(model.vit_predictor, checkpoint_dir / 'vit_predictor.pt')
        _load_state(model.encoder_to_predictor, checkpoint_dir / 'encoder_to_predictor.pt')
        _load_state(model.predictor_to_latent, checkpoint_dir / 'predictor_to_latent.pt')
        model.mask_query_tokens.data.copy_(torch.load(checkpoint_dir / 'mask_query_tokens.pt', map_location='cpu').to(model.mask_query_tokens.device))
    return model, '32x16'


def _build_1024x64_backbone(pretrained_dir: str | Path, *, load_weights: bool = True) -> tuple[TimeBand1024x64JEPAPipeline, str]:
    checkpoint_dir = _checkpoint_dir(pretrained_dir)
    metadata_dir = _metadata_dir(pretrained_dir)
    payload = json.loads((metadata_dir / 'run_config.json').read_text(encoding='utf-8'))
    resolved = payload['resolved_model_config']
    reference_domain = str(resolved.get('reference_domain', payload.get('reference_domain', '32x16')))
    model_config = TimeBand1024x64JEPAConfig(
        num_low_bands=int(resolved['num_low_bands']),
        predictor_dim=int(resolved['predictor_dim']),
        preprocessor_name_32x16=str(resolved['preprocessor_name_32x16']),
        preprocessor_name_128x64=str(resolved['preprocessor_name_128x64']),
        preprocessor_name_1024x64=str(resolved['preprocessor_name_1024x64']),
        tokenizer_name_32x16_3p5=str(resolved['tokenizer_name_32x16_3p5']),
        tokenizer_name_32x16_28=str(resolved['tokenizer_name_32x16_28']),
        tokenizer_name_128x64_3p5=str(resolved['tokenizer_name_128x64_3p5']),
        tokenizer_name_128x64_28=str(resolved['tokenizer_name_128x64_28']),
        tokenizer_name_1024x64_3p5=str(resolved['tokenizer_name_1024x64_3p5']),
        tokenizer_name_1024x64_28=str(resolved['tokenizer_name_1024x64_28']),
        num_tokens=int(resolved['num_tokens']),
        token_dim=int(resolved['token_dim']),
        tokenizer_output_norm_32x16=str(resolved['tokenizer_output_norm_32x16']),
        tokenizer_output_norm_128x64=str(resolved['tokenizer_output_norm_128x64']),
        tokenizer_output_norm_1024x64=str(resolved['tokenizer_output_norm_1024x64']),
        reference_domain=reference_domain,
        encoder=TimeBand1024ViTConfig(**resolved['encoder']),
        predictor=TimeBand1024ViTConfig(**resolved['predictor']),
    )
    model = TimeBand1024x64JEPAPipeline(mask_generator=TriScaleMaskGenerator(), config=model_config)
    if load_weights:
        _load_state(model.tokenizer_32x16_3p5, checkpoint_dir / 'student_tokenizer_32x16_3p5.pt')
        _load_state(model.tokenizer_32x16_28, checkpoint_dir / 'student_tokenizer_32x16_28.pt')
        _load_state(model.tokenizer_128x64_3p5, checkpoint_dir / 'student_tokenizer_128x64_3p5.pt')
        _load_state(model.tokenizer_128x64_28, checkpoint_dir / 'student_tokenizer_128x64_28.pt')
        _load_state(model.tokenizer_1024x64_3p5, checkpoint_dir / 'student_tokenizer_1024x64_3p5.pt')
        _load_state(model.tokenizer_1024x64_28, checkpoint_dir / 'student_tokenizer_1024x64_28.pt')
        _load_state(model.vit_encoder, checkpoint_dir / 'vit_encoder.pt')
        _load_state(model.vit_predictor, checkpoint_dir / 'vit_predictor.pt')
        _load_state(model.encoder_to_predictor, checkpoint_dir / 'encoder_to_predictor.pt')
        _load_state(model.predictor_to_latent, checkpoint_dir / 'predictor_to_latent.pt')
        model.mask_query_tokens.data.copy_(torch.load(checkpoint_dir / 'mask_query_tokens.pt', map_location='cpu').to(model.mask_query_tokens.device))
    return model, reference_domain


def build_conditioning_backbone(pretrained_dir: str | Path, *, freeze_backbone: bool = True, load_weights: bool = True) -> tuple[nn.Module, str, str]:
    checkpoint_dir = _checkpoint_dir(pretrained_dir)
    kind = _infer_backbone_kind(checkpoint_dir)
    if kind == '32x16':
        model, reference_domain = _build_32x16_backbone(pretrained_dir, load_weights=load_weights)
    elif kind == '128x64':
        model, reference_domain = _build_128x64_backbone(pretrained_dir, load_weights=load_weights)
    elif kind == '1024x64':
        model, reference_domain = _build_1024x64_backbone(pretrained_dir, load_weights=load_weights)
    else:
        raise ValueError(f'Unsupported pretrained backbone kind: {kind!r}')
    if freeze_backbone:
        _freeze_backbone_module(model)
    return model, kind, reference_domain


class ChannelPredReconstruction1024x64Task(nn.Module):
    def __init__(self, predictor_backbone: nn.Module, *, backbone_kind: str, reference_domain: str, config: ChannelPredReconstruction1024x64Config, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.predictor_backbone = predictor_backbone
        self.backbone_kind = str(backbone_kind)
        self.reference_domain = str(reference_domain)
        self.config = config
        self.freeze_backbone = bool(freeze_backbone)
        self.feature_dim = _backbone_feature_dim(self.predictor_backbone)
        self.reconstruction_preprocessor = build_preprocessor('identity_1024x64')
        self.pilot_align = None
        if config.pilot_align_mode == 'mlp':
            input_dim = int(2 * config.output_height * config.output_width)
            self.pilot_align = PilotAwareMLPAlignHead(
                input_dim=input_dim,
                feature_dim=self.feature_dim,
                num_tokens=int(self.predictor_backbone.num_cell_tokens),
                token_dim=int(self.predictor_backbone.embed_dim),
                input_hw=(config.output_height, config.output_width),
                input_mode=config.pilot_align_input_mode,
                num_heads=config.pilot_align_heads,
                dropout=config.pilot_align_dropout,
            )
        elif config.pilot_align_mode != 'none':
            raise ValueError(f'Unsupported pilot_align_mode: {config.pilot_align_mode!r}')
        self.regressor = PilotCrossAttentionPatchRegressor1024x64(
            token_dim=self.feature_dim,
            pilot_channels=2,
            out_channels=2,
            height=config.output_height,
            width=config.output_width,
            patch_size=config.regressor_patch_size,
            hidden_dim=config.regressor_hidden_dim,
            depth=config.regressor_depth,
            num_heads=config.regressor_num_heads,
            mlp_ratio=config.regressor_mlp_ratio,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.predictor_backbone.eval()
        return self

    def preprocess_channels(self, x: Tensor) -> Tensor:
        if x.ndim not in {5, 6}:
            raise ValueError(f'Expected x shape [B, N, 2, H, W] or [B, T, N, 2, H, W], got {tuple(x.shape)}')
        flat = x.reshape(-1, *x.shape[-3:])
        preprocessed = self.reconstruction_preprocessor(flat)
        return preprocessed.view(*x.shape[:-3], *preprocessed.shape[1:])

    def preprocess_channel_pair(self, channel: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if channel.shape != target.shape:
            raise ValueError(f'Expected paired channel/target shapes to match, got {tuple(channel.shape)} vs {tuple(target.shape)}')
        pair = torch.stack([channel, target], dim=1)
        pair_preprocessed = self.preprocess_channels(pair)
        return pair_preprocessed[:, 0], pair_preprocessed[:, 1]

    def encode_channels(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = _backbone_encode_reference_latents(self.predictor_backbone, self.backbone_kind, self.reference_domain, x, freeze_backbone=self.freeze_backbone)
        preprocessed = self.preprocess_channels(x)
        return features, preprocessed

    def _predict_current_latents(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor) -> Tensor:
        return _backbone_predict_current_latents(self.predictor_backbone, self.backbone_kind, self.reference_domain, history_channel, history_mask, current_channel, freeze_backbone=self.freeze_backbone, eval_history_mask_rate=self.config.eval_history_mask_rate)

    def _build_pilot_estimate_from_preprocessed(self, current_preprocessed: Tensor) -> Tensor:
        if current_preprocessed.ndim != 5:
            raise ValueError(f'Expected current_preprocessed [B, N, C, H, W], got {tuple(current_preprocessed.shape)}')
        pilot_mask = _pilot_mask_from_preprocessed(current_preprocessed)
        masked_preprocessed = current_preprocessed * pilot_mask.to(dtype=current_preprocessed.dtype)
        batch_size, num_bands, _channels, height, width = masked_preprocessed.shape
        noise_mode = str(self.config.pilot_est_noise_mode).strip().lower()
        input_noise_var = max(0.0, float(self.config.pilot_input_noise_var))
        sampled_noise_var = None
        if noise_mode == 'none':
            noisy_preprocessed = masked_preprocessed
        elif noise_mode == 'uniform_gaussian':
            channel_mse = masked_preprocessed.square().mean(dim=(2, 3, 4), keepdim=True)
            sampled_noise_var = torch.rand(batch_size, num_bands, 1, 1, 1, device=masked_preprocessed.device, dtype=masked_preprocessed.dtype) * channel_mse
            noisy_preprocessed = masked_preprocessed + torch.randn_like(masked_preprocessed) * sampled_noise_var.sqrt()
        elif noise_mode == 'fixed_nmse':
            fixed_nmse = float(max(0.0, self.config.pilot_fixed_nmse or 0.0))
            channel_mse = masked_preprocessed.square().mean(dim=(2, 3, 4), keepdim=True)
            sampled_noise_var = channel_mse * fixed_nmse
            noisy_preprocessed = masked_preprocessed + torch.randn_like(masked_preprocessed) * sampled_noise_var.sqrt()
        else:
            raise ValueError(f'Unsupported pilot_est_noise_mode={self.config.pilot_est_noise_mode!r}')
        if noise_mode == 'none':
            pilot_scale = torch.sqrt(torch.full((batch_size, num_bands, 1, 1, 1), 1.0 + input_noise_var, device=masked_preprocessed.device, dtype=masked_preprocessed.dtype))
        else:
            if sampled_noise_var is None:
                raise RuntimeError('sampled_noise_var must be defined for noisy pilot modes')
            pilot_scale = torch.sqrt(1.0 + input_noise_var + sampled_noise_var)
        noisy_for_estimate = noisy_preprocessed / pilot_scale
        stride_choices = tuple(sorted({max(1, int(v)) for v in self.config.pilot_est_subcarrier_strides}))
        pilot_estimate = torch.zeros_like(noisy_for_estimate)
        known_mask = torch.zeros(batch_size, num_bands, 1, height, width, device=noisy_preprocessed.device, dtype=torch.bool)
        if len(stride_choices) == 1:
            stride_choice = torch.full((batch_size, num_bands), int(stride_choices[0]), device=noisy_preprocessed.device, dtype=torch.long)
        else:
            idx = torch.randint(low=0, high=len(stride_choices), size=(batch_size, num_bands), device=noisy_preprocessed.device)
            stride_choice = torch.tensor(stride_choices, device=noisy_preprocessed.device, dtype=torch.long)[idx]
        for batch_index in range(batch_size):
            for band_index in range(num_bands):
                stride = int(stride_choice[batch_index, band_index].item())
                if stride in {128, 192, 256}:
                    # Align large-stride pilot patterns so they always start from the first 64 rows.
                    start_high = min(64, height)
                    start = int(torch.randint(low=0, high=start_high, size=(1,), device=noisy_preprocessed.device).item())
                else:
                    start = int(torch.randint(low=0, high=stride, size=(1,), device=noisy_preprocessed.device).item())
                indices = torch.arange(start, height, stride, device=noisy_preprocessed.device)
                pilot_estimate[batch_index, band_index, :, indices, :] = noisy_for_estimate[batch_index, band_index, :, indices, :]
                known_mask[batch_index, band_index, :, indices, :] = True
        known_mask_float = known_mask.to(dtype=pilot_estimate.dtype)
        pilot_energy = (pilot_estimate.square() * known_mask_float).sum(dim=(2, 3, 4), keepdim=True)
        known_value_count = known_mask_float.sum(dim=(2, 3, 4), keepdim=True) * pilot_estimate.shape[2]
        pilot_rms = torch.sqrt(pilot_energy / known_value_count.clamp_min(1.0)).clamp_min(1e-8)
        pilot_estimate = pilot_estimate / pilot_rms
        inverse_mask = (~known_mask).to(dtype=noisy_preprocessed.dtype)
        return torch.cat([pilot_estimate, inverse_mask], dim=2)

    def compute_condition(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor, subband_mask: Tensor | None = None, current_preprocessed: Tensor | None = None, predicted_latents: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if predicted_latents is None:
            predicted_latents = self._predict_current_latents(history_channel, history_mask, current_channel)
        if current_preprocessed is None:
            current_preprocessed = self.preprocess_channels(current_channel)
        if subband_mask is None:
            subband_mask = _subband_mask_from_channel(current_channel)
        effective_mask = subband_mask.to(dtype=torch.bool)
        pilot_estimate = self._build_pilot_estimate_from_preprocessed(current_preprocessed)
        pilot_estimate = pilot_estimate * effective_mask[..., None, None, None].to(dtype=pilot_estimate.dtype)
        if self.pilot_align is None:
            return predicted_latents, predicted_latents, predicted_latents, pilot_estimate, effective_mask
        calibrated_latents, pilot_latents = self.pilot_align(predicted_latents, pilot_estimate[:, :, :2], effective_mask)
        return predicted_latents, pilot_latents, calibrated_latents, pilot_estimate, effective_mask

    def _regress_current_target(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor, subband_mask: Tensor | None = None, current_gt: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        target_source = current_gt if current_gt is not None else current_channel
        current_preprocessed, target_preprocessed = self.preprocess_channel_pair(current_channel, target_source)
        target_latents, _ = self.encode_channels(current_channel)
        predicted_latents = self._predict_current_latents(history_channel, history_mask, current_channel)
        _, latent_target_power, latent_nmse = _latent_metric_triplet(predicted_latents, target_latents)
        predicted_latents, pilot_latents, calibrated_latents, pilot_estimate, _ = self.compute_condition(
            history_channel,
            history_mask,
            current_channel,
            subband_mask=subband_mask,
            current_preprocessed=current_preprocessed,
            predicted_latents=predicted_latents,
        )
        flat_target = target_preprocessed.reshape(-1, *target_preprocessed.shape[-3:])
        flat_pilot_estimate = pilot_estimate.reshape(-1, *pilot_estimate.shape[-3:])[:, :2]
        condition_tokens = calibrated_latents.reshape(-1, 1, calibrated_latents.shape[-1])
        predicted = self.regressor(flat_pilot_estimate, condition_tokens)
        return predicted, flat_target, latent_target_power, latent_nmse, predicted_latents, pilot_latents, calibrated_latents, pilot_estimate

    def training_step(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor, subband_mask: Tensor | None = None, current_gt: Tensor | None = None) -> ChannelPredTrainOutput:
        predicted, flat_target, latent_target_power, latent_nmse, predicted_latents, pilot_latents, calibrated_latents, pilot_estimate = self._regress_current_target(
            history_channel, history_mask, current_channel, subband_mask=subband_mask, current_gt=current_gt
        )
        x_loss = F.mse_loss(predicted, flat_target)
        recon_loss = _nmse_loss(predicted, flat_target)
        gt_recon_loss = recon_loss if current_gt is not None else None
        return ChannelPredTrainOutput(
            loss=x_loss,
            x_loss=x_loss,
            recon_loss=recon_loss,
            practical_recon_loss=recon_loss,
            gt_recon_loss=gt_recon_loss,
            latent_target_power=latent_target_power,
            latent_nmse=latent_nmse,
            predicted_x0=predicted,
            recovered=predicted,
            predicted_latents=predicted_latents,
            pilot_latents=pilot_latents,
            calibrated_latents=calibrated_latents,
            pilot_estimate=pilot_estimate,
        )

    @torch.no_grad()
    def testing_step(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor, subband_mask: Tensor | None = None, current_gt: Tensor | None = None) -> ChannelPredTrainOutput:
        return self.training_step(history_channel, history_mask, current_channel, subband_mask=subband_mask, current_gt=current_gt)
