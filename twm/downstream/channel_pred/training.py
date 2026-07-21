from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from twm.datasets.shard_access import TrackRef
from .data import build_window_slices, load_channel_pred_track_pairs, split_channel_pred_tracks_train_intra_cross
from .models import (
    ChannelPredConditionalDiffusionModel32x16,
    ChannelPredictionOutput,
    PilotAwareMLPAlignHead,
)


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def _nmse_loss(prediction: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Expected matching shapes for NMSE, got {tuple(prediction.shape)} vs {tuple(target.shape)}")
    mse = F.mse_loss(prediction, target)
    denom = target.pow(2).mean().clamp_min(float(eps))
    return mse / denom


def _latent_metric_triplet(prediction: Tensor, target: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor, Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(f"Expected matching latent shapes, got {tuple(prediction.shape)} vs {tuple(target.shape)}")
    latent_mse = F.mse_loss(prediction, target)
    latent_target_power = target.pow(2).mean().clamp_min(float(eps))
    latent_nmse = latent_mse / latent_target_power
    return latent_mse, latent_target_power, latent_nmse


def _compute_diffusion_scale_from_pilot(pilot_estimate: Tensor, eps: float = 1e-6) -> Tensor:
    if pilot_estimate.ndim != 4 or pilot_estimate.shape[1] < 2:
        raise ValueError(f'Expected pilot_estimate [B, C>=2, H, W], got {tuple(pilot_estimate.shape)}')
    return pilot_estimate.new_ones((pilot_estimate.shape[0], 1, 1, 1))


def _normalize_diffusion_target_and_pilot(target: Tensor, pilot_estimate: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor, Tensor]:
    if target.ndim != 4 or target.shape[1] != 2:
        raise ValueError(f'Expected target [B, 2, H, W], got {tuple(target.shape)}')
    scale = target.new_ones((target.shape[0], 1, 1, 1))
    return target, pilot_estimate, scale


def _denormalize_diffusion_tensor(value: Tensor, scale: Tensor) -> Tensor:
    if value.ndim != 4:
        raise ValueError(f'Expected value as a 4D tensor, got {tuple(value.shape)}')
    return value


def _sample_final_jepa_history_mask(history_mask: Tensor, fixed_rate: float | None = None) -> Tensor:
    if history_mask.ndim != 3:
        raise ValueError(f'Expected history_mask [B, T, N], got {tuple(history_mask.shape)}')
    batch_size = history_mask.shape[0]
    if fixed_rate is not None:
        rate_value = float(max(0.0, min(1.0, fixed_rate)))
        per_sample_rates = torch.full((batch_size,), rate_value, device=history_mask.device, dtype=torch.float32)
    else:
        beta_dist = torch.distributions.Beta(4.0, 1.2)
        scaled = beta_dist.sample((batch_size,)).to(device=history_mask.device, dtype=torch.float32)
        per_sample_rates = 0.1 + scaled * (0.9 - 0.1)
    random_values = torch.rand(history_mask.shape, device=history_mask.device)
    return (random_values < per_sample_rates[:, None, None]) & history_mask.to(device=history_mask.device, dtype=torch.bool)


def _subband_mask_from_channel(current_channel: Tensor) -> Tensor:
    if current_channel.ndim != 5:
        raise ValueError(f'Expected current_channel [B, N, 2, H, W], got {tuple(current_channel.shape)}')
    batch_size, num_bands = current_channel.shape[:2]
    if num_bands <= 0:
        raise ValueError('Expected at least one subband.')
    return torch.ones(batch_size, num_bands, device=current_channel.device, dtype=torch.bool)


def _pilot_mask_from_preprocessed(current_preprocessed: Tensor) -> Tensor:
    if current_preprocessed.ndim != 5:
        raise ValueError(
            f'Expected current_preprocessed [B, N, 2, H, W], got {tuple(current_preprocessed.shape)}'
        )
    return current_preprocessed.abs().sum(dim=2, keepdim=True) > 0


def _select_current_band_predictions(latent_predicted: Tensor, latent_masked_mask: Tensor | None, latent_masked_positions: Tensor | None, current_time_index: Tensor, num_bands: int) -> Tensor:
    if latent_masked_mask is None or latent_masked_positions is None:
        raise ValueError('JEPA output must include latent_masked_mask and latent_masked_positions.')
    if latent_predicted.ndim != 4:
        raise ValueError(f'Expected latent_predicted [B, M, K, D], got {tuple(latent_predicted.shape)}')
    if latent_masked_mask.shape[:2] != latent_predicted.shape[:2]:
        raise ValueError(
            f'latent_masked_mask shape {tuple(latent_masked_mask.shape)} does not match latent_predicted shape {tuple(latent_predicted.shape)}'
        )
    if latent_masked_positions.shape[:2] != latent_predicted.shape[:2] or latent_masked_positions.shape[-1] != 2:
        raise ValueError(
            f'latent_masked_positions shape {tuple(latent_masked_positions.shape)} does not match latent_predicted shape {tuple(latent_predicted.shape)}'
        )
    batch_size = latent_predicted.shape[0]
    feature_shape = latent_predicted.shape[2:]
    selected = latent_predicted.new_zeros((batch_size, num_bands, *feature_shape))
    for batch_index in range(batch_size):
        valid = latent_masked_mask[batch_index]
        positions = latent_masked_positions[batch_index, valid]
        predicted = latent_predicted[batch_index, valid]
        current_positions = positions[:, 0] == current_time_index[batch_index]
        current_band_positions = positions[current_positions, 1]
        current_predicted = predicted[current_positions]
        if current_predicted.shape[0] != num_bands:
            raise ValueError(
                f'Expected {num_bands} current-band predictions, got {current_predicted.shape[0]} for sample {batch_index}'
            )
        selected[batch_index, current_band_positions] = current_predicted
    return selected


@dataclass
class ChannelPredConfig:
    pilot_align_mode: str = 'none'
    channel_mode: str = 'complex'
    diffusion_timesteps: int = 1000
    ddpm_sample_steps: int = 10
    diffusion_prediction_target: str = 'x0'
    diffusion_sampler: str = 'ddpm'
    ddim_eta: float = 0.0
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    down_block_out_channels: tuple[int, ...] = (64, 128, 128)
    layers_per_block: int = 1
    history_num_layers: int = 1
    pilot_align_heads: int = 4
    pilot_align_dropout: float = 0.0
    pilot_align_input_mode: str = 'tok'
    pilot_est_subcarrier_strides: tuple[int, ...] = (8,)
    pilot_est_noise_mode: str = 'uniform_gaussian'
    pilot_est_max_noise_std: float = 1.224744871391589
    pilot_input_noise_var: float = 0.5
    pilot_total_noise_var_target: float = 0.5
    pilot_fixed_nmse: float | None = None
    pilot_input_blend_weight: float = 0.5
    fusion_mode: str = 'split_cross_attn'
    normalize_diffusion_io: bool = False
    eval_history_mask_rate: float | None = None


@dataclass
class ChannelPredBatch:
    history_channel: Tensor
    history_mask: Tensor
    current_channel: Tensor
    pilot_mask: Tensor
    subband_mask: Tensor
    current_gt: Tensor | None = None


@dataclass
class ChannelPredTrainOutput:
    loss: Tensor
    x_loss: Tensor
    recon_loss: Tensor
    practical_recon_loss: Tensor
    gt_recon_loss: Tensor | None
    latent_target_power: Tensor
    latent_nmse: Tensor
    predicted_x0: Tensor
    recovered: Tensor
    predicted_latents: Tensor
    pilot_latents: Tensor
    calibrated_latents: Tensor
    pilot_estimate: Tensor


class _ChannelPredTrackDataset(Dataset):
    def __init__(
        self,
        track_refs: Sequence[TrackRef],
        *,
        window_length: int,
        random_windows_per_sequence: int,
        seed: int,
    ) -> None:
        self.track_refs = list(track_refs)
        self.window_length = int(window_length)
        self.random_windows_per_sequence = int(random_windows_per_sequence)
        rng = np.random.default_rng(seed)
        self.index: list[tuple[int, int, int]] = []
        for track_idx, ref in enumerate(self.track_refs):
            for start_idx, length in build_window_slices(
                int(ref.shape[0]),
                window_length=self.window_length,
                random_windows_per_sequence=self.random_windows_per_sequence,
                rng=rng,
            ):
                self.index.append((track_idx, start_idx, length))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        track_idx, start_idx, length = self.index[index]
        ref = self.track_refs[track_idx]
        channel = torch.tensor(np.asarray(ref.load_gt()[start_idx : start_idx + length], dtype=np.float32))
        gt = torch.tensor(np.asarray(ref.load_gt()[start_idx : start_idx + length], dtype=np.float32)) if ref.gt_key is not None else None
        if channel.ndim != 5:
            raise ValueError(f'Expected channel track [T, N, 2, H, W], got {tuple(channel.shape)}')
        if channel.shape[0] < 2:
            raise ValueError('Each channel-pred window must contain at least 2 time steps.')
        history_channel = channel[:-1]
        current_channel = channel[-1]
        history_mask = history_channel.abs().sum(dim=(2, 3, 4)) > 0
        pilot_mask = current_channel.abs().sum(dim=1, keepdim=True) > 0
        subband_mask = _subband_mask_from_channel(current_channel.unsqueeze(0)).squeeze(0)
        return {
            'history_channel': history_channel,
            'history_mask': history_mask,
            'current_channel': current_channel,
            'pilot_mask': pilot_mask,
            'subband_mask': subband_mask,
            'current_gt': None if gt is None else gt[-1],
        }


def _collate_channel_pred_batch(items: list[dict[str, Tensor]]) -> ChannelPredBatch:
    if not items:
        raise ValueError('Cannot collate an empty batch.')
    batch_size = len(items)
    max_t = max(int(item['history_channel'].shape[0]) for item in items)
    num_bands = int(items[0]['history_channel'].shape[1])
    complex_dim = int(items[0]['history_channel'].shape[2])
    height = int(items[0]['history_channel'].shape[3])
    width = int(items[0]['history_channel'].shape[4])
    history_channel = torch.zeros(batch_size, max_t, num_bands, complex_dim, height, width, dtype=torch.float32)
    history_mask = torch.zeros(batch_size, max_t, num_bands, dtype=torch.bool)
    current_channel = torch.zeros(batch_size, num_bands, complex_dim, height, width, dtype=torch.float32)
    pilot_mask = torch.zeros(batch_size, num_bands, 1, height, width, dtype=torch.bool)
    subband_mask = torch.zeros(batch_size, num_bands, dtype=torch.bool)
    current_gt: Tensor | None = None
    has_gt = any(item['current_gt'] is not None for item in items)
    if has_gt:
        current_gt = torch.zeros(batch_size, num_bands, complex_dim, height, width, dtype=torch.float32)
    for idx, item in enumerate(items):
        length = int(item['history_channel'].shape[0])
        history_channel[idx, :length] = item['history_channel'].float()
        history_mask[idx, :length] = item['history_mask'].bool()
        current_channel[idx] = item['current_channel'].float()
        pilot_mask[idx] = item['pilot_mask'].bool()
        subband_mask[idx] = item['subband_mask'].bool()
        if current_gt is not None and item['current_gt'] is not None:
            current_gt[idx] = item['current_gt'].float()
    return ChannelPredBatch(
        history_channel=history_channel,
        history_mask=history_mask,
        current_channel=current_channel,
        pilot_mask=pilot_mask,
        subband_mask=subband_mask,
        current_gt=current_gt,
    )


def build_channel_pred_three_way_dataloaders(
    dataset_file: str | Path | Sequence[str | Path],
    *,
    batch_size: int,
    window_length: int = 16,
    random_windows_per_sequence: int = 30,
    num_workers: int = 0,
    seed: int = 42,
    train_paths: int = 512,
    test_paths: int = 88,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    track_refs, scenario_names = load_channel_pred_track_pairs(dataset_file=dataset_file, return_scenarios=True)
    train_refs, test_refs, cross_refs = split_channel_pred_tracks_train_intra_cross(
        tracks=track_refs,
        seed=seed,
        scenario_names=scenario_names,
        train_paths=train_paths,
        test_paths=test_paths,
    )
    train_dataset = _ChannelPredTrackDataset(
        train_refs,
        window_length=window_length,
        random_windows_per_sequence=random_windows_per_sequence,
        seed=seed,
    )
    test_dataset = _ChannelPredTrackDataset(
        test_refs,
        window_length=window_length,
        random_windows_per_sequence=max(1, random_windows_per_sequence),
        seed=seed + 1,
    )
    cross_dataset = _ChannelPredTrackDataset(
        cross_refs,
        window_length=window_length,
        random_windows_per_sequence=max(1, random_windows_per_sequence),
        seed=seed + 2,
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate_channel_pred_batch),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_channel_pred_batch),
        DataLoader(cross_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_channel_pred_batch),
    )


class ChannelPredTask(nn.Module):
    def __init__(
        self,
        predictor_backbone: nn.Module,
        *,
        num_low_bands: int,
        preprocessed_shape: tuple[int, int, int],
        config: ChannelPredConfig,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if config.pilot_align_mode not in {'none', 'mlp'}:
            raise ValueError(f'Unsupported pilot_align_mode: {config.pilot_align_mode}')
        if config.channel_mode != 'complex':
            raise ValueError(
                f"Channel prediction supports only channel_mode='complex'; got {config.channel_mode!r}. "
                "Amplitude-only mode is reserved for recovery tasks."
            )
        self.predictor_backbone = copy.deepcopy(predictor_backbone)
        self.tokenizer_3p5 = self.predictor_backbone.tokenizer_3p5
        self.tokenizer_28 = self.predictor_backbone.tokenizer_28
        self.preprocessor = self.predictor_backbone.preprocessor
        self.num_low_bands = int(num_low_bands)
        self.preprocessed_shape = tuple(int(v) for v in preprocessed_shape)
        self.config = config
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            if hasattr(self.predictor_backbone, 'freeze_tokenizers'):
                self.predictor_backbone.freeze_tokenizers()
            _freeze_module(self.preprocessor)
            _freeze_module(self.predictor_backbone.vit_encoder)
            _freeze_module(self.predictor_backbone.encoder_to_predictor)
            _freeze_module(self.predictor_backbone.vit_predictor)
            _freeze_module(self.predictor_backbone.predictor_to_latent)
            self.predictor_backbone.mask_query_tokens.requires_grad_(False)
            self.predictor_backbone.eval()
        self.num_tokens = int(self.tokenizer_3p5.Nt)
        self.token_width = int(self.tokenizer_3p5.Lt)
        self.feature_dim = self.num_tokens * self.token_width
        self.pilot_align = None
        self.latent_adapter = None
        if config.pilot_align_mode == 'mlp':
            input_dim = int(2 * self.preprocessed_shape[1] * self.preprocessed_shape[2])
            self.pilot_align = PilotAwareMLPAlignHead(
                input_dim=input_dim,
                feature_dim=self.feature_dim,
                num_tokens=self.num_tokens,
                token_dim=self.token_width,
                input_hw=(self.preprocessed_shape[1], self.preprocessed_shape[2]),
                input_mode=config.pilot_align_input_mode,
                num_heads=config.pilot_align_heads,
                dropout=config.pilot_align_dropout,
            )
        self.diffusion_model = ChannelPredConditionalDiffusionModel32x16(
            token_dim=self.feature_dim,
            down_block_out_channels=tuple(int(v) for v in config.down_block_out_channels),
            layers_per_block=int(config.layers_per_block),
            num_train_timesteps=int(config.diffusion_timesteps),
            beta_start=float(config.beta_start),
            beta_end=float(config.beta_end),
            sample_channels=2,
        )

    @classmethod
    def from_backbone(cls, backbone: nn.Module, config: ChannelPredConfig, *, freeze_backbone: bool = True) -> 'ChannelPredTask':
        return cls(
            predictor_backbone=backbone,
            num_low_bands=int(backbone.num_low_bands),
            preprocessed_shape=tuple(int(v) for v in backbone.preprocessed_shape),
            config=config,
            freeze_backbone=freeze_backbone,
        )

    @classmethod
    def from_stage2_checkpoint(
        cls,
        backbone: nn.Module,
        pretrained_dir: str | Path,
        config: ChannelPredConfig,
        *,
        freeze_backbone: bool = True,
    ) -> 'ChannelPredTask':
        loaded_backbone = copy.deepcopy(backbone)
        if not hasattr(loaded_backbone, 'load_predictor_core'):
            raise TypeError('Stage-2 backbone must provide load_predictor_core(...)')
        loaded_backbone.load_predictor_core(pretrained_dir)
        return cls.from_backbone(loaded_backbone, config=config, freeze_backbone=freeze_backbone)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.predictor_backbone.eval()
            self.tokenizer_3p5.eval()
            self.tokenizer_28.eval()
            self.preprocessor.eval()
        return self

    def _encode_group(self, x: Tensor, tokenizer: AbstractCellTokenizer) -> tuple[Tensor, Tensor]:
        flat = x.reshape(-1, *x.shape[-3:])
        if self.freeze_backbone:
            with torch.no_grad():
                preprocessed = self.preprocessor(flat)
                tokens = tokenizer(preprocessed)
        else:
            preprocessed = self.preprocessor(flat)
            tokens = tokenizer(preprocessed)
        features = tokens.flatten(start_dim=1)
        preprocessed = preprocessed.view(*x.shape[:-3], *preprocessed.shape[1:])
        features = features.view(*x.shape[:-3], -1)
        return features, preprocessed

    def preprocess_channels(self, x: Tensor) -> Tensor:
        if x.ndim not in {5, 6}:
            raise ValueError(f'Expected x shape [B, N, 2, H, W] or [B, T, N, 2, H, W], got {tuple(x.shape)}')
        num_bands = x.shape[-4]
        if self.num_low_bands <= 0 or self.num_low_bands >= num_bands:
            raise ValueError(f'num_low_bands must be in (0, N), got {self.num_low_bands} for N={num_bands}')
        flat = x.reshape(-1, *x.shape[-3:])
        if self.freeze_backbone:
            with torch.no_grad():
                preprocessed = self.preprocessor(flat)
        else:
            preprocessed = self.preprocessor(flat)
        return preprocessed.view(*x.shape[:-3], *preprocessed.shape[1:])

    def preprocess_channel_pair(self, channel: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if channel.shape != target.shape:
            raise ValueError(f'Expected paired channel/target shapes to match, got {tuple(channel.shape)} vs {tuple(target.shape)}')
        if channel.ndim not in {5, 6}:
            raise ValueError(f'Expected channel/target shape [B, N, 2, H, W] or [B, T, N, 2, H, W], got {tuple(channel.shape)}')
        pair = torch.stack([channel, target], dim=1)
        pair_preprocessed = self.preprocess_channels(pair)
        return pair_preprocessed[:, 0], pair_preprocessed[:, 1]

    def encode_channels(self, x: Tensor) -> tuple[Tensor, Tensor]:
        preprocessed = self.preprocess_channels(x)
        low_pre = preprocessed[..., : self.num_low_bands, :, :, :]
        high_pre = preprocessed[..., self.num_low_bands :, :, :, :]
        low_features = self.tokenizer_3p5(low_pre.reshape(-1, *low_pre.shape[-3:]))
        high_features = self.tokenizer_28(high_pre.reshape(-1, *high_pre.shape[-3:]))
        low_features = low_features.flatten(start_dim=1).view(*low_pre.shape[:-3], -1)
        high_features = high_features.flatten(start_dim=1).view(*high_pre.shape[:-3], -1)
        features = torch.cat([low_features, high_features], dim=-2)
        return features, preprocessed

    def _predict_current_latents(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor) -> Tensor:
        
        if history_channel.ndim != 6:
            raise ValueError(f'Expected history_channel [B, T, N, 2, H, W], got {tuple(history_channel.shape)}')
        if history_mask.shape != history_channel.shape[:3]:
            raise ValueError(f'Expected history_mask shape {tuple(history_channel.shape[:3])}, got {tuple(history_mask.shape)}')
        if current_channel.shape != history_channel[:, 0].shape:
            raise ValueError(f'Expected current_channel shape {tuple(history_channel[:, 0].shape)}, got {tuple(current_channel.shape)}')

        batch_size, max_history, num_bands = history_channel.shape[:3]
        history_lengths = history_mask.any(dim=2).sum(dim=1).to(dtype=torch.long)
        sampled_history_mask = _sample_final_jepa_history_mask(history_mask.to(device=history_channel.device, dtype=torch.bool), fixed_rate=self.config.eval_history_mask_rate)
        full_channel = history_channel.new_zeros(batch_size, max_history + 1, num_bands, *history_channel.shape[3:])
        mask = torch.ones(batch_size, max_history + 1, num_bands, device=history_channel.device, dtype=torch.bool)
        for batch_index in range(batch_size):
            history_length = int(history_lengths[batch_index].item())
            if history_length > 0:
                full_channel[batch_index, :history_length] = history_channel[batch_index, :history_length]
                mask[batch_index, :history_length] = sampled_history_mask[batch_index, :history_length]
            full_channel[batch_index, history_length] = current_channel[batch_index]
            mask[batch_index, history_length] = True
        if self.freeze_backbone:
            with torch.no_grad():
                output = self.predictor_backbone(full_channel, mask=mask)
        else:
            output = self.predictor_backbone(full_channel, mask=mask)
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

    def _normalize_condition_latents(self, latents: Tensor) -> Tensor:
        return latents

    def _sample_noisy_target(self, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        noise = torch.randn_like(target)
        timesteps = self.diffusion_model.sample_timesteps(target.shape[0], target.device)
        noisy_target = self.diffusion_model.add_noise(target, noise, timesteps)
        return noisy_target, noise, timesteps

    def _predict_noise(self, noisy_target: Tensor, predicted_x0: Tensor, timesteps: Tensor) -> Tensor:
        return self.diffusion_model.predict_noise(noisy_target, predicted_x0, timesteps)

    def _predict_x0(self, noisy_target: Tensor, predicted_noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.diffusion_model.predict_x0(noisy_target, predicted_noise, timesteps)

    def _predict_velocity_target(self, target: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.diffusion_model.predict_velocity_target(target, noise, timesteps)

    def _predict_x0_from_velocity(self, noisy_target: Tensor, predicted_v: Tensor, timesteps: Tensor) -> Tensor:
        return self.diffusion_model.predict_x0_from_velocity(noisy_target, predicted_v, timesteps)

    def _predict_noise_from_velocity(self, noisy_target: Tensor, predicted_v: Tensor, timesteps: Tensor) -> Tensor:
        return self.diffusion_model.predict_noise_from_velocity(noisy_target, predicted_v, timesteps)

    def _prediction_target_mode(self) -> str:
        target = str(self.config.diffusion_prediction_target).strip().lower()
        if target in {'x0', 'sample'}:
            return 'x0'
        if target in {'v', 'v_prediction', 'velocity'}:
            return 'v_prediction'
        if target in {'noise', 'epsilon', 'eps'}:
            return 'noise'
        raise ValueError(f'Unsupported diffusion_prediction_target={self.config.diffusion_prediction_target!r}')

    def _build_pilot_estimate_from_preprocessed(self, current_preprocessed: Tensor) -> Tensor:
        if current_preprocessed.ndim != 5:
            raise ValueError(f'Expected current_preprocessed [B, N, C, H, W], got {tuple(current_preprocessed.shape)}')

        # Derive the known mask directly from the realized preprocessed tensor so the
        # coarse estimate follows the same downsampling path as the diffusion target.
        pilot_mask = _pilot_mask_from_preprocessed(current_preprocessed)
        masked_preprocessed = current_preprocessed * pilot_mask.to(dtype=current_preprocessed.dtype)
        batch_size, num_bands, channels, height, width = masked_preprocessed.shape

        noise_mode = str(self.config.pilot_est_noise_mode).strip().lower()
        max_noise_std = max(0.0, float(self.config.pilot_est_max_noise_std))
        input_noise_var = max(0.0, float(getattr(self.config, 'pilot_input_noise_var', 0.5)))
        total_noise_var_target = max(input_noise_var, float(getattr(self.config, 'pilot_total_noise_var_target', input_noise_var + max_noise_std * max_noise_std)))
        effective_max_noise_var = max(0.0, total_noise_var_target - input_noise_var)
        effective_max_noise_std = min(max_noise_std, float(effective_max_noise_var ** 0.5))
        sampled_noise_var = None
        if noise_mode == 'none':
            noisy_preprocessed = masked_preprocessed
        elif noise_mode == 'fixed_gaussian':
            noisy_preprocessed = masked_preprocessed + torch.randn_like(masked_preprocessed) * effective_max_noise_std
            sampled_noise_var = torch.full(
                (batch_size, num_bands, 1, 1, 1),
                effective_max_noise_std * effective_max_noise_std,
                device=masked_preprocessed.device,
                dtype=masked_preprocessed.dtype,
            )
        elif noise_mode == 'uniform_gaussian':
            channel_mse = masked_preprocessed.square().mean(dim=(2, 3, 4), keepdim=True)
            sampled_noise_var = torch.rand(
                batch_size,
                num_bands,
                1,
                1,
                1,
                device=masked_preprocessed.device,
                dtype=masked_preprocessed.dtype,
            ) * channel_mse
            noise_std = sampled_noise_var.sqrt()
            noisy_preprocessed = masked_preprocessed + torch.randn_like(masked_preprocessed) * noise_std
        elif noise_mode == 'fixed_nmse':
            fixed_nmse = float(max(0.0, getattr(self.config, 'pilot_fixed_nmse', 0.0) or 0.0))
            channel_mse = masked_preprocessed.square().mean(dim=(2, 3, 4), keepdim=True)
            sampled_noise_var = channel_mse * fixed_nmse
            noise_std = sampled_noise_var.sqrt()
            noisy_preprocessed = masked_preprocessed + torch.randn_like(masked_preprocessed) * noise_std
        else:
            raise ValueError(
                f"Unsupported pilot_est_noise_mode={self.config.pilot_est_noise_mode!r}; expected one of 'none', 'fixed_gaussian', 'uniform_gaussian', 'fixed_nmse'."
            )

        if noise_mode == 'none':
            pilot_scale = torch.sqrt(torch.full((batch_size, num_bands, 1, 1, 1), 1.0 + input_noise_var, device=masked_preprocessed.device, dtype=masked_preprocessed.dtype))
        else:
            if sampled_noise_var is None:
                raise RuntimeError('sampled_noise_var must be defined for noisy pilot modes.')
            pilot_scale = torch.sqrt(1.0 + input_noise_var + sampled_noise_var)
        noisy_for_estimate = noisy_preprocessed / pilot_scale

        stride_choices = tuple(sorted({max(1, int(v)) for v in self.config.pilot_est_subcarrier_strides}))
        if not stride_choices:
            raise ValueError('pilot_est_subcarrier_strides must contain at least one positive stride.')

        pilot_estimate = torch.zeros_like(noisy_for_estimate)
        known_mask = torch.zeros(batch_size, num_bands, 1, height, width, device=noisy_preprocessed.device, dtype=torch.bool)
        if len(stride_choices) == 1:
            stride_choice = torch.full((batch_size, num_bands), int(stride_choices[0]), device=noisy_preprocessed.device, dtype=torch.long)
        else:
            stride_choice_idx = torch.randint(
                low=0,
                high=len(stride_choices),
                size=(batch_size, num_bands),
                device=noisy_preprocessed.device,
            )
            stride_choice = torch.tensor(stride_choices, device=noisy_preprocessed.device, dtype=torch.long)[stride_choice_idx]
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

    @torch.no_grad()
    def _sample_channel(self, condition_tokens: Tensor, pilot_estimate: Tensor, init_source: Tensor) -> Tensor:
        base_scheduler: DDPMScheduler = self.diffusion_model.noise_scheduler
        scheduler = DDIMScheduler.from_config(base_scheduler.config, clip_sample=False) if self.config.diffusion_sampler == 'ddim' else base_scheduler
        if init_source.ndim != 4 or tuple(init_source.shape[1:]) != (2, 32, 16):
            raise ValueError(f"Expected init_source shape [B, 2, 32, 16], got {tuple(init_source.shape)}")
        expected_pilot_shape = (init_source.shape[0], 3, init_source.shape[2], init_source.shape[3])
        if pilot_estimate.shape != expected_pilot_shape:
            raise ValueError(f"Expected pilot_estimate shape {expected_pilot_shape}, got {tuple(pilot_estimate.shape)}")

        pilot_channels = pilot_estimate[:, :2]
        known_mask = (1.0 - pilot_estimate[:, 2:3]).to(dtype=pilot_channels.dtype)
        blend_weight = float(max(0.0, min(1.0, self.config.pilot_input_blend_weight)))

        sample_steps = max(1, min(int(self.config.ddpm_sample_steps), int(scheduler.config.num_train_timesteps)))
        scheduler.set_timesteps(sample_steps, device=condition_tokens.device)
        initial_timestep = scheduler.timesteps[0]
        batch_t0 = torch.full((init_source.shape[0],), int(initial_timestep.item()), device=condition_tokens.device, dtype=torch.long)
        init_noise = torch.randn_like(init_source)
        sample = scheduler.add_noise(init_source, init_noise, batch_t0)

        # Keep known pilot locations anchored to the coarse estimate even when starting from a noised channel.
        sample = sample * (1.0 - known_mask) + pilot_channels * known_mask

        for timestep in scheduler.timesteps:
            batch_t = torch.full((init_source.shape[0],), int(timestep.item()), device=condition_tokens.device, dtype=torch.long)
            raw_prediction = self.diffusion_model(sample, pilot_estimate, condition_tokens, batch_t).predicted_x0
            prediction_mode = self._prediction_target_mode()
            if prediction_mode == 'v_prediction':
                predicted_noise = self._predict_noise_from_velocity(sample, raw_prediction, batch_t)
            elif prediction_mode == 'noise':
                predicted_noise = raw_prediction
            else:
                predicted_noise = self._predict_noise(sample, raw_prediction, batch_t)
            if self.config.diffusion_sampler == 'ddim':
                sample = scheduler.step(predicted_noise, timestep, sample, eta=float(self.config.ddim_eta)).prev_sample
            else:
                sample = scheduler.step(predicted_noise, timestep, sample).prev_sample
            blended_known = (1.0 - blend_weight) * sample + blend_weight * pilot_channels
            sample = sample * (1.0 - known_mask) + blended_known * known_mask
        return sample

    def _prepare_diffusion_io(self, target: Tensor, pilot_estimate: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.config.normalize_diffusion_io:
            return _normalize_diffusion_target_and_pilot(target, pilot_estimate)
        scale = torch.ones((target.shape[0], 1, 1, 1), dtype=target.dtype, device=target.device)
        return target, pilot_estimate, scale

    def compute_condition(
        self,
        history_channel: Tensor,
        history_mask: Tensor,
        current_channel: Tensor,
        subband_mask: Tensor | None = None,
        current_preprocessed: Tensor | None = None,
        predicted_latents: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            # In the no-align path, the model must never access current-frame latents.
            # Diffusion always receives predicted history latents as token condition, while
            # pilot_estimate is nonzero only for known current subbands.
            return predicted_latents, predicted_latents, predicted_latents, pilot_estimate, effective_mask
        calibrated_latents, pilot_latents = self.pilot_align(
            predicted_latents,
            pilot_estimate[:, :, :2],
            effective_mask,
        )
        return predicted_latents, pilot_latents, calibrated_latents, pilot_estimate, effective_mask

    @torch.no_grad()
    def latent_diagnostics_step(self, history_channel: Tensor, history_mask: Tensor, current_channel: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        predicted_latents = self._predict_current_latents(history_channel, history_mask, current_channel)
        target_latents, _ = self.encode_channels(current_channel)
        _, latent_target_power, latent_nmse = _latent_metric_triplet(predicted_latents, target_latents)
        return predicted_latents, latent_target_power, latent_nmse

    def training_step(
        self,
        history_channel: Tensor,
        history_mask: Tensor,
        current_channel: Tensor,
        subband_mask: Tensor | None = None,
        current_gt: Tensor | None = None,
    ) -> ChannelPredTrainOutput:
        target_source = current_gt if current_gt is not None else current_channel
        current_preprocessed, target_preprocessed = self.preprocess_channel_pair(current_channel, target_source)
        target_latents, _ = self.encode_channels(current_channel)
        predicted_latents = self._predict_current_latents(history_channel, history_mask, current_channel)
        _, latent_target_power, latent_nmse = _latent_metric_triplet(predicted_latents, target_latents)
        predicted_latents, pilot_latents, calibrated_latents, pilot_estimate, effective_mask = self.compute_condition(
            history_channel, history_mask, current_channel, subband_mask=subband_mask, current_preprocessed=current_preprocessed, predicted_latents=predicted_latents
        )
        current_target = target_preprocessed
        flat_target = current_target.reshape(-1, *current_target.shape[-3:])
        flat_pilot_estimate = pilot_estimate.reshape(-1, *pilot_estimate.shape[-3:])
        normalized_target, normalized_pilot_estimate, diffusion_scale = self._prepare_diffusion_io(flat_target, flat_pilot_estimate)
        condition_tokens = calibrated_latents.reshape(-1, 1, calibrated_latents.shape[-1])
        noisy_target, true_noise, timesteps = self._sample_noisy_target(normalized_target)
        raw_prediction = self.diffusion_model(noisy_target, normalized_pilot_estimate, condition_tokens, timesteps).predicted_x0
        prediction_mode = self._prediction_target_mode()
        if prediction_mode == 'v_prediction':
            velocity_target = self._predict_velocity_target(normalized_target, true_noise, timesteps)
            x_loss = F.mse_loss(raw_prediction, velocity_target)
            predicted_x0_normalized = self._predict_x0_from_velocity(noisy_target, raw_prediction, timesteps)
        elif prediction_mode == 'noise':
            x_loss = F.mse_loss(raw_prediction, true_noise)
            predicted_x0_normalized = self._predict_x0(noisy_target, raw_prediction, timesteps)
        else:
            x_loss = F.mse_loss(raw_prediction, normalized_target)
            predicted_x0_normalized = raw_prediction
        predicted_x0 = _denormalize_diffusion_tensor(predicted_x0_normalized, diffusion_scale)
        recovered = predicted_x0
        recon_loss = _nmse_loss(recovered, flat_target)
        gt_recon_loss = recon_loss if current_gt is not None else None
        return ChannelPredTrainOutput(
            loss=x_loss,
            x_loss=x_loss,
            recon_loss=recon_loss,
            practical_recon_loss=recon_loss,
            gt_recon_loss=gt_recon_loss,
            latent_target_power=latent_target_power,
            latent_nmse=latent_nmse,
            predicted_x0=predicted_x0,
            recovered=recovered,
            predicted_latents=predicted_latents,
            pilot_latents=pilot_latents,
            calibrated_latents=calibrated_latents,
            pilot_estimate=pilot_estimate,
        )

    @torch.no_grad()
    def testing_step(
        self,
        history_channel: Tensor,
        history_mask: Tensor,
        current_channel: Tensor,
        subband_mask: Tensor | None = None,
        current_gt: Tensor | None = None,
    ) -> ChannelPredTrainOutput:
        target_source = current_gt if current_gt is not None else current_channel
        current_preprocessed, target_preprocessed = self.preprocess_channel_pair(current_channel, target_source)
        target_latents, _ = self.encode_channels(current_channel)
        predicted_latents = self._predict_current_latents(history_channel, history_mask, current_channel)
        _, latent_target_power, latent_nmse = _latent_metric_triplet(predicted_latents, target_latents)
        predicted_latents, pilot_latents, calibrated_latents, pilot_estimate, effective_mask = self.compute_condition(
            history_channel, history_mask, current_channel, subband_mask=subband_mask, current_preprocessed=current_preprocessed, predicted_latents=predicted_latents
        )
        current_target = target_preprocessed
        flat_target = current_target.reshape(-1, *current_target.shape[-3:])
        condition_tokens = calibrated_latents.reshape(-1, 1, calibrated_latents.shape[-1])
        flat_pilot_estimate = pilot_estimate.reshape(-1, *pilot_estimate.shape[-3:])
        normalized_target, normalized_pilot_estimate, diffusion_scale = self._prepare_diffusion_io(flat_target, flat_pilot_estimate)
        noisy_target, true_noise, timesteps = self._sample_noisy_target(normalized_target)
        raw_prediction = self.diffusion_model(noisy_target, normalized_pilot_estimate, condition_tokens, timesteps).predicted_x0
        prediction_mode = self._prediction_target_mode()
        if prediction_mode == 'v_prediction':
            velocity_target = self._predict_velocity_target(normalized_target, true_noise, timesteps)
            x_loss = F.mse_loss(raw_prediction, velocity_target)
            predicted_x0_normalized = self._predict_x0_from_velocity(noisy_target, raw_prediction, timesteps)
        elif prediction_mode == 'noise':
            x_loss = F.mse_loss(raw_prediction, true_noise)
            predicted_x0_normalized = self._predict_x0(noisy_target, raw_prediction, timesteps)
        else:
            x_loss = F.mse_loss(raw_prediction, normalized_target)
            predicted_x0_normalized = raw_prediction
        predicted_x0 = _denormalize_diffusion_tensor(predicted_x0_normalized, diffusion_scale)
        recovered_normalized = self._sample_channel(condition_tokens, normalized_pilot_estimate, normalized_target)
        recovered = _denormalize_diffusion_tensor(recovered_normalized, diffusion_scale)
        recon_loss = _nmse_loss(recovered, flat_target)
        gt_recon_loss: Tensor | None = None
        if current_gt is not None:
            _, gt_preprocessed = self.preprocess_channel_pair(current_channel, current_gt)
            gt_target = gt_preprocessed
            gt_recon_loss = _nmse_loss(recovered, gt_target.reshape(-1, *gt_target.shape[-3:]))
        return ChannelPredTrainOutput(
            loss=x_loss,
            x_loss=x_loss,
            recon_loss=recon_loss,
            practical_recon_loss=recon_loss,
            gt_recon_loss=gt_recon_loss,
            latent_target_power=latent_target_power,
            latent_nmse=latent_nmse,
            predicted_x0=predicted_x0,
            recovered=recovered,
            predicted_latents=predicted_latents,
            pilot_latents=pilot_latents,
            calibrated_latents=calibrated_latents,
            pilot_estimate=pilot_estimate,
        )


@torch.no_grad()
def evaluate_channel_pred_batch(
    task: ChannelPredTask,
    batch: ChannelPredBatch,
) -> ChannelPredTrainOutput:
    was_training = task.training
    task.eval()
    result = task.testing_step(
        batch.history_channel,
        batch.history_mask,
        batch.current_channel,
        subband_mask=batch.subband_mask,
        current_gt=batch.current_gt,
    )
    if was_training:
        task.train()
    return result


def train_channel_pred_batch(
    task: ChannelPredTask,
    batch: ChannelPredBatch,
    optimizer: torch.optim.Optimizer,
) -> ChannelPredTrainOutput:
    task.train()
    optimizer.zero_grad(set_to_none=True)
    result = task.training_step(
        batch.history_channel,
        batch.history_mask,
        batch.current_channel,
        subband_mask=batch.subband_mask,
        current_gt=batch.current_gt,
    )
    result.loss.backward()
    optimizer.step()
    return result
