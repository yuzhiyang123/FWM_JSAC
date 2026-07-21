from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor, nn
from torch.utils.data import DataLoader

from twm.datasets import build_time_band_channel_three_way_dataloaders
from .pipeline import TimeBandJEPAConfig, TimeBandJEPAOutput, TimeBandJEPAPipeline, TimeBandViTConfig
from .time_freq_band import (
    CurriculumRandomMaskExceptLastGenerator,
    RandomMaskGenerator,
    RandomMaskExceptLastGenerator,
    RandomMaskExceptLastVariableRateGenerator,
)
from .default_pipeline_configs import build_lazy_config


@dataclass
class NamedJEPAConfig:
    preset_name: str = '32x16_medium_deep_medium'
    num_low_bands: int = 5
    num_tokens: int = 8
    token_dim: int = 128
    predictor_dim: int = 256
    tokenizer_output_norm: str = 'layernorm'

    def build(self) -> TimeBandJEPAConfig:
        return build_lazy_config(
            self.preset_name,
            num_low_bands=self.num_low_bands,
            num_tokens=self.num_tokens,
            token_dim=self.token_dim,
            predictor_dim=self.predictor_dim,
            tokenizer_output_norm=self.tokenizer_output_norm,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BasicJEPATrainingConfig:
    jepa_config: NamedJEPAConfig = field(default_factory=NamedJEPAConfig)
    batch_size: int = 8
    lr: float = 1e-4


@dataclass
class JEPATrainingConfig(BasicJEPATrainingConfig):
    dataset_file: list[str] = field(default_factory=list)
    working_dir: str = ''
    epochs: int = 200
    num_workers: int = 2
    window_length: int = 16
    random_windows_per_sequence: int = 30
    test_ratio: float = 0.2
    train_paths: int = 512
    test_paths: int = 88
    seed: int = 42
    device: str = 'cuda'
    weight_decay: float = 1e-4
    teacher_momentum: float = 0.999
    masked_target_source: str = 'teacher_tokenizer'
    tokenizer_output_norm: str = 'layernorm'
    mask_style: str = 'curriculum_random_except_last'
    mask_rate: float = 0.5
    curriculum_start_mask_rate: float = 0.3
    curriculum_end_mask_rate: float = 0.9
    curriculum_ramp_epochs: int = 50
    curriculum_sample_jitter: float = 0.1
    final_min_mask_rate: float = 0.1
    final_max_mask_rate: float = 0.9
    final_beta_alpha: float = 4.0
    final_beta_beta: float = 1.2
    jepa_loss_weight: float = 1.0
    remained_nt_vicreg_weight: float = 0.0
    remained_nt_vicreg_floor: float = 1.0
    remained_cov_fro_weight: float = 0.0
    alignment_loss_weight: float = 1.0
    recovery_loss_weight: float = 0.0
    amplitude_recovery_loss_weight: float = 0.0
    diffusion_loss_weight: float = 0.0
    save_every: int = 20
    stage2_epochs: int = 0
    stage2_freeze_tokenizers: int = 0
    stage2_jepa_only: int = 0
    wandb_project: str = 'twm_jepa_main'
    wandb_run_name: str = ''
    estimate_main_task: int = 0
    estimate_steps: int = 3
    estimate_warmup_steps: int = 1

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> 'JEPATrainingConfig':
        payload = vars(args).copy()
        dataset_file = payload.get('dataset_file', [])
        if isinstance(dataset_file, (str, Path)):
            payload['dataset_file'] = [str(dataset_file)]
        else:
            payload['dataset_file'] = [str(item) for item in dataset_file]
        payload['working_dir'] = str(payload.get('working_dir', ''))
        jepa_config = NamedJEPAConfig(
            preset_name=str(payload.pop('jepa_config_name')),
            num_low_bands=int(payload.pop('num_low_bands')),
            num_tokens=int(payload.pop('num_tokens')),
            token_dim=int(payload.pop('token_dim')),
            predictor_dim=int(payload.pop('predictor_dim')),
        )
        payload['jepa_config'] = jepa_config
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload['dataset_file'] = list(self.dataset_file)
        payload['working_dir'] = self.working_dir
        return payload

    def to_cli_args(self) -> list[str]:
        args: list[str] = [
            '--dataset-file', *self.dataset_file,
            '--working-dir', self.working_dir,
            '--jepa-config-name', self.jepa_config.preset_name,
            '--num-low-bands', str(self.jepa_config.num_low_bands),
            '--num-tokens', str(self.jepa_config.num_tokens),
            '--token-dim', str(self.jepa_config.token_dim),
            '--predictor-dim', str(self.jepa_config.predictor_dim),
            '--batch-size', str(self.batch_size),
            '--lr', str(self.lr),
        ]
        scalar_map = {
            'epochs': self.epochs,
            'num_workers': self.num_workers,
            'window_length': self.window_length,
            'random_windows_per_sequence': self.random_windows_per_sequence,
            'test_ratio': self.test_ratio,
            'train_paths': self.train_paths,
            'test_paths': self.test_paths,
            'seed': self.seed,
            'device': self.device,
            'weight_decay': self.weight_decay,
            'teacher_momentum': self.teacher_momentum,
            'masked_target_source': self.masked_target_source,
            'tokenizer_output_norm': self.tokenizer_output_norm,
            'mask_style': self.mask_style,
            'mask_rate': self.mask_rate,
            'curriculum_start_mask_rate': self.curriculum_start_mask_rate,
            'curriculum_end_mask_rate': self.curriculum_end_mask_rate,
            'curriculum_ramp_epochs': self.curriculum_ramp_epochs,
            'curriculum_sample_jitter': self.curriculum_sample_jitter,
            'final_min_mask_rate': self.final_min_mask_rate,
            'final_max_mask_rate': self.final_max_mask_rate,
            'final_beta_alpha': self.final_beta_alpha,
            'final_beta_beta': self.final_beta_beta,
            'jepa_loss_weight': self.jepa_loss_weight,
            'remained_nt_vicreg_weight': self.remained_nt_vicreg_weight,
            'remained_nt_vicreg_floor': self.remained_nt_vicreg_floor,
            'remained_cov_fro_weight': self.remained_cov_fro_weight,
            'alignment_loss_weight': self.alignment_loss_weight,
            'recovery_loss_weight': self.recovery_loss_weight,
            'amplitude_recovery_loss_weight': self.amplitude_recovery_loss_weight,
            'diffusion_loss_weight': self.diffusion_loss_weight,
            'save_every': self.save_every,
            'stage2_epochs': self.stage2_epochs,
            'stage2_freeze_tokenizers': self.stage2_freeze_tokenizers,
            'stage2_jepa_only': self.stage2_jepa_only,
            'wandb_project': self.wandb_project,
            'wandb_run_name': self.wandb_run_name,
            'estimate_main_task': self.estimate_main_task,
            'estimate_steps': self.estimate_steps,
            'estimate_warmup_steps': self.estimate_warmup_steps,
        }
        for key, value in scalar_map.items():
            flag = '--' + key.replace('_', '-')
            args.extend([flag, str(value)])
        return args


@dataclass
class JEPAStage2TrainingConfig(BasicJEPATrainingConfig):
    dataset_file: list[str] = field(default_factory=list)
    pretrained_dir: str = ''
    working_dir: str = ''
    epochs: int = 50
    num_workers: int = 0
    window_length: int = 16
    random_windows_per_sequence: int = 30
    test_ratio: float = 0.2
    train_paths: int = 512
    test_paths: int = 88
    seed: int = 42
    device: str = 'cuda'
    weight_decay: float = 1e-4
    start_mask_rate: float = 0.3
    final_min_mask_rate: float = 0.1
    final_max_mask_rate: float = 0.9
    curriculum_ramp_epochs: int = 50
    curriculum_sample_jitter: float = 0.1
    final_beta_alpha: float = 4.0
    final_beta_beta: float = 1.2
    jepa_loss_weight: float = 1.0
    remained_m_vicreg_weight: float = 0.0
    remained_m_vicreg_floor: float = 1.0
    remained_cov_fro_weight: float = 0.0
    save_every: int = 1
    wandb_project: str = 'twm_jepa_stage2_main'
    wandb_run_name: str = ''

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> 'JEPAStage2TrainingConfig':
        payload = vars(args).copy()
        dataset_file = payload.get('dataset_file', [])
        if isinstance(dataset_file, (str, Path)):
            payload['dataset_file'] = [str(dataset_file)]
        else:
            payload['dataset_file'] = [str(item) for item in dataset_file]
        payload['pretrained_dir'] = str(payload.get('pretrained_dir', ''))
        payload['working_dir'] = str(payload.get('working_dir', ''))
        jepa_config = NamedJEPAConfig(
            preset_name=str(payload.pop('jepa_config_name')),
            num_low_bands=int(payload.pop('num_low_bands')),
            num_tokens=int(payload.pop('num_tokens')),
            token_dim=int(payload.pop('token_dim')),
            predictor_dim=int(payload.pop('predictor_dim')),
        )
        payload['jepa_config'] = jepa_config
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload['dataset_file'] = list(self.dataset_file)
        payload['pretrained_dir'] = self.pretrained_dir
        payload['working_dir'] = self.working_dir
        return payload

    def to_cli_args(self) -> list[str]:
        args: list[str] = [
            '--dataset-file', *self.dataset_file,
            '--pretrained-dir', self.pretrained_dir,
            '--working-dir', self.working_dir,
            '--jepa-config-name', self.jepa_config.preset_name,
            '--num-low-bands', str(self.jepa_config.num_low_bands),
            '--num-tokens', str(self.jepa_config.num_tokens),
            '--token-dim', str(self.jepa_config.token_dim),
            '--predictor-dim', str(self.jepa_config.predictor_dim),
            '--batch-size', str(self.batch_size),
            '--lr', str(self.lr),
        ]
        scalar_map = {
            'epochs': self.epochs,
            'num_workers': self.num_workers,
            'window_length': self.window_length,
            'random_windows_per_sequence': self.random_windows_per_sequence,
            'test_ratio': self.test_ratio,
            'train_paths': self.train_paths,
            'test_paths': self.test_paths,
            'seed': self.seed,
            'device': self.device,
            'weight_decay': self.weight_decay,
            'start_mask_rate': self.start_mask_rate,
            'final_min_mask_rate': self.final_min_mask_rate,
            'final_max_mask_rate': self.final_max_mask_rate,
            'curriculum_ramp_epochs': self.curriculum_ramp_epochs,
            'curriculum_sample_jitter': self.curriculum_sample_jitter,
            'final_beta_alpha': self.final_beta_alpha,
            'final_beta_beta': self.final_beta_beta,
            'jepa_loss_weight': self.jepa_loss_weight,
            'remained_m_vicreg_weight': self.remained_m_vicreg_weight,
            'remained_m_vicreg_floor': self.remained_m_vicreg_floor,
            'remained_cov_fro_weight': self.remained_cov_fro_weight,
            'save_every': self.save_every,
            'stage2_epochs': self.stage2_epochs,
            'stage2_freeze_tokenizers': self.stage2_freeze_tokenizers,
            'stage2_jepa_only': self.stage2_jepa_only,
            'wandb_project': self.wandb_project,
            'wandb_run_name': self.wandb_run_name,
        }
        for key, value in scalar_map.items():
            flag = '--' + key.replace('_', '-')
            args.extend([flag, str(value)])
        return args


@dataclass
class JEPALossOutput:
    total_loss: Tensor
    components: dict[str, Tensor]
    weights: dict[str, float]


class AbstractJEPALossComponent(nn.Module, ABC):
    name: str

    @abstractmethod
    def forward(self, output: TimeBandJEPAOutput) -> Tensor:
        raise NotImplementedError


class StandardJEPALossComponent(AbstractJEPALossComponent):
    name = 'jepa_l1'

    def forward(self, output: TimeBandJEPAOutput) -> Tensor:
        if output.latent_masked.shape != output.latent_predicted.shape:
            raise ValueError(
                'latent_masked and latent_predicted must have the same shape, '
                f'got {tuple(output.latent_masked.shape)} and {tuple(output.latent_predicted.shape)}'
            )
        valid_mask = output.latent_masked_mask
        if valid_mask is None:
            if output.latent_masked.numel() == 0:
                return output.latent_predicted.sum() * 0.0
            return F.l1_loss(output.latent_predicted, output.latent_masked)

        expanded_mask = valid_mask[:, :, None, None]
        if not expanded_mask.any():
            return output.latent_predicted.sum() * 0.0
        diff = (output.latent_predicted - output.latent_masked).abs()
        return diff.masked_select(expanded_mask).mean()


def _masked_aux_zero(output: TimeBandJEPAOutput) -> Tensor:
    return output.latent_predicted.sum() * 0.0


def _masked_aux_squared_error_means(
    output: TimeBandJEPAOutput,
    *,
    pred_attr: str,
    target_attr: str,
) -> Tensor:
    values: list[Tensor] = []
    for aux in [output.masked_aux_3p5, output.masked_aux_28]:
        if aux is None or aux.valid_mask is None:
            continue
        prediction = getattr(aux, pred_attr)
        target = getattr(aux, target_attr)
        if prediction is None or target is None or prediction.numel() == 0:
            continue
        expanded_mask = aux.valid_mask[(...,) + (None,) * (prediction.ndim - 2)]
        selected = (prediction - target).pow(2).masked_select(expanded_mask)
        if selected.numel() > 0:
            values.append(selected)
    if not values:
        return _masked_aux_zero(output)
    return torch.cat([value.reshape(-1) for value in values]).mean()


class MaskedRecoveryLossComponent(AbstractJEPALossComponent):
    name = 'masked_recovery_mse'

    def forward(self, output: TimeBandJEPAOutput) -> Tensor:
        return _masked_aux_squared_error_means(output, pred_attr='recovered', target_attr='target')


class MaskedDiffusionNoiseLossComponent(AbstractJEPALossComponent):
    name = 'masked_diffusion_noise_mse'

    def forward(self, output: TimeBandJEPAOutput) -> Tensor:
        return _masked_aux_squared_error_means(
            output,
            pred_attr='diffusion_predicted_noise',
            target_attr='diffusion_target_noise',
        )


class MaskedAmplitudeRecoveryLossComponent(AbstractJEPALossComponent):
    name = 'masked_amplitude_recovery_mse'

    def forward(self, output: TimeBandJEPAOutput) -> Tensor:
        values: list[Tensor] = []
        for aux in [output.masked_aux_3p5, output.masked_aux_28]:
            if aux is None or aux.valid_mask is None or aux.recovered is None or aux.target is None:
                continue
            pred_amp = torch.linalg.vector_norm(aux.recovered, ord=2, dim=-3, keepdim=True)
            target_amp = torch.linalg.vector_norm(aux.target, ord=2, dim=-3, keepdim=True)
            expanded_mask = aux.valid_mask[(...,) + (None,) * (pred_amp.ndim - 2)]
            selected = (pred_amp - target_amp).pow(2).masked_select(expanded_mask)
            if selected.numel() > 0:
                values.append(selected)
        if not values:
            return _masked_aux_zero(output)
        return torch.cat([value.reshape(-1) for value in values]).mean()


class VICRegNtVarianceLossComponent(AbstractJEPALossComponent):
    name = 'latent_remained_nt_var'

    def __init__(self, variance_floor: float = 1.0, eps: float = 1e-4) -> None:
        super().__init__()
        self.variance_floor = float(variance_floor)
        self.eps = float(eps)

    def forward(self, output: TimeBandJEPAOutput) -> Tensor:
        latent_remained = output.latent_all if output.latent_all is not None else output.latent_remained
        if latent_remained.ndim != 4:
            raise ValueError(
                'latent_remained must have shape [B, M, Nt, Lt], '
                f'got {tuple(latent_remained.shape)}'
            )
        if latent_remained.numel() == 0 or latent_remained.shape[2] <= 1:
            return latent_remained.sum() * 0.0

        variance = latent_remained.var(dim=2, unbiased=False)
        std = torch.sqrt(variance + self.eps)
        penalty = torch.relu(self.variance_floor - std)
        valid_mask = output.latent_all_mask if output.latent_all is not None else output.latent_remained_mask
        if valid_mask is None:
            return penalty.mean()
        expanded_mask = valid_mask[:, :, None]
        if not expanded_mask.any():
            return latent_remained.sum() * 0.0
        return penalty.masked_select(expanded_mask).mean()


class MAxisVarianceLossComponent(AbstractJEPALossComponent):
    name = 'latent_remained_m_var'

    def __init__(self, variance_floor: float = 1.0, eps: float = 1e-4) -> None:
        super().__init__()
        self.variance_floor = float(variance_floor)
        self.eps = float(eps)

    def forward(self, output: TimeBandJEPAOutput) -> Tensor:
        latent_remained = output.latent_all if output.latent_all is not None else output.latent_remained
        if latent_remained.ndim != 4:
            raise ValueError(
                'latent_remained must have shape [B, M, Nt, Lt], '
                f'got {tuple(latent_remained.shape)}'
            )
        if latent_remained.numel() == 0 or latent_remained.shape[1] <= 1:
            return latent_remained.sum() * 0.0

        flattened = latent_remained.flatten(start_dim=2)
        valid_mask = output.latent_all_mask if output.latent_all is not None else output.latent_remained_mask
        if valid_mask is None:
            valid_mask = torch.ones(flattened.shape[:2], device=flattened.device, dtype=torch.bool)

        slot_losses: list[Tensor] = []
        for slot_idx in range(flattened.shape[1]):
            slot_values = flattened[:, slot_idx, :]
            slot_mask = valid_mask[:, slot_idx]
            valid_slot_values = slot_values[slot_mask]
            if valid_slot_values.shape[0] <= 1:
                continue
            variance = valid_slot_values.var(dim=0, unbiased=False)
            std = torch.sqrt(variance + self.eps)
            slot_losses.append(torch.relu(self.variance_floor - std).mean())

        if not slot_losses:
            return latent_remained.sum() * 0.0
        return torch.stack(slot_losses).mean()


class FlattenedOffDiagonalCovarianceFrobeniusLossComponent(AbstractJEPALossComponent):
    name = 'latent_remained_offdiag_cov_fro'

    def forward(self, output: TimeBandJEPAOutput) -> Tensor:
        latent_remained = output.latent_all if output.latent_all is not None else output.latent_remained
        if latent_remained.ndim != 4:
            raise ValueError(
                'latent_remained must have shape [B, M, Nt, Lt], '
                f'got {tuple(latent_remained.shape)}'
            )
        if latent_remained.numel() == 0:
            return latent_remained.sum() * 0.0

        flattened = latent_remained.flatten(start_dim=2)
        valid_mask = output.latent_all_mask if output.latent_all is not None else output.latent_remained_mask
        if valid_mask is None:
            valid_mask = torch.ones(flattened.shape[:2], device=flattened.device, dtype=torch.bool)

        slot_losses: list[Tensor] = []
        for slot_idx in range(flattened.shape[1]):
            slot_values = flattened[:, slot_idx, :]
            slot_mask = valid_mask[:, slot_idx]
            valid_slot_values = slot_values[slot_mask]
            if valid_slot_values.shape[0] <= 1:
                continue
            centered = valid_slot_values - valid_slot_values.mean(dim=0, keepdim=True)
            covariance = centered @ centered.transpose(0, 1) / float(valid_slot_values.shape[0] - 1)
            offdiag_mask = ~torch.eye(covariance.shape[0], device=covariance.device, dtype=torch.bool)
            offdiag_values = covariance.masked_select(offdiag_mask)
            if offdiag_values.numel() == 0:
                continue
            # Use mean squared off-diagonal covariance so the penalty scale does not grow
            # with the number of valid batch samples and gradients stay linear in the squared term.
            slot_losses.append(offdiag_values.square().mean())

        if not slot_losses:
            return latent_remained.sum() * 0.0
        return torch.stack(slot_losses).mean()


class AbstractJEPALoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, output: TimeBandJEPAOutput) -> JEPALossOutput:
        raise NotImplementedError


class WeightedJEPALoss(AbstractJEPALoss):
    def __init__(self, components: list[tuple[AbstractJEPALossComponent, float]]) -> None:
        super().__init__()
        if not components:
            raise ValueError('WeightedJEPALoss requires at least one loss component')
        self.components = nn.ModuleList([component for component, _ in components])
        self.component_weights = {component.name: float(weight) for component, weight in components}

    def forward(self, output: TimeBandJEPAOutput) -> JEPALossOutput:
        component_values: dict[str, Tensor] = {}
        total_loss: Tensor | None = None
        for component in self.components:
            value = component(output)
            component_values[component.name] = value
            weighted = value * self.component_weights[component.name]
            total_loss = weighted if total_loss is None else total_loss + weighted
        if total_loss is None:
            raise RuntimeError('No loss components were evaluated')
        return JEPALossOutput(
            total_loss=total_loss,
            components=component_values,
            weights=dict(self.component_weights),
        )


@dataclass
class WarmupCosineScheduleConfig:
    total_steps: int
    warmup_ratio: float = 0.1

    @property
    def warmup_steps(self) -> int:
        if self.total_steps <= 0:
            return 0
        return min(self.total_steps, max(1, int(round(self.total_steps * self.warmup_ratio))))


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    schedule = WarmupCosineScheduleConfig(total_steps=int(total_steps), warmup_ratio=float(warmup_ratio))
    if schedule.total_steps <= 0:
        raise ValueError(f'total_steps must be positive, got {schedule.total_steps}')

    warmup_steps = schedule.warmup_steps

    def lr_lambda(current_step: int) -> float:
        step = min(max(int(current_step), 0), schedule.total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if schedule.total_steps == warmup_steps:
            return 0.0
        progress = float(step - warmup_steps) / float(schedule.total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate_jepa_batch(
    model: TimeBandJEPAPipeline,
    batch_x: Tensor,
    loss_fn: AbstractJEPALoss,
    *,
    mask: Tensor | None = None,
) -> tuple[TimeBandJEPAOutput, JEPALossOutput]:
    was_training = model.training
    model.eval()
    output = model(batch_x, mask=mask)
    loss_output = loss_fn(output)
    if was_training:
        model.train()
    return output, loss_output


def train_jepa_batch(
    model: TimeBandJEPAPipeline,
    batch_x: Tensor,
    loss_fn: AbstractJEPALoss,
    optimizer: torch.optim.Optimizer,
    *,
    accelerator: Accelerator,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    mask: Tensor | None = None,
    teacher_momentum: float | None = None,
) -> tuple[TimeBandJEPAOutput, JEPALossOutput]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch_x, mask=mask)
    loss_output = loss_fn(output)
    accelerator.backward(loss_output.total_loss)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if teacher_momentum is not None:
        model.synchronize_teacher(momentum=teacher_momentum)
    return output, loss_output


def make_mask_generator(args: argparse.Namespace):
    if args.mask_style == 'random':
        return RandomMaskGenerator(mask_rate=args.mask_rate)
    if args.mask_style == 'curriculum_random_except_last':
        return CurriculumRandomMaskExceptLastGenerator(
            start_mask_rate=args.curriculum_start_mask_rate,
            end_mask_rate=args.curriculum_end_mask_rate,
            ramp_epochs=args.curriculum_ramp_epochs,
            sample_jitter=args.curriculum_sample_jitter,
            final_min_mask_rate=getattr(args, 'final_min_mask_rate', 0.1),
            final_max_mask_rate=getattr(args, 'final_max_mask_rate', 0.9),
            final_beta_alpha=getattr(args, 'final_beta_alpha', 4.0),
            final_beta_beta=getattr(args, 'final_beta_beta', 1.2),
        )
    return RandomMaskExceptLastGenerator(mask_rate=args.mask_rate)


def _mean_metric(accelerator: Accelerator, values: list[Tensor]) -> float:
    if not values:
        return 0.0
    stacked = torch.stack(values)
    gathered = accelerator.gather_for_metrics(stacked)
    return float(gathered.mean().item())


def _mean_offdiag_cosine(vectors: Tensor) -> Tensor:
    if vectors.shape[0] <= 1:
        return vectors.new_tensor(float('nan'))
    normalized = F.normalize(vectors.float(), dim=-1, eps=1e-8)
    similarity = normalized @ normalized.transpose(0, 1)
    mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    values = similarity.masked_select(mask)
    if values.numel() == 0:
        return vectors.new_tensor(float('nan'))
    return values.mean()


def _collapse_metrics(output: TimeBandJEPAOutput) -> dict[str, Tensor]:
    remained = output.latent_remained.flatten(start_dim=2)
    remained_mask = output.latent_remained_mask

    if remained_mask is None:
        remained_mask = torch.ones(remained.shape[:2], dtype=torch.bool, device=remained.device)

    values: list[Tensor] = []
    for sample, sample_mask in zip(remained, remained_mask):
        valid = sample[sample_mask]
        metric = _mean_offdiag_cosine(valid)
        if not torch.isnan(metric):
            values.append(metric)

    return {
        'remained_pairwise_cosine': torch.stack(values).mean() if values else remained.new_tensor(float('nan'))
    }


def _jepa_nmse(output: TimeBandJEPAOutput, eps: float = 1e-8) -> Tensor:
    if output.latent_masked.shape != output.latent_predicted.shape:
        raise ValueError(
            'latent_masked and latent_predicted must have the same shape, '
            f'got {tuple(output.latent_masked.shape)} and {tuple(output.latent_predicted.shape)}'
        )
    valid_mask = output.latent_masked_mask
    diff_sq = (output.latent_predicted - output.latent_masked).pow(2)
    target_sq = output.latent_masked.pow(2)
    if valid_mask is None:
        if diff_sq.numel() == 0:
            return diff_sq.sum() * 0.0
        return diff_sq.mean() / target_sq.mean().clamp_min(eps)

    expanded_mask = valid_mask[:, :, None, None]
    if not expanded_mask.any():
        return diff_sq.sum() * 0.0
    numerator = diff_sq.masked_select(expanded_mask).mean()
    denominator = target_sq.masked_select(expanded_mask).mean().clamp_min(eps)
    return numerator / denominator


def _prediction_metrics(output: TimeBandJEPAOutput) -> dict[str, Tensor]:
    metrics = _collapse_metrics(output)
    metrics['jepa_nmse'] = _jepa_nmse(output)
    return metrics


def _estimate_main_task_profile(
    model: TimeBandJEPAPipeline,
    train_loader: DataLoader,
    loss_fn: AbstractJEPALoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    accelerator: Accelerator,
    *,
    teacher_momentum: float | None,
    total_steps: int,
    measure_steps: int,
    warmup_steps: int,
) -> dict[str, float]:
    if measure_steps <= 0:
        return {}
    unwrapped_model = accelerator.unwrap_model(model)
    model_state = {k: v.detach().cpu().clone() for k, v in unwrapped_model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None

    data_iter = iter(train_loader)
    for _ in range(max(0, int(warmup_steps))):
        try:
            batch_x = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch_x = next(data_iter)
        batch_x = batch_x.to(dtype=torch.float32)
        train_jepa_batch(
            model,
            batch_x,
            loss_fn,
            optimizer,
            accelerator=accelerator,
            scheduler=scheduler,
            teacher_momentum=teacher_momentum,
        )

    cuda_enabled = accelerator.device.type == 'cuda' and torch.cuda.is_available()
    if cuda_enabled:
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        torch.cuda.synchronize(accelerator.device)

    step_times: list[float] = []
    for _ in range(int(measure_steps)):
        try:
            batch_x = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch_x = next(data_iter)
        batch_x = batch_x.to(dtype=torch.float32)
        if cuda_enabled:
            torch.cuda.synchronize(accelerator.device)
        t0 = time.perf_counter()
        train_jepa_batch(
            model,
            batch_x,
            loss_fn,
            optimizer,
            accelerator=accelerator,
            scheduler=scheduler,
            teacher_momentum=teacher_momentum,
        )
        if cuda_enabled:
            torch.cuda.synchronize(accelerator.device)
        step_times.append(time.perf_counter() - t0)

    unwrapped_model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    avg_step_time = float(sum(step_times) / max(1, len(step_times)))
    estimated_total_hours = float(avg_step_time * float(total_steps) / 3600.0)
    estimated_epoch_minutes = float(avg_step_time * float(len(train_loader)) / 60.0)
    metrics = {
        'estimate/main_task_step_seconds': avg_step_time,
        'estimate/main_task_epoch_minutes': estimated_epoch_minutes,
        'estimate/main_task_total_hours': estimated_total_hours,
    }
    if cuda_enabled:
        metrics['estimate/main_task_peak_allocated_gb'] = float(torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3))
        metrics['estimate/main_task_peak_reserved_gb'] = float(torch.cuda.max_memory_reserved(accelerator.device) / (1024 ** 3))
    return metrics


def run_jepa_training(args: argparse.Namespace | JEPATrainingConfig) -> dict[str, object]:
    config = args if isinstance(args, JEPATrainingConfig) else JEPATrainingConfig.from_namespace(args)
    working_dir = Path(config.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    accelerator = Accelerator(log_with='wandb')

    train_loader, test_loader, cross_test_loader = build_time_band_channel_three_way_dataloaders(
        dataset_file=config.dataset_file,
        batch_size=config.batch_size,
        window_length=config.window_length,
        random_windows_per_sequence=config.random_windows_per_sequence,
        num_workers=config.num_workers,
        seed=config.seed,
        train_paths=config.train_paths,
        test_paths=config.test_paths,
    )

    mask_generator = make_mask_generator(argparse.Namespace(
        mask_style=config.mask_style,
        mask_rate=config.mask_rate,
        curriculum_start_mask_rate=config.curriculum_start_mask_rate,
        curriculum_end_mask_rate=config.curriculum_end_mask_rate,
        curriculum_ramp_epochs=config.curriculum_ramp_epochs,
        curriculum_sample_jitter=config.curriculum_sample_jitter,
        final_min_mask_rate=config.final_min_mask_rate,
        final_max_mask_rate=config.final_max_mask_rate,
        final_beta_alpha=config.final_beta_alpha,
        final_beta_beta=config.final_beta_beta,
    ))
    jepa_config = config.jepa_config.build()
    jepa_config.enable_recovery_head = bool((config.recovery_loss_weight > 0.0) or (config.amplitude_recovery_loss_weight > 0.0))
    jepa_config.enable_cond_diffusion_head = bool(config.diffusion_loss_weight > 0.0)
    jepa_config.masked_target_source = str(config.masked_target_source)
    jepa_config.tokenizer_output_norm = str(config.tokenizer_output_norm)

    model = TimeBandJEPAPipeline.from_config(mask_generator=mask_generator, config=jepa_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    stage1_total_steps = max(1, config.epochs * len(train_loader))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=stage1_total_steps, warmup_ratio=0.1)
    loss_components: list[tuple[AbstractJEPALossComponent, float]] = [
        (StandardJEPALossComponent(), config.jepa_loss_weight),
        (
            MAxisVarianceLossComponent(variance_floor=config.remained_nt_vicreg_floor),
            config.remained_nt_vicreg_weight,
        ),
        (
            FlattenedOffDiagonalCovarianceFrobeniusLossComponent(),
            config.remained_cov_fro_weight,
        ),
    ]
    if config.recovery_loss_weight != 0.0:
        loss_components.append((MaskedRecoveryLossComponent(), config.recovery_loss_weight))
    if config.amplitude_recovery_loss_weight != 0.0:
        loss_components.append((MaskedAmplitudeRecoveryLossComponent(), config.amplitude_recovery_loss_weight))
    if config.diffusion_loss_weight != 0.0:
        loss_components.append((MaskedDiffusionNoiseLossComponent(), config.diffusion_loss_weight))
    loss_fn = WeightedJEPALoss(loss_components)

    model, optimizer, train_loader, test_loader, cross_test_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        test_loader,
        cross_test_loader,
    )

    config_payload = config.to_dict()
    config_payload['stage1_total_steps'] = stage1_total_steps
    config_payload['stage2_total_steps'] = max(1, config.stage2_epochs * len(train_loader)) if config.stage2_epochs > 0 else 0
    config_payload['world_size'] = accelerator.num_processes
    wandb_init_kwargs = {
        'name': config.wandb_run_name or working_dir.name,
        'dir': str(working_dir),
    }
    wandb_entity = os.environ.get('WANDB_ENTITY', '').strip()
    if wandb_entity:
        wandb_init_kwargs['entity'] = wandb_entity
    accelerator.init_trackers(
        project_name=config.wandb_project,
        config=config_payload,
        init_kwargs={'wandb': wandb_init_kwargs},
    )

    history: list[dict[str, float]] = []
    best_test_loss: float | None = None
    best_epoch: int | None = None

    estimate_metrics: dict[str, float] = {}
    if int(config.estimate_main_task) != 0:
        estimate_metrics = _estimate_main_task_profile(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            accelerator,
            teacher_momentum=config.teacher_momentum,
            total_steps=stage1_total_steps,
            measure_steps=int(config.estimate_steps),
            warmup_steps=int(config.estimate_warmup_steps),
        )
        if estimate_metrics:
            accelerator.log(estimate_metrics, step=0)
            accelerator.print(json.dumps(estimate_metrics))

    if accelerator.is_main_process:
        (working_dir / 'run_config.json').write_text(json.dumps(config_payload, indent=2) + '\n', encoding='utf-8')

    def run_phase(
        *,
        phase_name: str,
        num_epochs: int,
        phase_loss_fn: AbstractJEPALoss,
        phase_optimizer: torch.optim.Optimizer,
        phase_scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        teacher_momentum: float | None,
    ) -> None:
        nonlocal best_test_loss, best_epoch
        if num_epochs <= 0:
            return
        for phase_epoch in range(1, num_epochs + 1):
            global_epoch = len(history) + 1
            if hasattr(mask_generator, 'set_epoch'):
                mask_generator.set_epoch(phase_epoch, num_epochs)
            train_total_values: list[Tensor] = []
            train_component_values: dict[str, list[Tensor]] = {}
            train_aux_metrics: dict[str, list[Tensor]] = {}
            for batch_x in train_loader:
                batch_x = batch_x.to(dtype=torch.float32)
                output, loss_output = train_jepa_batch(
                    model,
                    batch_x,
                    phase_loss_fn,
                    phase_optimizer,
                    accelerator=accelerator,
                    scheduler=phase_scheduler,
                    teacher_momentum=teacher_momentum,
                )
                train_total_values.append(loss_output.total_loss.detach())
                for name, value in loss_output.components.items():
                    train_component_values.setdefault(name, []).append(value.detach())
                for name, value in _prediction_metrics(output).items():
                    if not torch.isnan(value):
                        train_aux_metrics.setdefault(name, []).append(value.detach())

            epoch_metrics = {
                'epoch': float(global_epoch),
                'phase_epoch': float(phase_epoch),
                'train/total_loss': _mean_metric(accelerator, train_total_values),
                'train/lr': float(phase_optimizer.param_groups[0]['lr']),
                'phase/stage1': 1.0 if phase_name == 'stage1' else 0.0,
                'phase/stage2': 1.0 if phase_name == 'stage2' else 0.0,
            }
            for name, values in train_component_values.items():
                epoch_metrics[f'train/{name}'] = _mean_metric(accelerator, values)
            for name, values in train_aux_metrics.items():
                epoch_metrics[f'train/{name}'] = _mean_metric(accelerator, values)

            if phase_epoch % 5 == 0:
                for split_name, split_loader in [('test', test_loader)]:
                    test_total_values: list[Tensor] = []
                    test_component_values: dict[str, list[Tensor]] = {}
                    test_aux_metrics: dict[str, list[Tensor]] = {}
                    for batch_x in split_loader:
                        batch_x = batch_x.to(dtype=torch.float32)
                        output, loss_output = evaluate_jepa_batch(
                            model,
                            batch_x,
                            phase_loss_fn,
                        )
                        test_total_values.append(loss_output.total_loss.detach())
                        for name, value in loss_output.components.items():
                            test_component_values.setdefault(name, []).append(value.detach())
                        for name, value in _prediction_metrics(output).items():
                            if not torch.isnan(value):
                                test_aux_metrics.setdefault(name, []).append(value.detach())
                    epoch_metrics[f'{split_name}/total_loss'] = _mean_metric(accelerator, test_total_values)
                    for name, values in test_component_values.items():
                        epoch_metrics[f'{split_name}/{name}'] = _mean_metric(accelerator, values)
                    for name, values in test_aux_metrics.items():
                        epoch_metrics[f'{split_name}/{name}'] = _mean_metric(accelerator, values)

            for name, weight in phase_loss_fn.component_weights.items():
                epoch_metrics[f'loss_weight/{name}'] = float(weight)

            history.append(epoch_metrics)
            accelerator.log(epoch_metrics, step=global_epoch)

            current_test_loss = epoch_metrics.get('test/total_loss')
            unwrapped_model = accelerator.unwrap_model(model)
            if current_test_loss is not None and (best_test_loss is None or current_test_loss < best_test_loss):
                best_test_loss = current_test_loss
                best_epoch = global_epoch
                if accelerator.is_main_process:
                    unwrapped_model.save(working_dir / 'best_model')

            if config.save_every > 0 and phase_epoch % config.save_every == 0 and accelerator.is_main_process:
                unwrapped_model.save(working_dir / f'checkpoint_{phase_name}_epoch_{phase_epoch:03d}')

    run_phase(
        phase_name='stage1',
        num_epochs=config.epochs,
        phase_loss_fn=loss_fn,
        phase_optimizer=optimizer,
        phase_scheduler=scheduler,
        teacher_momentum=config.teacher_momentum,
    )

    if config.stage2_epochs > 0:
        unwrapped_model = accelerator.unwrap_model(model)
        if int(config.stage2_freeze_tokenizers) != 0:
            unwrapped_model.freeze_tokenizers()
        stage2_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not stage2_parameters:
            raise RuntimeError('Stage-2 training found no trainable parameters after freezing tokenizers')
        stage2_optimizer = torch.optim.AdamW(stage2_parameters, lr=config.lr, weight_decay=config.weight_decay)
        stage2_scheduler = build_warmup_cosine_scheduler(
            stage2_optimizer,
            total_steps=max(1, config.stage2_epochs * len(train_loader)),
            warmup_ratio=0.1,
        )
        stage2_loss_fn = WeightedJEPALoss([(StandardJEPALossComponent(), config.jepa_loss_weight)]) if int(config.stage2_jepa_only) != 0 else loss_fn
        if accelerator.is_main_process:
            unwrapped_model.save(working_dir / 'stage1_last_model')
        run_phase(
            phase_name='stage2',
            num_epochs=config.stage2_epochs,
            phase_loss_fn=stage2_loss_fn,
            phase_optimizer=stage2_optimizer,
            phase_scheduler=stage2_scheduler,
            teacher_momentum=None,
        )

    summary = {
        'best_epoch': best_epoch,
        'best_test_loss': best_test_loss,
        'estimate_metrics': estimate_metrics,
        'history': history,
    }
    if accelerator.is_main_process:
        (working_dir / 'results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        accelerator.unwrap_model(model).save(working_dir / 'last_model')
    accelerator.end_training()
    return summary



def run_jepa_stage2_training(args: argparse.Namespace | JEPAStage2TrainingConfig) -> dict[str, object]:
    config = args if isinstance(args, JEPAStage2TrainingConfig) else JEPAStage2TrainingConfig.from_namespace(args)
    working_dir = Path(config.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    accelerator = Accelerator(log_with='wandb')

    train_loader, test_loader, cross_test_loader = build_time_band_channel_three_way_dataloaders(
        dataset_file=config.dataset_file,
        batch_size=config.batch_size,
        window_length=config.window_length,
        random_windows_per_sequence=config.random_windows_per_sequence,
        num_workers=config.num_workers,
        seed=config.seed,
        train_paths=config.train_paths,
        test_paths=config.test_paths,
    )

    mask_generator = CurriculumRandomMaskExceptLastGenerator(
        start_mask_rate=config.start_mask_rate,
        end_mask_rate=config.final_max_mask_rate,
        ramp_epochs=config.curriculum_ramp_epochs,
        sample_jitter=config.curriculum_sample_jitter,
        final_min_mask_rate=config.final_min_mask_rate,
        final_max_mask_rate=config.final_max_mask_rate,
        final_beta_alpha=config.final_beta_alpha,
        final_beta_beta=config.final_beta_beta,
    )
    jepa_config = config.jepa_config.build()
    jepa_config.enable_recovery_head = False
    jepa_config.enable_cond_diffusion_head = False

    model = TimeBandJEPAPipeline.from_config(mask_generator=mask_generator, config=jepa_config)
    model.load_predictor_core(config.pretrained_dir)
    model.freeze_tokenizers()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError('Stage-2 training found no trainable predictor parameters')

    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(train_loader))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.1)
    loss_components: list[tuple[AbstractJEPALossComponent, float]] = [
        (StandardJEPALossComponent(), config.jepa_loss_weight),
        (
            MAxisVarianceLossComponent(variance_floor=config.remained_m_vicreg_floor),
            config.remained_m_vicreg_weight,
        ),
        (
            FlattenedOffDiagonalCovarianceFrobeniusLossComponent(),
            config.remained_cov_fro_weight,
        ),
    ]
    loss_fn = WeightedJEPALoss(loss_components)

    model, optimizer, train_loader, test_loader, cross_test_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        test_loader,
        cross_test_loader,
    )

    config_payload = config.to_dict()
    config_payload['total_steps'] = total_steps
    config_payload['world_size'] = accelerator.num_processes
    wandb_init_kwargs = {
        'name': config.wandb_run_name or working_dir.name,
        'dir': str(working_dir),
    }
    wandb_entity = os.environ.get('WANDB_ENTITY', '').strip()
    if wandb_entity:
        wandb_init_kwargs['entity'] = wandb_entity
    accelerator.init_trackers(
        project_name=config.wandb_project,
        config=config_payload,
        init_kwargs={'wandb': wandb_init_kwargs},
    )

    history: list[dict[str, float]] = []
    best_test_loss: float | None = None
    best_epoch: int | None = None

    if accelerator.is_main_process:
        (working_dir / 'run_config.json').write_text(json.dumps(config_payload, indent=2) + '\n', encoding='utf-8')

    for epoch in range(1, config.epochs + 1):
        if hasattr(mask_generator, 'set_epoch'):
            mask_generator.set_epoch(epoch, config.epochs)
        train_total_values: list[Tensor] = []
        train_component_values: dict[str, list[Tensor]] = {}
        train_collapse_values: dict[str, list[Tensor]] = {}
        train_mask_rates: list[float] = []
        for batch_x in train_loader:
            batch_x = batch_x.to(dtype=torch.float32)
            output, loss_output = train_jepa_batch(
                model,
                batch_x,
                loss_fn,
                optimizer,
                accelerator=accelerator,
                scheduler=scheduler,
                teacher_momentum=None,
            )
            train_total_values.append(loss_output.total_loss.detach())
            for name, value in loss_output.components.items():
                train_component_values.setdefault(name, []).append(value.detach())
            for name, value in _collapse_metrics(output).items():
                if not torch.isnan(value):
                    train_collapse_values.setdefault(name, []).append(value.detach())
            unwrapped_model = accelerator.unwrap_model(model)
            current_rate = getattr(unwrapped_model.mask_generator, 'last_mask_rate', None)
            if current_rate is not None:
                train_mask_rates.append(float(current_rate))

        epoch_metrics = {
            'epoch': float(epoch),
            'train/total_loss': _mean_metric(accelerator, train_total_values),
            'train/lr': float(optimizer.param_groups[0]['lr']),
        }
        if train_mask_rates:
            epoch_metrics['train/mask_rate'] = float(sum(train_mask_rates) / len(train_mask_rates))
        for name, values in train_component_values.items():
            epoch_metrics[f'train/{name}'] = _mean_metric(accelerator, values)
        for name, values in train_collapse_values.items():
            epoch_metrics[f'train/{name}'] = _mean_metric(accelerator, values)

        if epoch % 5 == 0:
            for split_name, split_loader in [('test', test_loader)]:
                test_total_values: list[Tensor] = []
                test_component_values: dict[str, list[Tensor]] = {}
                test_collapse_values: dict[str, list[Tensor]] = {}
                test_mask_rates: list[float] = []
                for batch_x in split_loader:
                    batch_x = batch_x.to(dtype=torch.float32)
                    output, loss_output = evaluate_jepa_batch(
                        model,
                        batch_x,
                        loss_fn,
                    )
                    test_total_values.append(loss_output.total_loss.detach())
                    for name, value in loss_output.components.items():
                        test_component_values.setdefault(name, []).append(value.detach())
                    for name, value in _collapse_metrics(output).items():
                        if not torch.isnan(value):
                            test_collapse_values.setdefault(name, []).append(value.detach())
                    unwrapped_model = accelerator.unwrap_model(model)
                    current_rate = getattr(unwrapped_model.mask_generator, 'last_mask_rate', None)
                    if current_rate is not None:
                        test_mask_rates.append(float(current_rate))
                epoch_metrics[f'{split_name}/total_loss'] = _mean_metric(accelerator, test_total_values)
                if test_mask_rates:
                    epoch_metrics[f'{split_name}/mask_rate'] = float(sum(test_mask_rates) / len(test_mask_rates))
                for name, values in test_component_values.items():
                    epoch_metrics[f'{split_name}/{name}'] = _mean_metric(accelerator, values)
                for name, values in test_collapse_values.items():
                    epoch_metrics[f'{split_name}/{name}'] = _mean_metric(accelerator, values)

        for name, weight in loss_fn.component_weights.items():
            epoch_metrics[f'loss_weight/{name}'] = float(weight)

        history.append(epoch_metrics)
        accelerator.log(epoch_metrics, step=epoch)

        current_test_loss = epoch_metrics.get('test/total_loss')
        unwrapped_model = accelerator.unwrap_model(model)
        if current_test_loss is not None and (best_test_loss is None or current_test_loss < best_test_loss):
            best_test_loss = current_test_loss
            best_epoch = epoch
            if accelerator.is_main_process:
                unwrapped_model.save(working_dir / 'best_model')

        if config.save_every > 0 and epoch % config.save_every == 0 and accelerator.is_main_process:
            unwrapped_model.save(working_dir / f'checkpoint_epoch_{epoch:03d}')

    summary = {
        'best_epoch': best_epoch,
        'best_test_loss': best_test_loss,
        'history': history,
    }
    if accelerator.is_main_process:
        (working_dir / 'results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        accelerator.unwrap_model(model).save(working_dir / 'last_model')
    accelerator.end_training()
    return summary
