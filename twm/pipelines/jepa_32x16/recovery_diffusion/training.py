from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from twm.pipelines.tokenizer import AbstractCellTokenizer

from .models import (
    DiffusersConditionalRecoveryModel32x16,
    DiffusersUnconditionalRecoveryModel32x16,
    RecoveryDiffusionOutput,
)


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


@dataclass
class RecoveryDiffusionConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    down_block_out_channels: tuple[int, ...] = (64, 128, 128)
    layers_per_block: int = 1


@dataclass
class RecoveryDiffusionBatchOutput:
    loss: Tensor
    noise_loss: Tensor
    recon_loss: Tensor
    practical_recon_loss: Tensor
    gt_recon_loss: Tensor | None
    predicted_noise: Tensor
    recovered: Tensor
    target: Tensor
    gt_target: Tensor | None


class RecoveryDiffusionTask(nn.Module):
    def __init__(
        self,
        tokenizer: AbstractCellTokenizer,
        preprocessor: nn.Module,
        config: RecoveryDiffusionConfig,
        *,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = copy.deepcopy(tokenizer)
        self.preprocessor = copy.deepcopy(preprocessor)
        self.config = config
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            _freeze_module(self.tokenizer)
            _freeze_module(self.preprocessor)
        self.diffusion_model = DiffusersConditionalRecoveryModel32x16(
            token_dim=int(self.tokenizer.Lt),
            down_block_out_channels=tuple(int(v) for v in config.down_block_out_channels),
            layers_per_block=int(config.layers_per_block),
            num_train_timesteps=int(config.num_train_timesteps),
            beta_start=float(config.beta_start),
            beta_end=float(config.beta_end),
        )

    @classmethod
    def from_backbone_3p5(cls, backbone: nn.Module, config: RecoveryDiffusionConfig) -> RecoveryDiffusionTask:
        return cls(tokenizer=backbone.tokenizer_3p5, preprocessor=backbone.preprocessor, config=config)

    @classmethod
    def from_backbone_28(cls, backbone: nn.Module, config: RecoveryDiffusionConfig) -> RecoveryDiffusionTask:
        return cls(tokenizer=backbone.tokenizer_28, preprocessor=backbone.preprocessor, config=config)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.tokenizer.eval()
            self.preprocessor.eval()
        return self

    def encode_tokens_and_target(self, channel: Tensor) -> tuple[Tensor, Tensor]:
        if channel.ndim == 5:
            flat_channel = channel.reshape(channel.shape[0] * channel.shape[1], *channel.shape[2:])
        elif channel.ndim == 4:
            flat_channel = channel
        else:
            raise ValueError(f'Expected channel shape [B, N, 2, H, W] or [B, 2, H, W], got {tuple(channel.shape)}')
        if self.freeze_backbone:
            with torch.no_grad():
                target = self.preprocessor(flat_channel)
                tokens = self.tokenizer(target)
        else:
            target = self.preprocessor(flat_channel)
            tokens = self.tokenizer(target)
        return tokens, target

    def _sample_noisy_target(self, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        noise = torch.randn_like(target)
        timesteps = self.diffusion_model.sample_timesteps(target.shape[0], target.device)
        noisy_target = self.diffusion_model.add_noise(target, noise, timesteps)
        return noisy_target, noise, timesteps

    def _predict_x0(self, noisy_target: Tensor, predicted_noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.diffusion_model.predict_x0(noisy_target, predicted_noise, timesteps)

    def forward(self, channel: Tensor) -> RecoveryDiffusionOutput:
        tokens, target = self.encode_tokens_and_target(channel)
        noisy_target, _, timesteps = self._sample_noisy_target(target)
        return self.diffusion_model(noisy_target, tokens, timesteps)

    def training_step(self, channel: Tensor, gt: Tensor | None = None) -> RecoveryDiffusionBatchOutput:
        tokens, target = self.encode_tokens_and_target(channel)
        noisy_target, true_noise, timesteps = self._sample_noisy_target(target)
        output = self.diffusion_model(noisy_target, tokens, timesteps)
        noise_loss = F.mse_loss(output.predicted_noise, true_noise)
        recovered = self._predict_x0(noisy_target, output.predicted_noise, timesteps)
        recon_loss = F.mse_loss(recovered, target)
        gt_target: Tensor | None = None
        gt_recon_loss: Tensor | None = None
        if gt is not None:
            if gt.ndim == 5:
                flat_gt = gt.reshape(gt.shape[0] * gt.shape[1], *gt.shape[2:])
            elif gt.ndim == 4:
                flat_gt = gt
            else:
                raise ValueError(f'Expected gt shape [B, N, 2, H, W] or [B, 2, H, W], got {tuple(gt.shape)}')
            gt_target = self.preprocessor(flat_gt)
            gt_recon_loss = F.mse_loss(recovered, gt_target)
        return RecoveryDiffusionBatchOutput(
            loss=noise_loss,
            noise_loss=noise_loss,
            recon_loss=recon_loss,
            practical_recon_loss=recon_loss,
            gt_recon_loss=gt_recon_loss,
            predicted_noise=output.predicted_noise,
            recovered=recovered,
            target=target,
            gt_target=gt_target,
        )


@torch.no_grad()
def evaluate_recovery_diffusion_batch(task: RecoveryDiffusionTask, channel: Tensor, gt: Tensor | None = None) -> RecoveryDiffusionBatchOutput:
    was_training = task.training
    task.eval()
    result = task.training_step(channel, gt=gt)
    if was_training:
        task.train()
    return result


def train_recovery_diffusion_batch(
    task: RecoveryDiffusionTask,
    channel: Tensor,
    optimizer: torch.optim.Optimizer,
    gt: Tensor | None = None,
) -> RecoveryDiffusionBatchOutput:
    task.train()
    optimizer.zero_grad(set_to_none=True)
    result = task.training_step(channel, gt=gt)
    result.loss.backward()
    optimizer.step()
    return result


class RecoveryUnconditionalDiffusionTask(nn.Module):
    def __init__(
        self,
        preprocessor: nn.Module,
        config: RecoveryDiffusionConfig,
    ) -> None:
        super().__init__()
        self.preprocessor = copy.deepcopy(preprocessor)
        self.config = config
        self.diffusion_model = DiffusersUnconditionalRecoveryModel32x16(
            down_block_out_channels=tuple(int(v) for v in config.down_block_out_channels),
            layers_per_block=int(config.layers_per_block),
            num_train_timesteps=int(config.num_train_timesteps),
            beta_start=float(config.beta_start),
            beta_end=float(config.beta_end),
        )

    def target_from_channel(self, channel: Tensor) -> Tensor:
        if channel.ndim == 5:
            flat_channel = channel.reshape(channel.shape[0] * channel.shape[1], *channel.shape[2:])
        elif channel.ndim == 4:
            flat_channel = channel
        else:
            raise ValueError(f'Expected channel shape [B, N, 2, H, W] or [B, 2, H, W], got {tuple(channel.shape)}')
        return self.preprocessor(flat_channel)

    def _sample_noisy_target(self, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        noise = torch.randn_like(target)
        timesteps = self.diffusion_model.sample_timesteps(target.shape[0], target.device)
        noisy_target = self.diffusion_model.add_noise(target, noise, timesteps)
        return noisy_target, noise, timesteps

    def _predict_x0(self, noisy_target: Tensor, predicted_noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.diffusion_model.predict_x0(noisy_target, predicted_noise, timesteps)

    def forward(self, channel: Tensor) -> RecoveryDiffusionOutput:
        target = self.target_from_channel(channel)
        noisy_target, _, timesteps = self._sample_noisy_target(target)
        return self.diffusion_model(noisy_target, timesteps)

    def training_step(self, channel: Tensor, gt: Tensor | None = None) -> RecoveryDiffusionBatchOutput:
        target = self.target_from_channel(channel)
        noisy_target, true_noise, timesteps = self._sample_noisy_target(target)
        output = self.diffusion_model(noisy_target, timesteps)
        noise_loss = F.mse_loss(output.predicted_noise, true_noise)
        recovered = self._predict_x0(noisy_target, output.predicted_noise, timesteps)
        recon_loss = F.mse_loss(recovered, target)
        gt_target: Tensor | None = None
        gt_recon_loss: Tensor | None = None
        if gt is not None:
            if gt.ndim == 5:
                flat_gt = gt.reshape(gt.shape[0] * gt.shape[1], *gt.shape[2:])
            elif gt.ndim == 4:
                flat_gt = gt
            else:
                raise ValueError(f'Expected gt shape [B, N, 2, H, W] or [B, 2, H, W], got {tuple(gt.shape)}')
            gt_target = self.preprocessor(flat_gt)
            gt_recon_loss = F.mse_loss(recovered, gt_target)
        return RecoveryDiffusionBatchOutput(
            loss=noise_loss,
            noise_loss=noise_loss,
            recon_loss=recon_loss,
            practical_recon_loss=recon_loss,
            gt_recon_loss=gt_recon_loss,
            predicted_noise=output.predicted_noise,
            recovered=recovered,
            target=target,
            gt_target=gt_target,
        )


@torch.no_grad()
def evaluate_recovery_unconditional_diffusion_batch(
    task: RecoveryUnconditionalDiffusionTask,
    channel: Tensor,
    gt: Tensor | None = None,
) -> RecoveryDiffusionBatchOutput:
    was_training = task.training
    task.eval()
    result = task.training_step(channel, gt=gt)
    if was_training:
        task.train()
    return result


def train_recovery_unconditional_diffusion_batch(
    task: RecoveryUnconditionalDiffusionTask,
    channel: Tensor,
    optimizer: torch.optim.Optimizer,
    gt: Tensor | None = None,
) -> RecoveryDiffusionBatchOutput:
    task.train()
    optimizer.zero_grad(set_to_none=True)
    result = task.training_step(channel, gt=gt)
    result.loss.backward()
    optimizer.step()
    return result
