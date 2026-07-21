from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from twm.pipelines.tokenizer import AbstractCellTokenizer

from .models import RecoveryHeadOutput, SymmetricRecoveryDecoder32x16Large


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


@dataclass
class RecoveryProbeConfig:
    pass


@dataclass
class RecoveryProbeBatchOutput:
    loss: Tensor
    recon_loss: Tensor
    practical_recon_loss: Tensor
    gt_recon_loss: Tensor | None
    recovered: Tensor
    target: Tensor
    gt_target: Tensor | None


class RecoveryProbeTask(nn.Module):
    def __init__(
        self,
        tokenizer: AbstractCellTokenizer,
        preprocessor: nn.Module,
        config: RecoveryProbeConfig,
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
        self.recovery_head = SymmetricRecoveryDecoder32x16Large(
            num_tokens=int(self.tokenizer.Nt),
            token_dim=int(self.tokenizer.Lt),
        )

    @classmethod
    def from_backbone_3p5(cls, backbone: nn.Module, config: RecoveryProbeConfig) -> RecoveryProbeTask:
        return cls(tokenizer=backbone.tokenizer_3p5, preprocessor=backbone.preprocessor, config=config)

    @classmethod
    def from_backbone_28(cls, backbone: nn.Module, config: RecoveryProbeConfig) -> RecoveryProbeTask:
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

    def _compute_loss(self, recovered: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(recovered, target)

    def forward(self, channel: Tensor) -> RecoveryHeadOutput:
        tokens, _ = self.encode_tokens_and_target(channel)
        return self.recovery_head(tokens)

    def training_step(self, channel: Tensor, gt: Tensor | None = None) -> RecoveryProbeBatchOutput:
        tokens, target = self.encode_tokens_and_target(channel)
        output = self.recovery_head(tokens)
        recon_loss = self._compute_loss(output.recovered, target)
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
            gt_recon_loss = self._compute_loss(output.recovered, gt_target)
        return RecoveryProbeBatchOutput(
            loss=recon_loss,
            recon_loss=recon_loss,
            practical_recon_loss=recon_loss,
            gt_recon_loss=gt_recon_loss,
            recovered=output.recovered,
            target=target,
            gt_target=gt_target,
        )


@torch.no_grad()
def evaluate_recovery_probe_batch(task: RecoveryProbeTask, channel: Tensor, gt: Tensor | None = None) -> RecoveryProbeBatchOutput:
    was_training = task.training
    task.eval()
    result = task.training_step(channel, gt=gt)
    if was_training:
        task.train()
    return result


def train_recovery_probe_batch(
    task: RecoveryProbeTask,
    channel: Tensor,
    optimizer: torch.optim.Optimizer,
    gt: Tensor | None = None,
) -> RecoveryProbeBatchOutput:
    task.train()
    optimizer.zero_grad(set_to_none=True)
    result = task.training_step(channel, gt=gt)
    result.loss.backward()
    optimizer.step()
    return result
