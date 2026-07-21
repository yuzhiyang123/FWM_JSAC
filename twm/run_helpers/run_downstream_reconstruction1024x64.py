from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import wandb

from twm.downstream.channel_pred import build_channel_pred_three_way_dataloaders
from twm.downstream.channel_pred_reconstruction1024x64.training import (
    ChannelPredReconstruction1024x64Config,
    ChannelPredReconstruction1024x64Task,
    build_conditioning_backbone,
)
from twm.run_helpers.run_downstream_common import _accumulate_channel_pred_metrics, _mean, _move_batch_to_device, _save_model_state
from twm.pipelines.jepa_32x16 import build_warmup_cosine_scheduler


@dataclass
class ChannelPredReconstruction1024x64TrainingConfig:
    dataset_file: list[str] = field(default_factory=list)
    pretrained_dir: str = ''
    working_dir: str = ''
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    device: str = 'cuda'
    train_paths: int = 512
    test_paths: int = 88
    epochs: int = 100
    window_length: int = 16
    random_windows_per_sequence: int = 30
    eval_every_epochs: int = 10
    wandb_project: str = 'twm_downstream_reconstruction1024x64_open'
    wandb_run_name: str = ''
    freeze_backbone: bool = True
    pilot_align_mode: str = 'none'
    pilot_est_subcarrier_strides: tuple[int, ...] = (8,)
    pilot_est_noise_mode: str = 'uniform_gaussian'
    pilot_fixed_nmse: float | None = None
    eval_history_mask_rate: float | None = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> 'ChannelPredReconstruction1024x64TrainingConfig':
        payload = vars(args).copy()
        payload['dataset_file'] = [str(x) for x in payload['dataset_file']]
        payload['pretrained_dir'] = str(payload['pretrained_dir'])
        payload['working_dir'] = str(payload['working_dir'])
        payload['pilot_est_subcarrier_strides'] = tuple(int(x) for x in payload['pilot_est_subcarrier_strides'])
        return cls(**payload)


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = ChannelPredReconstruction1024x64TrainingConfig()
    parser = argparse.ArgumentParser(description='Train the 1024x64 downstream reconstruction task with a frozen 128x64/1024x64 JEPA backbone.')
    parser.add_argument('--dataset-file', nargs='+', required=True)
    parser.add_argument('--pretrained-dir', required=True)
    parser.add_argument('--working-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=defaults.batch_size)
    parser.add_argument('--lr', type=float, default=defaults.lr)
    parser.add_argument('--weight-decay', type=float, default=defaults.weight_decay)
    parser.add_argument('--num-workers', type=int, default=defaults.num_workers)
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--device', default=defaults.device)
    parser.add_argument('--train-paths', type=int, default=defaults.train_paths)
    parser.add_argument('--test-paths', type=int, default=defaults.test_paths)
    parser.add_argument('--epochs', type=int, default=defaults.epochs)
    parser.add_argument('--window-length', type=int, default=defaults.window_length)
    parser.add_argument('--random-windows-per-sequence', type=int, default=defaults.random_windows_per_sequence)
    parser.add_argument('--eval-every-epochs', type=int, default=defaults.eval_every_epochs)
    parser.add_argument('--wandb-project', default=defaults.wandb_project)
    parser.add_argument('--wandb-run-name', default=defaults.wandb_run_name)
    parser.add_argument('--pilot-align-mode', choices=['none', 'mlp'], default=defaults.pilot_align_mode)
    parser.add_argument('--pilot-est-subcarrier-strides', nargs='+', type=int, default=list(defaults.pilot_est_subcarrier_strides))
    parser.add_argument('--pilot-est-noise-mode', choices=['none', 'uniform_gaussian', 'fixed_nmse'], default=defaults.pilot_est_noise_mode)
    parser.add_argument('--pilot-fixed-nmse', type=float, default=defaults.pilot_fixed_nmse)
    parser.add_argument('--eval-history-mask-rate', type=float, default=defaults.eval_history_mask_rate)
    parser.add_argument('--freeze-backbone', action='store_true', default=True)
    parser.add_argument('--unfreeze-backbone', action='store_false', dest='freeze_backbone')
    return parser


def _evaluate_loader(task: ChannelPredReconstruction1024x64Task, loader, device: torch.device) -> dict[str, float]:
    task.eval()
    losses: list[float] = []
    x_losses: list[float] = []
    recon: list[float] = []
    practical: list[float] = []
    gt_recon: list[float] = []
    latent_target_power: list[float] = []
    latent_nmse: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            result = task.testing_step(
                batch.history_channel,
                batch.history_mask,
                batch.current_channel,
                subband_mask=batch.subband_mask,
                current_gt=batch.current_gt,
            )
            _accumulate_channel_pred_metrics(result, losses, x_losses, recon, practical, gt_recon, latent_target_power, latent_nmse)
    metrics = {
        'loss': _mean(losses),
        'x_loss': _mean(x_losses),
        'recon_loss': _mean(recon),
        'practical_recon_loss': _mean(practical),
        'latent_target_power': _mean(latent_target_power),
        'latent_nmse': _mean(latent_nmse),
    }
    if gt_recon:
        metrics['gt_recon_loss'] = _mean(gt_recon)
    return metrics


def run_downstream_reconstruction1024x64(args: argparse.Namespace | ChannelPredReconstruction1024x64TrainingConfig) -> dict[str, object]:
    config = args if isinstance(args, ChannelPredReconstruction1024x64TrainingConfig) else ChannelPredReconstruction1024x64TrainingConfig.from_namespace(args)
    working_dir = Path(config.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    train_loader, test_loader, cross_test_loader = build_channel_pred_three_way_dataloaders(
        dataset_file=config.dataset_file,
        batch_size=config.batch_size,
        window_length=config.window_length,
        random_windows_per_sequence=config.random_windows_per_sequence,
        num_workers=config.num_workers,
        seed=config.seed,
        train_paths=config.train_paths,
        test_paths=config.test_paths,
    )

    predictor_backbone, backbone_kind, reference_domain = build_conditioning_backbone(
        config.pretrained_dir,
        freeze_backbone=config.freeze_backbone,
    )
    task = ChannelPredReconstruction1024x64Task(
        predictor_backbone,
        backbone_kind=backbone_kind,
        reference_domain=reference_domain,
        config=ChannelPredReconstruction1024x64Config(
            pilot_align_mode=config.pilot_align_mode,
            pilot_est_subcarrier_strides=config.pilot_est_subcarrier_strides,
            pilot_est_noise_mode=config.pilot_est_noise_mode,
            pilot_fixed_nmse=config.pilot_fixed_nmse,
            eval_history_mask_rate=config.eval_history_mask_rate,
        ),
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    trainable_parameters = [p for p in task.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(train_loader))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.1)

    config_payload = asdict(config)
    config_payload['task_type'] = 'downstream_reconstruction1024x64'
    config_payload['resolved_backbone_kind'] = backbone_kind
    config_payload['resolved_reference_domain'] = reference_domain
    config_payload['reporting_paths'] = {
        'working_dir': str(working_dir),
        'pretrained_dir': str(config.pretrained_dir),
        'dataset_file': list(config.dataset_file),
    }
    (working_dir / 'run_config.json').write_text(json.dumps(config_payload, indent=2) + "\n", encoding='utf-8')

    wandb_init_kwargs = {'name': config.wandb_run_name or working_dir.name, 'dir': str(working_dir), 'config': config_payload}
    wandb_entity = os.environ.get('WANDB_ENTITY', '').strip()
    if wandb_entity:
        wandb_init_kwargs['entity'] = wandb_entity
    wandb_run = wandb.init(project=config.wandb_project, **wandb_init_kwargs)

    history: list[dict[str, float]] = []
    best_epoch: int | None = None
    best_test_loss: float | None = None
    eval_every_epochs = max(1, int(config.eval_every_epochs))
    try:
        for epoch in range(1, config.epochs + 1):
            task.train()
            train_losses: list[float] = []
            train_x: list[float] = []
            train_recon: list[float] = []
            train_practical: list[float] = []
            train_gt: list[float] = []
            train_latent_target_power: list[float] = []
            train_latent_nmse: list[float] = []
            for batch in train_loader:
                batch = _move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = task.training_step(
                    batch.history_channel,
                    batch.history_mask,
                    batch.current_channel,
                    subband_mask=batch.subband_mask,
                    current_gt=batch.current_gt,
                )
                output.loss.backward()
                optimizer.step()
                scheduler.step()
                train_losses.append(float(output.loss.detach().item()))
                train_x.append(float(output.x_loss.detach().item()))
                train_recon.append(float(output.recon_loss.detach().item()))
                train_practical.append(float(output.practical_recon_loss.detach().item()))
                train_latent_target_power.append(float(output.latent_target_power.detach().item()))
                train_latent_nmse.append(float(output.latent_nmse.detach().item()))
                if output.gt_recon_loss is not None:
                    train_gt.append(float(output.gt_recon_loss.detach().item()))

            epoch_metrics: dict[str, float] = {
                'epoch': float(epoch),
                'train/loss': _mean(train_losses),
                'train/x_loss': _mean(train_x),
                'train/recon_loss': _mean(train_recon),
                'train/practical_recon_loss': _mean(train_practical),
                'train/latent_target_power': _mean(train_latent_target_power),
                'train/latent_nmse': _mean(train_latent_nmse),
                'train/lr': float(optimizer.param_groups[0]['lr']),
            }
            if train_gt:
                epoch_metrics['train/gt_recon_loss'] = _mean(train_gt)

            if epoch % eval_every_epochs == 0 or epoch == config.epochs:
                test_metrics = _evaluate_loader(task, test_loader, device)
                epoch_metrics.update({f'test/{name}': value for name, value in test_metrics.items()})
                cross_metrics = _evaluate_loader(task, cross_test_loader, device)
                epoch_metrics.update({f'cross_test/{name}': value for name, value in cross_metrics.items()})
                current_test_loss = test_metrics['loss']
                if best_test_loss is None or current_test_loss < best_test_loss:
                    best_test_loss = current_test_loss
                    best_epoch = epoch
                    _save_model_state(task, working_dir / 'best_model.pt')
            history.append(epoch_metrics)
            if wandb_run is not None:
                wandb.log(epoch_metrics, step=epoch)

        summary = {
            'best_epoch': best_epoch,
            'best_test_loss': best_test_loss,
            'history': history,
        }
        (working_dir / 'results.json').write_text(json.dumps(summary, indent=2) + "\n", encoding='utf-8')
        _save_model_state(task, working_dir / 'last_model.pt')
        return summary
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    run_downstream_reconstruction1024x64(args)
