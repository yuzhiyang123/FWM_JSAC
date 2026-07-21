from __future__ import annotations

import argparse
import json
import re
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import wandb

from twm.downstream.channel_pred import (
    ChannelPredBatch,
    ChannelPredConfig,
    ChannelPredTask,
    ChannelPredTrainOutput,
    build_channel_pred_three_way_dataloaders,
)
FAILED_JEPA_EXIT_CODE = 88

from twm.pipelines.jepa_32x16 import (
    NamedJEPAConfig,
    RandomMaskExceptLastGenerator,
    TimeBandJEPAPipeline,
    build_warmup_cosine_scheduler,
)


@dataclass
class ChannelPredTrainingConfig:
    dataset_file: list[str] = field(default_factory=list)
    pretrained_dir: str = ''
    working_dir: str = ''
    jepa_config_name: str = NamedJEPAConfig().preset_name
    num_low_bands: int = NamedJEPAConfig().num_low_bands
    num_tokens: int = NamedJEPAConfig().num_tokens
    token_dim: int = NamedJEPAConfig().token_dim
    predictor_dim: int = NamedJEPAConfig().predictor_dim
    tokenizer_output_norm: str = 'layernorm'
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    device: str = 'cuda'
    train_paths: int = 512
    test_paths: int = 88
    epochs: int = 300
    window_length: int = 16
    random_windows_per_sequence: int = 30
    eval_every_epochs: int = 10
    pilot_est_subcarrier_strides: tuple[int, ...] = (4,)
    pilot_est_max_noise_std: float = 1.224744871391589
    pilot_input_noise_var: float = 0.5
    pilot_total_noise_var_target: float = 0.5
    fusion_mode: str = 'split_cross_attn'
    wandb_project: str = 'twm_downstream_common_open'
    wandb_run_name: str = ''
    failed_jepa_list: str = ''
    latent_nmse_fail_threshold: float = 0.8
    diffusion_prediction_target: str = 'noise'
    normalize_diffusion_io: bool = False
    final_test_sample_steps: int = 10
    skip_post_training_generation: bool = False
    train_mode: str = 'pointwise'

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> 'ChannelPredTrainingConfig':
        payload = vars(args).copy()
        payload['dataset_file'] = [str(x) for x in payload['dataset_file']]
        payload['pretrained_dir'] = str(payload['pretrained_dir'])
        payload['working_dir'] = str(payload['working_dir'])
        payload['pilot_est_subcarrier_strides'] = tuple(int(x) for x in payload['pilot_est_subcarrier_strides'])
        return cls(**payload)


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = ChannelPredTrainingConfig()
    parser = argparse.ArgumentParser(description='Train the pilot-conditioned channel prediction task with a frozen JEPA backbone.')
    parser.add_argument('--dataset-file', nargs='+', required=True)
    parser.add_argument('--pretrained-dir', required=True)
    parser.add_argument('--working-dir', required=True)
    parser.add_argument('--jepa-config-name', default=defaults.jepa_config_name)
    parser.add_argument('--num-low-bands', type=int, default=defaults.num_low_bands)
    parser.add_argument('--num-tokens', type=int, default=defaults.num_tokens)
    parser.add_argument('--token-dim', type=int, default=defaults.token_dim)
    parser.add_argument('--predictor-dim', type=int, default=defaults.predictor_dim)
    parser.add_argument('--tokenizer-output-norm', choices=['batchnorm', 'layernorm', 'none'], default=defaults.tokenizer_output_norm)
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
    parser.add_argument('--pilot-est-subcarrier-strides', nargs='+', type=int, default=list(defaults.pilot_est_subcarrier_strides))
    parser.add_argument('--pilot-est-max-noise-std', type=float, default=defaults.pilot_est_max_noise_std)
    parser.add_argument('--pilot-input-noise-var', type=float, default=defaults.pilot_input_noise_var)
    parser.add_argument('--pilot-total-noise-var-target', type=float, default=defaults.pilot_total_noise_var_target)
    parser.add_argument('--fusion-mode', choices=['pilot_only', 'split_cross_attn', 'gated_fusion', 'latent_residual', 'late_fusion'], default=defaults.fusion_mode)
    parser.add_argument('--wandb-project', default=defaults.wandb_project)
    parser.add_argument('--wandb-run-name', default=defaults.wandb_run_name)
    parser.add_argument('--failed-jepa-list', default=defaults.failed_jepa_list)
    parser.add_argument('--latent-nmse-fail-threshold', type=float, default=defaults.latent_nmse_fail_threshold)
    parser.add_argument('--diffusion-prediction-target', choices=['x0', 'v_prediction', 'noise'], default=defaults.diffusion_prediction_target)
    parser.add_argument('--normalize-diffusion-io', action='store_true', default=defaults.normalize_diffusion_io)
    parser.add_argument('--final-test-sample-steps', type=int, default=defaults.final_test_sample_steps)
    parser.add_argument('--skip-post-training-generation', action='store_true', default=defaults.skip_post_training_generation)
    return parser


def _move_batch_to_device(batch: ChannelPredBatch, device: torch.device) -> ChannelPredBatch:
    return ChannelPredBatch(
        history_channel=batch.history_channel.to(device=device, dtype=torch.float32),
        history_mask=batch.history_mask.to(device=device, dtype=torch.bool),
        current_channel=batch.current_channel.to(device=device, dtype=torch.float32),
        pilot_mask=batch.pilot_mask.to(device=device, dtype=torch.bool),
        subband_mask=batch.subband_mask.to(device=device, dtype=torch.bool),
        current_gt=None if batch.current_gt is None else batch.current_gt.to(device=device, dtype=torch.float32),
    )


def _infer_run_info(path_like: str | Path) -> tuple[int | None, str]:
    path = Path(path_like)
    name = path.name
    if path.is_dir() and path.name == 'best_model':
        name = path.parent.name
    elif path.suffix == '.pt':
        name = path.parent.name
    match = re.match(r'^(\d+)_', name)
    return (int(match.group(1)) if match else None, name)


def _append_failed_jepa(list_path: str | Path, *, run_id: int | None, run_name: str, reason: str, latent_nmse: float, latent_target_power: float) -> None:
    path = Path(list_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(payload, list):
                payload = []
        except Exception:
            payload = []
    else:
        payload = []
    entry = {
        'run_id': run_id,
        'run_name': run_name,
        'reason': reason,
        'latent_nmse': float(latent_nmse),
        'latent_target_power': float(latent_target_power),
    }
    key = (run_id, run_name)
    existing = {(item.get('run_id'), item.get('run_name')) for item in payload if isinstance(item, dict)}
    if key not in existing:
        payload.append(entry)
        path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _latent_precheck(task, loader, device: torch.device, *, max_batches: int = 3) -> dict[str, float]:
    latent_nmse_values: list[float] = []
    latent_target_power_values: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = _move_batch_to_device(batch, device)
            diagnostics = task.latent_diagnostics_step(batch.history_channel, batch.history_mask, batch.current_channel)
            latent_target_power_values.append(float(diagnostics[-2].detach().item()))
            latent_nmse_values.append(float(diagnostics[-1].detach().item()))
    return {
        'latent_target_power': _mean(latent_target_power_values),
        'latent_nmse': _mean(latent_nmse_values),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _save_model_state(task: ChannelPredTask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(task.state_dict(), path)


def _infer_tokenizer_output_norm_from_pretrained(pretrained_dir: str | Path) -> str | None:
    # Legacy checkpoints often do not expose affine-free output norm keys in the
    # tokenizer state dict, so introspection is unreliable. Let the explicit
    # downstream training config decide when provided; otherwise fall back to the
    # saved JEPA config in run_config.json.
    _ = pretrained_dir
    return None


def _resolve_named_jepa_from_pretrained(config: ChannelPredTrainingConfig) -> NamedJEPAConfig | None:
    pretrained_dir = Path(config.pretrained_dir)
    if not pretrained_dir.exists():
        return None
    if pretrained_dir.is_dir() and pretrained_dir.name == "best_model":
        run_config_path = pretrained_dir.parent / "run_config.json"
    elif pretrained_dir.suffix == ".pt":
        run_config_path = pretrained_dir.parent / "run_config.json"
    else:
        run_config_path = pretrained_dir / "run_config.json"
    if not run_config_path.exists():
        return None
    try:
        payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    jepa_cfg = payload.get("jepa_config")
    if not isinstance(jepa_cfg, dict):
        return None
    try:
        inferred_tokenizer_output_norm = _infer_tokenizer_output_norm_from_pretrained(config.pretrained_dir)
        requested_tokenizer_output_norm = getattr(config, 'tokenizer_output_norm', None)
        resolved_tokenizer_output_norm = requested_tokenizer_output_norm or inferred_tokenizer_output_norm or jepa_cfg.get('tokenizer_output_norm', None) or 'layernorm'
        return NamedJEPAConfig(
            preset_name=str(jepa_cfg.get("preset_name", config.jepa_config_name)),
            num_low_bands=int(jepa_cfg.get("num_low_bands", config.num_low_bands)),
            num_tokens=int(jepa_cfg.get("num_tokens", config.num_tokens)),
            token_dim=int(jepa_cfg.get("token_dim", config.token_dim)),
            predictor_dim=int(jepa_cfg.get("predictor_dim", config.predictor_dim)),
            tokenizer_output_norm=str(resolved_tokenizer_output_norm),
        )
    except Exception:
        return None

def _build_backbone(config: ChannelPredTrainingConfig) -> TimeBandJEPAPipeline:
    named = _resolve_named_jepa_from_pretrained(config)
    if named is None:
        named = NamedJEPAConfig(
            preset_name=config.jepa_config_name,
            num_low_bands=config.num_low_bands,
            num_tokens=config.num_tokens,
            token_dim=config.token_dim,
            predictor_dim=config.predictor_dim,
            tokenizer_output_norm=config.tokenizer_output_norm,
        )
    jepa_config = named.build()
    jepa_config.enable_recovery_head = False
    jepa_config.enable_cond_diffusion_head = False
    return TimeBandJEPAPipeline.from_config(
        mask_generator=RandomMaskExceptLastGenerator(mask_rate=0.5),
        config=jepa_config,
    )


def _accumulate_channel_pred_metrics(result: ChannelPredTrainOutput, losses: list[float], noise: list[float], recon: list[float], practical: list[float], gt_recon: list[float], latent_target_power: list[float], latent_nmse: list[float]) -> None:
    losses.append(float(result.loss.detach().item()))
    noise.append(float(result.x_loss.detach().item()))
    recon.append(float(result.recon_loss.detach().item()))
    practical.append(float(result.practical_recon_loss.detach().item()))
    latent_target_power.append(float(result.latent_target_power.detach().item()))
    latent_nmse.append(float(result.latent_nmse.detach().item()))
    if result.gt_recon_loss is not None:
        gt_recon.append(float(result.gt_recon_loss.detach().item()))


def _evaluate_channel_pred_loader(task: ChannelPredTask, loader, device: torch.device) -> dict[str, float]:
    task.eval()
    split_losses: list[float] = []
    split_noise: list[float] = []
    split_recon: list[float] = []
    split_practical: list[float] = []
    split_gt_recon: list[float] = []
    split_latent_target_power: list[float] = []
    split_latent_nmse: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            result = task.training_step(
                batch.history_channel,
                batch.history_mask,
                batch.current_channel,
                subband_mask=batch.subband_mask,
                current_gt=batch.current_gt,
            )
            _accumulate_channel_pred_metrics(result, split_losses, split_noise, split_recon, split_practical, split_gt_recon, split_latent_target_power, split_latent_nmse)
    metrics = {
        'loss': _mean(split_losses),
        'x_loss': _mean(split_noise),
        'recon_loss': _mean(split_recon),
        'practical_recon_loss': _mean(split_practical),
        'latent_target_power': _mean(split_latent_target_power),
        'latent_nmse': _mean(split_latent_nmse),
    }
    if split_gt_recon:
        metrics['gt_recon_loss'] = _mean(split_gt_recon)
    return metrics


def _evaluate_channel_pred_generation_loader(task: ChannelPredTask, loader, device: torch.device) -> dict[str, float]:
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
            losses.append(float(result.loss.detach().item()))
            x_losses.append(float(result.x_loss.detach().item()))
            recon.append(float(result.recon_loss.detach().item()))
            practical.append(float(result.practical_recon_loss.detach().item()))
            latent_target_power.append(float(result.latent_target_power.detach().item()))
            latent_nmse.append(float(result.latent_nmse.detach().item()))
            if result.gt_recon_loss is not None:
                gt_recon.append(float(result.gt_recon_loss.detach().item()))
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


def _run_post_training_generation_test(task: ChannelPredTask, test_loader, cross_loader, device: torch.device, *, sample_steps: int) -> dict[str, object]:
    previous_steps = int(task.config.ddpm_sample_steps)
    task.config.ddpm_sample_steps = int(sample_steps)
    try:
        return {
            'sample_steps': int(sample_steps),
            'test': _evaluate_channel_pred_generation_loader(task, test_loader, device),
            'cross_test': _evaluate_channel_pred_generation_loader(task, cross_loader, device),
        }
    finally:
        task.config.ddpm_sample_steps = previous_steps


def _flatten_metric_dict(prefix: str, metrics: dict[str, object]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat.update(_flatten_metric_dict(f'{prefix}/{key}', value))
        elif isinstance(value, (int, float)):
            flat[f'{prefix}/{key}'] = float(value)
    return flat


def run_downstream_common(args: argparse.Namespace | ChannelPredTrainingConfig) -> dict[str, object]:
    config = args if isinstance(args, ChannelPredTrainingConfig) else ChannelPredTrainingConfig.from_namespace(args)
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

    backbone = _build_backbone(config)
    task = ChannelPredTask.from_stage2_checkpoint(
        backbone,
        pretrained_dir=config.pretrained_dir,
        config=ChannelPredConfig(
            pilot_align_mode='none',
            channel_mode='complex',
            pilot_est_subcarrier_strides=config.pilot_est_subcarrier_strides,
            pilot_est_max_noise_std=config.pilot_est_max_noise_std,
            pilot_input_noise_var=config.pilot_input_noise_var,
            pilot_total_noise_var_target=config.pilot_total_noise_var_target,
            fusion_mode=config.fusion_mode,
            diffusion_prediction_target=config.diffusion_prediction_target,
            normalize_diffusion_io=config.normalize_diffusion_io,
        ),
        freeze_backbone=True,
    ).to(device)

    trainable_parameters = [p for p in task.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError('Channel prediction training found no trainable parameters.')

    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(train_loader))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.1)

    failed_jepa_list_path = config.failed_jepa_list or str((working_dir.parent / 'failed_jepa_runs.json'))

    config_payload = asdict(config)
    config_payload['pilot_align_mode'] = 'none'
    config_payload['channel_mode'] = 'complex'
    config_payload['freeze_backbone'] = True
    config_payload['total_steps'] = total_steps
    config_payload['reporting_paths'] = {
        'working_dir': str(working_dir),
        'pretrained_dir': str(config.pretrained_dir),
        'dataset_file': list(config.dataset_file),
        'failed_jepa_list': failed_jepa_list_path,
    }
    (working_dir / 'run_config.json').write_text(json.dumps(config_payload, indent=2) + '\n', encoding='utf-8')

    wandb_init_kwargs = {
        'name': config.wandb_run_name or working_dir.name,
        'dir': str(working_dir),
        'config': config_payload,
    }
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
            train_noise: list[float] = []
            train_recon: list[float] = []
            train_practical: list[float] = []
            train_gt_recon: list[float] = []
            train_latent_target_power: list[float] = []
            train_latent_nmse: list[float] = []

            for batch in train_loader:
                batch = _move_batch_to_device(batch, device)
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
                scheduler.step()
                _accumulate_channel_pred_metrics(result, train_losses, train_noise, train_recon, train_practical, train_gt_recon, train_latent_target_power, train_latent_nmse)

            epoch_metrics: dict[str, float] = {
                'epoch': float(epoch),
                'train/loss': _mean(train_losses),
                'train/x_loss': _mean(train_noise),
                'train/recon_loss': _mean(train_recon),
                'train/practical_recon_loss': _mean(train_practical),
                'train/latent_target_power': _mean(train_latent_target_power),
                'train/latent_nmse': _mean(train_latent_nmse),
                'train/lr': float(optimizer.param_groups[0]['lr']),
            }
            if train_gt_recon:
                epoch_metrics['train/gt_recon_loss'] = _mean(train_gt_recon)

            should_eval = (epoch % eval_every_epochs) == 0
            if should_eval:
                test_metrics = _evaluate_channel_pred_loader(task, test_loader, device)
                for key, value in test_metrics.items():
                    epoch_metrics[f'test/{key}'] = value

                current_test_loss = epoch_metrics['test/loss']
                if best_test_loss is None or current_test_loss < best_test_loss:
                    best_test_loss = current_test_loss
                    best_epoch = epoch
                    _save_model_state(task, working_dir / 'best_model.pt')

            history.append(epoch_metrics)
            wandb.log(epoch_metrics, step=epoch)

        _save_model_state(task, working_dir / 'last_model.pt')
        best_checkpoint = working_dir / 'best_model.pt'
        if best_epoch is None or not best_checkpoint.exists():
            best_epoch = config.epochs
            _save_model_state(task, best_checkpoint)

        summary = {
            'best_epoch': best_epoch,
            'best_test_loss': best_test_loss,
            'history': history,
        }
        if not config.skip_post_training_generation:
            task.load_state_dict(torch.load(best_checkpoint, map_location='cpu'))
            post_training_generation = _run_post_training_generation_test(
                task,
                test_loader,
                cross_test_loader,
                device,
                sample_steps=config.final_test_sample_steps,
            )
            wandb.log(_flatten_metric_dict('post_training_generation', post_training_generation), step=config.epochs + 1)
            summary['post_training_generation'] = {
                'checkpoint_path': str(best_checkpoint),
                **post_training_generation,
            }
        (working_dir / 'results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        if wandb_run is not None:
            wandb_run.summary['best_epoch'] = best_epoch
            wandb_run.summary['best_test_loss'] = best_test_loss
            wandb_run.summary['working_dir'] = str(working_dir)
            wandb_run.summary['pretrained_dir'] = str(config.pretrained_dir)
            if not config.skip_post_training_generation:
                wandb_run.summary['post_training_generation_checkpoint_path'] = str(best_checkpoint)
                wandb_run.summary['post_training_generation_sample_steps'] = int(config.final_test_sample_steps)
                test_metrics = post_training_generation['test']
                if isinstance(test_metrics, dict) and 'loss' in test_metrics:
                    wandb_run.summary['post_training_generation_test_loss'] = float(test_metrics['loss'])
        return summary
    finally:
        if wandb.run is not None:
            wandb.finish()


def main() -> None:
    args = build_arg_parser().parse_args()
    run_downstream_common(args)


if __name__ == '__main__':
    main()
