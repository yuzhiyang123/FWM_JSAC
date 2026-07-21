from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from twm.datasets import build_time_band_channel_three_way_dataloaders
from .pipeline import DualScaleMaskGenerator
from .pipeline_ema32 import TimeBand128x64EMA32JEPAOutput, TimeBand128x64EMA32JEPAPipeline
from .training import (
    JEPA128x64TrainingConfig,
    LossBreakdown,
    _cov_penalty,
    _jepa_l1,
    _jepa_nmse,
    _loss_weights,
    _mean_metric,
    _resolve_model_config,
    build_warmup_cosine_scheduler,
)


@dataclass
class JEPA128x64EMA32TrainingConfig(JEPA128x64TrainingConfig):
    teacher_momentum: float = 0.95
    wandb_project: str = 'twm_jepa_128x64_ema32'

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> 'JEPA128x64EMA32TrainingConfig':
        payload = vars(args).copy()
        payload['dataset_file'] = [str(item) for item in payload.get('dataset_file', [])]
        payload['pretrained_32x16_dir'] = str(payload.get('pretrained_32x16_dir', ''))
        payload['working_dir'] = str(payload.get('working_dir', ''))
        return cls(**payload)


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = JEPA128x64EMA32TrainingConfig()
    parser = argparse.ArgumentParser(description='Train the dual-scale 128x64 JEPA pipeline with a trainable 32x16 student and EMA 32x16 teacher.')
    parser.add_argument('--dataset-file', nargs='+', required=True)
    parser.add_argument('--pretrained-32x16-dir', required=True)
    parser.add_argument('--working-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=defaults.batch_size)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument('--lr', type=float, default=defaults.lr)
    parser.add_argument('--epochs', type=int, default=defaults.epochs)
    parser.add_argument('--num-workers', type=int, default=defaults.num_workers)
    parser.add_argument('--window-length', type=int, default=defaults.window_length)
    parser.add_argument('--random-windows-per-sequence', type=int, default=defaults.random_windows_per_sequence)
    parser.add_argument('--train-paths', type=int, default=defaults.train_paths)
    parser.add_argument('--test-paths', type=int, default=defaults.test_paths)
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--device', default=defaults.device)
    parser.add_argument('--weight-decay', type=float, default=defaults.weight_decay)
    parser.add_argument('--final-min-mask-rate', type=float, default=defaults.final_min_mask_rate)
    parser.add_argument('--final-max-mask-rate', type=float, default=defaults.final_max_mask_rate)
    parser.add_argument('--final-beta-alpha', type=float, default=defaults.final_beta_alpha)
    parser.add_argument('--final-beta-beta', type=float, default=defaults.final_beta_beta)
    parser.add_argument('--known-split-prob', type=float, default=defaults.known_split_prob)
    parser.add_argument('--alignment-warmup-ratio', type=float, default=defaults.alignment_warmup_ratio)
    parser.add_argument('--cov-fro-weight', type=float, default=defaults.cov_fro_weight)
    parser.add_argument('--teacher-momentum', type=float, default=defaults.teacher_momentum)
    parser.add_argument('--tokenizer-128x64-variant', choices=['auto', 's', 'm', 'l'], default=defaults.tokenizer_128x64_variant)
    parser.add_argument('--tokenizer-output-norm-128x64', choices=['auto', 'batchnorm', 'layernorm', 'none'], default=defaults.tokenizer_output_norm_128x64)
    parser.add_argument('--save-every', type=int, default=defaults.save_every)
    parser.add_argument('--wandb-project', default=defaults.wandb_project)
    parser.add_argument('--wandb-run-name', default=defaults.wandb_run_name)
    return parser


def _alignment_mse(output: TimeBand128x64EMA32JEPAOutput) -> Tensor:
    return F.mse_loss(output.latent_128x64_all, output.latent_32x16_teacher_all)


def _compute_loss(output: TimeBand128x64EMA32JEPAOutput, *, cov_fro_weight: float, step: int, total_steps: int, warmup_steps: int) -> LossBreakdown:
    jepa_l1 = _jepa_l1(output)
    cov = _cov_penalty(output)
    alignment = _alignment_mse(output)
    w_align, w_jepa = _loss_weights(step, total_steps, warmup_steps)
    jepa_branch = jepa_l1 + float(cov_fro_weight) * cov
    total = float(w_align) * alignment + float(w_jepa) * jepa_branch
    return LossBreakdown(
        total_loss=total,
        components={
            'alignment_mse': alignment.detach(),
            'jepa_l1': jepa_l1.detach(),
            'latent_all_offdiag_cov_fro': cov.detach(),
            'jepa_nmse': _jepa_nmse(output).detach(),
            'loss_weight_alignment': output.latent_predicted.new_tensor(w_align),
            'loss_weight_jepa': output.latent_predicted.new_tensor(w_jepa),
        },
    )


def train_batch(model: TimeBand128x64EMA32JEPAPipeline, batch_x: Tensor, optimizer: torch.optim.Optimizer, accelerator: Accelerator, *, cov_fro_weight: float, teacher_momentum: float, step: int, total_steps: int, warmup_steps: int, scheduler: torch.optim.lr_scheduler._LRScheduler | None) -> tuple[TimeBand128x64EMA32JEPAOutput, LossBreakdown, bool]:
    model.train()
    with accelerator.accumulate(model):
        output = model(batch_x)
        loss = _compute_loss(output, cov_fro_weight=cov_fro_weight, step=step, total_steps=total_steps, warmup_steps=warmup_steps)
        accelerator.backward(loss.total_loss)
        optimizer.step()
        did_step = bool(accelerator.sync_gradients)
        if did_step and step >= warmup_steps:
            accelerator.unwrap_model(model).synchronize_teacher(momentum=teacher_momentum)
        if did_step and scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return output, loss, did_step


def evaluate_batch(model: TimeBand128x64EMA32JEPAPipeline, batch_x: Tensor, *, cov_fro_weight: float, step: int, total_steps: int, warmup_steps: int) -> tuple[TimeBand128x64EMA32JEPAOutput, LossBreakdown]:
    model.eval()
    with torch.no_grad():
        output = model(batch_x)
        loss = _compute_loss(output, cov_fro_weight=cov_fro_weight, step=step, total_steps=total_steps, warmup_steps=warmup_steps)
    return output, loss


def run_jepa_128x64_ema32_training(args: argparse.Namespace | JEPA128x64EMA32TrainingConfig) -> dict[str, object]:
    config = args if isinstance(args, JEPA128x64EMA32TrainingConfig) else JEPA128x64EMA32TrainingConfig.from_namespace(args)
    working_dir = Path(config.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    accelerator = Accelerator(log_with='wandb', gradient_accumulation_steps=config.gradient_accumulation_steps)
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
    model_config, resolved_meta = _resolve_model_config(config)
    mask_generator = DualScaleMaskGenerator(
        final_min_mask_rate=config.final_min_mask_rate,
        final_max_mask_rate=config.final_max_mask_rate,
        final_beta_alpha=config.final_beta_alpha,
        final_beta_beta=config.final_beta_beta,
        known_split_prob=config.known_split_prob,
    )
    model = TimeBand128x64EMA32JEPAPipeline(mask_generator=mask_generator, config=model_config)
    model.load_pretrained_32x16_core(config.pretrained_32x16_dir)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=config.weight_decay)
    updates_per_epoch = max(1, (len(train_loader) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps)
    total_steps = max(1, config.epochs * updates_per_epoch)
    warmup_steps = max(1, int(total_steps * config.alignment_warmup_ratio))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_ratio=config.alignment_warmup_ratio)

    model, optimizer, train_loader, test_loader, cross_test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader, cross_test_loader)

    config_payload = config.to_dict()
    config_payload['resolved_model_config'] = {
        'num_low_bands': model_config.num_low_bands,
        'predictor_dim': model_config.predictor_dim,
        'tokenizer_name_32x16_3p5': model_config.tokenizer_name_32x16_3p5,
        'tokenizer_name_32x16_28': model_config.tokenizer_name_32x16_28,
        'tokenizer_name_128x64_3p5': model_config.tokenizer_name_128x64_3p5,
        'tokenizer_name_128x64_28': model_config.tokenizer_name_128x64_28,
        'num_tokens': model_config.num_tokens,
        'token_dim': model_config.token_dim,
        'tokenizer_output_norm_32x16': model_config.tokenizer_output_norm_32x16,
        'tokenizer_output_norm_128x64': model_config.tokenizer_output_norm_128x64,
        'encoder': asdict(model_config.encoder),
        'predictor': asdict(model_config.predictor),
    }
    config_payload['resolved_meta'] = resolved_meta
    config_payload['gradient_accumulation_steps'] = config.gradient_accumulation_steps
    config_payload['total_steps'] = total_steps
    config_payload['warmup_steps'] = warmup_steps
    if accelerator.is_main_process:
        (working_dir / 'run_config.json').write_text(json.dumps(config_payload, indent=2) + '\n', encoding='utf-8')
    accelerator.init_trackers(project_name=config.wandb_project, config=config_payload, init_kwargs={'wandb': {'name': config.wandb_run_name or working_dir.name, 'dir': str(working_dir)}})

    history: list[dict[str, float]] = []
    best_test_loss = None
    best_epoch = None
    global_step = 0
    for epoch in range(1, config.epochs + 1):
        train_total_values = []
        train_component_values: dict[str, list[Tensor]] = {}
        for batch_x in train_loader:
            batch_x = batch_x.to(dtype=torch.float32)
            _, loss, did_step = train_batch(
                model,
                batch_x,
                optimizer,
                accelerator,
                cov_fro_weight=config.cov_fro_weight,
                teacher_momentum=config.teacher_momentum,
                step=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                scheduler=scheduler,
            )
            if did_step:
                global_step += 1
            train_total_values.append(loss.total_loss.detach())
            for name, value in loss.components.items():
                train_component_values.setdefault(name, []).append(value.detach())

        epoch_metrics = {
            'epoch': float(epoch),
            'train/total_loss': _mean_metric(accelerator, train_total_values),
            'train/lr': float(optimizer.param_groups[0]['lr']),
        }
        for name, values in train_component_values.items():
            epoch_metrics[f'train/{name}'] = _mean_metric(accelerator, values)

        if epoch % 5 == 0:
            test_total_values = []
            test_component_values: dict[str, list[Tensor]] = {}
            eval_step = min(global_step, total_steps - 1)
            for batch_x in test_loader:
                batch_x = batch_x.to(dtype=torch.float32)
                _, loss = evaluate_batch(model, batch_x, cov_fro_weight=config.cov_fro_weight, step=eval_step, total_steps=total_steps, warmup_steps=warmup_steps)
                test_total_values.append(loss.total_loss.detach())
                for name, value in loss.components.items():
                    test_component_values.setdefault(name, []).append(value.detach())
            epoch_metrics['test/total_loss'] = _mean_metric(accelerator, test_total_values)
            for name, values in test_component_values.items():
                epoch_metrics[f'test/{name}'] = _mean_metric(accelerator, values)

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

    summary = {'best_epoch': best_epoch, 'best_test_loss': best_test_loss, 'history': history}
    if accelerator.is_main_process:
        (working_dir / 'results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        accelerator.unwrap_model(model).save(working_dir / 'last_model')
    accelerator.end_training()
    return summary
