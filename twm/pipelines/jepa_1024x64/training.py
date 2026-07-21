from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from twm.datasets import build_time_band_channel_three_way_dataloaders
from .pipeline import TriScaleMaskGenerator, TimeBand1024x64JEPAConfig, TimeBand1024x64JEPAOutput, TimeBand1024x64JEPAPipeline, TimeBandViTConfig

_TOKENIZER_MAP = {
    'small': ('32x16_s', '32x16_s'),
    'medium': ('32x16_m', '32x16_m'),
    'attn': ('32x16_a', '32x16_a'),
    'large': ('32x16_l', '32x16_l'),
}

_LAYER_MAP = {
    'shallow': {'depth': 2, 'num_heads': 4},
    'medium': {'depth': 4, 'num_heads': 8},
    'deep': {'depth': 6, 'num_heads': 8},
}

def _resolve_pretrained_architecture(jepa_cfg: dict[str, object]) -> dict[str, object]:
    parts = str(jepa_cfg['preset_name']).strip().lower().split('_')
    if len(parts) != 4 or parts[0] != '32x16':
        raise ValueError(f"Unsupported pretrained preset: {jepa_cfg['preset_name']!r}")
    _, tokenizer_size, encoder_size, predictor_size = parts
    tok_3p5, tok_28 = _TOKENIZER_MAP[tokenizer_size]
    return {'tokenizer_name_3p5': tok_3p5, 'tokenizer_name_28': tok_28, 'encoder': _LAYER_MAP[encoder_size], 'predictor': _LAYER_MAP[predictor_size]}

def _map_32x16_tokenizer_to_128x64(name: str) -> str:
    key = str(name).strip().lower()
    if key.endswith('_s'):
        return '128x64_s'
    if key.endswith('_m') or key.endswith('_a'):
        return '128x64_m'
    if key.endswith('_l'):
        return '128x64_l'
    raise ValueError(f'Unsupported 32x16 tokenizer name for 128x64 mapping: {name!r}')

@dataclass
class JEPA1024x64TrainingConfig:
    dataset_file: list[str] = field(default_factory=list)
    pretrained_32x16_dir: str = ''
    pretrained_128x64_dir: str = ''
    working_dir: str = ''
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    lr: float = 1e-4
    epochs: int = 200
    num_workers: int = 4
    window_length: int = 16
    random_windows_per_sequence: int = 1
    train_paths: int = 466
    test_paths: int = 83
    seed: int = 42
    device: str = 'cuda'
    weight_decay: float = 1e-4
    final_min_mask_rate: float = 0.1
    final_max_mask_rate: float = 0.9
    final_beta_alpha: float = 4.0
    final_beta_beta: float = 1.2
    alignment_warmup_ratio: float = 0.1
    cov_fro_weight: float = 0.1
    reference_domain: str = '32x16'
    save_every: int = 20
    wandb_project: str = 'twm_jepa_1024x64_main'
    wandb_run_name: str = ''

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> 'JEPA1024x64TrainingConfig':
        payload = vars(args).copy()
        payload['dataset_file'] = [str(item) for item in payload.get('dataset_file', [])]
        payload['pretrained_32x16_dir'] = str(payload.get('pretrained_32x16_dir', ''))
        payload['pretrained_128x64_dir'] = str(payload.get('pretrained_128x64_dir', ''))
        payload['working_dir'] = str(payload.get('working_dir', ''))
        return cls(**payload)
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = JEPA1024x64TrainingConfig()
    parser = argparse.ArgumentParser(description='Train the tri-scale 1024x64 JEPA pipeline initialized from pretrained 32x16 and 128x64 runs.')
    parser.add_argument('--dataset-file', nargs='+', required=True)
    parser.add_argument('--pretrained-32x16-dir', required=True)
    parser.add_argument('--pretrained-128x64-dir', required=True)
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
    parser.add_argument('--alignment-warmup-ratio', type=float, default=defaults.alignment_warmup_ratio)
    parser.add_argument('--cov-fro-weight', type=float, default=defaults.cov_fro_weight)
    parser.add_argument('--reference-domain', choices=['32x16', '128x64'], default=defaults.reference_domain)
    parser.add_argument('--save-every', type=int, default=defaults.save_every)
    parser.add_argument('--wandb-project', default=defaults.wandb_project)
    parser.add_argument('--wandb-run-name', default=defaults.wandb_run_name)
    return parser

@dataclass
class LossBreakdown:
    total_loss: Tensor
    components: dict[str, Tensor]

def _resolve_model_config(config: JEPA1024x64TrainingConfig) -> tuple[TimeBand1024x64JEPAConfig, dict[str, object]]:
    path32 = Path(config.pretrained_32x16_dir)
    meta32 = path32 if (path32/'run_config.json').exists() else path32.parent
    payload32 = json.loads((meta32/'run_config.json').read_text(encoding='utf-8'))
    arch32 = _resolve_pretrained_architecture(payload32['jepa_config'])
    path128 = Path(config.pretrained_128x64_dir)
    meta128 = path128 if (path128/'run_config.json').exists() else path128.parent
    payload128 = json.loads((meta128/'run_config.json').read_text(encoding='utf-8'))
    resolved128 = payload128['resolved_model_config']
    model_config = TimeBand1024x64JEPAConfig(
        num_low_bands=int(payload32['jepa_config']['num_low_bands']),
        predictor_dim=int(resolved128['predictor_dim']),
        preprocessor_name_32x16='uniform_grid_32x16_quadrant',
        preprocessor_name_128x64='uniform_grid_128x64',
        preprocessor_name_1024x64='identity_1024x64',
        tokenizer_name_32x16_3p5=str(arch32['tokenizer_name_3p5']),
        tokenizer_name_32x16_28=str(arch32['tokenizer_name_28']),
        tokenizer_name_128x64_3p5=str(resolved128['tokenizer_name_128x64_3p5']),
        tokenizer_name_128x64_28=str(resolved128['tokenizer_name_128x64_28']),
        tokenizer_name_1024x64_3p5='1024x64_xl',
        tokenizer_name_1024x64_28='1024x64_xl',
        num_tokens=int(resolved128['num_tokens']),
        token_dim=int(resolved128['token_dim']),
        tokenizer_output_norm_32x16='layernorm',
        tokenizer_output_norm_128x64=str(resolved128['tokenizer_output_norm_128x64']),
        tokenizer_output_norm_1024x64='layernorm',
        reference_domain=str(config.reference_domain),
        encoder=TimeBandViTConfig(**resolved128['encoder']),
        predictor=TimeBandViTConfig(**resolved128['predictor']),
    )
    meta = {'pretrained_32x16_jepa_config': payload32['jepa_config'], 'pretrained_128x64_resolved_model_config': resolved128}
    return model_config, meta

def build_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float = 0.1):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def _loss_weights(step: int, total_steps: int, warmup_steps: int) -> tuple[float, float]:
    if step < warmup_steps:
        return 1.0, 0.0
    if total_steps <= warmup_steps:
        return 0.0, 1.0
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
    progress = min(max(progress, 0.0), 1.0)
    w_align = 0.5 * (1.0 + math.cos(math.pi * progress))
    return w_align, 1.0 - w_align

def _jepa_l1(output: TimeBand1024x64JEPAOutput) -> Tensor:
    mask = output.latent_masked_mask[:, :, None, None]
    if not mask.any():
        return output.latent_predicted.sum() * 0.0
    return (output.latent_predicted - output.latent_masked).abs().masked_select(mask).mean()

def _alignment_mse(output: TimeBand1024x64JEPAOutput, reference_domain: str) -> Tensor:
    key = str(reference_domain).strip().lower()
    reference = output.latent_32x16_all if key == '32x16' else output.latent_128x64_all
    return F.mse_loss(output.latent_1024x64_all, reference)

def _cov_penalty(output: TimeBand1024x64JEPAOutput) -> Tensor:
    latent = output.latent_all
    valid_mask = output.latent_all_mask
    flattened = latent.flatten(start_dim=2)
    values = []
    for slot_idx in range(flattened.shape[1]):
        slot_values = flattened[:, slot_idx, :]
        slot_mask = valid_mask[:, slot_idx]
        valid_slot_values = slot_values[slot_mask]
        if valid_slot_values.shape[0] <= 1:
            continue
        centered = valid_slot_values - valid_slot_values.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / float(valid_slot_values.shape[0])
        offdiag = covariance - torch.diag(torch.diagonal(covariance))
        values.append(offdiag.pow(2).mean())
    if not values:
        return latent.sum() * 0.0
    return torch.stack(values).mean()

def _jepa_nmse(output: TimeBand1024x64JEPAOutput, eps: float = 1e-8) -> Tensor:
    mask = output.latent_masked_mask[:, :, None, None]
    if not mask.any():
        return output.latent_predicted.sum() * 0.0
    diff_sq = (output.latent_predicted - output.latent_masked).pow(2).masked_select(mask).mean()
    target_sq = output.latent_masked.pow(2).masked_select(mask).mean().clamp_min(eps)
    return diff_sq / target_sq

def _mean_metric(accelerator: Accelerator, values: list[Tensor]) -> float:
    if not values:
        return float('nan')
    stacked = torch.stack([value.detach() for value in values])
    gathered = accelerator.gather(stacked.mean()[None])
    return float(gathered.mean().item())

def _compute_loss(output: TimeBand1024x64JEPAOutput, *, cov_fro_weight: float, reference_domain: str, step: int, total_steps: int, warmup_steps: int) -> LossBreakdown:
    jepa_l1 = _jepa_l1(output)
    cov = _cov_penalty(output)
    alignment = _alignment_mse(output, reference_domain)
    w_align, w_jepa = _loss_weights(step, total_steps, warmup_steps)
    jepa_branch = jepa_l1 + float(cov_fro_weight) * cov
    total = float(w_align) * alignment + float(w_jepa) * jepa_branch
    return LossBreakdown(total_loss=total, components={'alignment_mse': alignment.detach(), 'jepa_l1': jepa_l1.detach(), 'latent_all_offdiag_cov_fro': cov.detach(), 'jepa_nmse': _jepa_nmse(output).detach(), 'loss_weight_alignment': output.latent_predicted.new_tensor(w_align), 'loss_weight_jepa': output.latent_predicted.new_tensor(w_jepa)})

def train_batch(model: TimeBand1024x64JEPAPipeline, batch_x: Tensor, optimizer: torch.optim.Optimizer, accelerator: Accelerator, *, cov_fro_weight: float, reference_domain: str, step: int, total_steps: int, warmup_steps: int, scheduler: torch.optim.lr_scheduler._LRScheduler | None):
    model.train()
    with accelerator.accumulate(model):
        output = model(batch_x)
        loss = _compute_loss(output, cov_fro_weight=cov_fro_weight, reference_domain=reference_domain, step=step, total_steps=total_steps, warmup_steps=warmup_steps)
        accelerator.backward(loss.total_loss)
        optimizer.step()
        did_step = bool(accelerator.sync_gradients)
        if did_step and scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    model.freeze_pretrained_branches()
    return output, loss, did_step

def evaluate_batch(model: TimeBand1024x64JEPAPipeline, batch_x: Tensor, *, cov_fro_weight: float, reference_domain: str, step: int, total_steps: int, warmup_steps: int):
    model.eval()
    with torch.no_grad():
        output = model(batch_x)
        loss = _compute_loss(output, cov_fro_weight=cov_fro_weight, reference_domain=reference_domain, step=step, total_steps=total_steps, warmup_steps=warmup_steps)
    return output, loss

def run_jepa_1024x64_training(args: argparse.Namespace | JEPA1024x64TrainingConfig) -> dict[str, object]:
    config = args if isinstance(args, JEPA1024x64TrainingConfig) else JEPA1024x64TrainingConfig.from_namespace(args)
    working_dir = Path(config.working_dir); working_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    accelerator = Accelerator(log_with='wandb', gradient_accumulation_steps=config.gradient_accumulation_steps)
    train_loader, test_loader, cross_test_loader = build_time_band_channel_three_way_dataloaders(dataset_file=config.dataset_file, batch_size=config.batch_size, window_length=config.window_length, random_windows_per_sequence=config.random_windows_per_sequence, num_workers=config.num_workers, seed=config.seed, train_paths=config.train_paths, test_paths=config.test_paths)
    model_config, resolved_meta = _resolve_model_config(config)
    model = TimeBand1024x64JEPAPipeline(mask_generator=TriScaleMaskGenerator(final_min_mask_rate=config.final_min_mask_rate, final_max_mask_rate=config.final_max_mask_rate, final_beta_alpha=config.final_beta_alpha, final_beta_beta=config.final_beta_beta), config=model_config)
    model.load_pretrained_cores(config.pretrained_32x16_dir, config.pretrained_128x64_dir)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=config.weight_decay)
    updates_per_epoch = max(1, (len(train_loader) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps)
    total_steps = max(1, config.epochs * updates_per_epoch)
    warmup_steps = max(1, int(total_steps * config.alignment_warmup_ratio))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_ratio=config.alignment_warmup_ratio)
    model, optimizer, train_loader, test_loader, cross_test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader, cross_test_loader)
    payload = config.to_dict(); payload['resolved_model_config'] = asdict(model_config); payload['resolved_meta'] = resolved_meta; payload['total_steps']=total_steps; payload['warmup_steps']=warmup_steps
    if accelerator.is_main_process: (working_dir/'run_config.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    accelerator.init_trackers(project_name=config.wandb_project, config=payload, init_kwargs={'wandb': {'name': config.wandb_run_name or working_dir.name, 'dir': str(working_dir)}})
    history=[]; best_test_loss=None; best_epoch=None; global_step=0
    for epoch in range(1, config.epochs+1):
        train_total_values=[]; train_component_values={}
        for batch_x in train_loader:
            batch_x=batch_x.to(dtype=torch.float32)
            _, loss, did_step = train_batch(model,batch_x,optimizer,accelerator,cov_fro_weight=config.cov_fro_weight,reference_domain=config.reference_domain,step=global_step,total_steps=total_steps,warmup_steps=warmup_steps,scheduler=scheduler)
            if did_step: global_step += 1
            train_total_values.append(loss.total_loss.detach())
            for name, value in loss.components.items(): train_component_values.setdefault(name, []).append(value.detach())
        epoch_metrics={'epoch': float(epoch), 'train/total_loss': _mean_metric(accelerator, train_total_values), 'train/lr': float(optimizer.param_groups[0]['lr'])}
        for name, values in train_component_values.items(): epoch_metrics[f'train/{name}']=_mean_metric(accelerator, values)
        if epoch % 5 == 0:
            test_total_values=[]; test_component_values={}; eval_step=min(global_step, total_steps-1)
            for batch_x in test_loader:
                batch_x=batch_x.to(dtype=torch.float32)
                _, loss = evaluate_batch(model, batch_x, cov_fro_weight=config.cov_fro_weight, reference_domain=config.reference_domain, step=eval_step, total_steps=total_steps, warmup_steps=warmup_steps)
                test_total_values.append(loss.total_loss.detach())
                for name, value in loss.components.items(): test_component_values.setdefault(name, []).append(value.detach())
            epoch_metrics['test/total_loss'] = _mean_metric(accelerator, test_total_values)
            for name, values in test_component_values.items(): epoch_metrics[f'test/{name}'] = _mean_metric(accelerator, values)
        history.append(epoch_metrics); accelerator.log(epoch_metrics, step=epoch)
        current_test_loss = epoch_metrics.get('test/total_loss'); unwrapped_model = accelerator.unwrap_model(model)
        if current_test_loss is not None and (best_test_loss is None or current_test_loss < best_test_loss):
            best_test_loss=current_test_loss; best_epoch=epoch
            if accelerator.is_main_process: unwrapped_model.save(working_dir/'best_model')
        if config.save_every > 0 and epoch % config.save_every == 0 and accelerator.is_main_process: unwrapped_model.save(working_dir/f'checkpoint_epoch_{epoch:03d}')
    summary={'best_epoch': best_epoch, 'best_test_loss': best_test_loss, 'history': history}
    if accelerator.is_main_process:
        (working_dir/'results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        accelerator.unwrap_model(model).save(working_dir/'last_model')
    accelerator.end_training(); return summary
