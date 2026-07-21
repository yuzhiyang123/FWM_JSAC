from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor, nn

from twm.datasets import build_time_band_channel_three_way_dataloaders
from twm.pipelines.jepa_32x16.default_pipeline_configs import _LAYER_MAP, _TOKENIZER_MAP
from twm.pipelines.jepa_32x16.time_freq_band import CurriculumRandomMaskExceptLastGenerator
from .pipeline import TimeBand1024x64JEPAConfig, TimeBandViTConfig, TriScaleMaskSplit
from .pipeline_ema import TimeBand1024x64EMAJEPAOutput, TimeBand1024x64EMAJEPAPipeline
from .training import LossBreakdown, _cov_penalty, _jepa_l1, _jepa_nmse, _loss_weights, _map_32x16_tokenizer_to_128x64, _mean_metric, build_warmup_cosine_scheduler

@dataclass
class JEPA1024x64EMAScratchTrainingConfig:
    dataset_file: list[str]
    working_dir: str
    jepa_config_name: str = '32x16_medium_medium_shallow'
    num_low_bands: int = 5
    num_tokens: int = 4
    token_dim: int = 192
    predictor_dim: int = 192
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    lr: float = 1e-4
    epochs: int = 400
    num_workers: int = 4
    window_length: int = 16
    random_windows_per_sequence: int = 1
    train_paths: int = 466
    test_paths: int = 83
    seed: int = 42
    device: str = 'cuda'
    weight_decay: float = 1e-4
    curriculum_start_mask_rate: float = 0.3
    final_min_mask_rate: float = 0.1
    final_max_mask_rate: float = 0.9
    final_beta_alpha: float = 4.0
    final_beta_beta: float = 1.2
    alignment_warmup_ratio: float = 0.1
    cov_fro_weight: float = 0.1
    teacher_momentum: float = 0.95
    tokenizer_output_norm_32x16: str = 'layernorm'
    tokenizer_output_norm_128x64: str = 'layernorm'
    tokenizer_output_norm_1024x64: str = 'layernorm'
    save_every: int = 20
    wandb_project: str = 'twm_jepa_1024x64_ema_scratch'
    wandb_run_name: str = ''
    @classmethod
    def from_namespace(cls, args: argparse.Namespace):
        payload = vars(args).copy(); payload['dataset_file']=[str(item) for item in payload.get('dataset_file', [])]; payload['working_dir']=str(payload.get('working_dir', '')); return cls(**payload)
    def to_dict(self): return asdict(self)

class CurriculumTriScaleMaskGenerator(nn.Module):
    def __init__(self, *, start_mask_rate: float = 0.3, final_min_mask_rate: float = 0.1, final_max_mask_rate: float = 0.9, final_beta_alpha: float = 4.0, final_beta_beta: float = 1.2) -> None:
        super().__init__()
        self.base_generator = CurriculumRandomMaskExceptLastGenerator(start_mask_rate=float(start_mask_rate), end_mask_rate=float(final_max_mask_rate), final_min_mask_rate=float(final_min_mask_rate), final_max_mask_rate=float(final_max_mask_rate), final_beta_alpha=float(final_beta_alpha), final_beta_beta=float(final_beta_beta))
        self.last_mask_rate = float(start_mask_rate)
    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.base_generator.set_epoch(epoch, total_epochs)
    def forward(self, x: Tensor):
        unknown = self.base_generator(x)
        self.last_mask_rate = float(self.base_generator.last_mask_rate)
        retained = ~unknown
        batch_size, time_steps, num_bands = unknown.shape
        for batch_idx in range(batch_size):
            if not retained[batch_idx].any():
                time_idx = int(torch.randint(0, max(1, time_steps - 1), (1,), device=x.device).item())
                band_idx = int(torch.randint(0, num_bands, (1,), device=x.device).item())
                unknown[batch_idx, time_idx, band_idx] = False
        retained = ~unknown
        split_random = torch.rand(batch_size, time_steps, num_bands, device=x.device)
        known_32x16 = retained & (split_random < (1.0/3.0))
        known_128x64 = retained & (split_random >= (1.0/3.0)) & (split_random < (2.0/3.0))
        known_1024x64 = retained & (split_random >= (2.0/3.0))
        return TriScaleMaskSplit(known_32x16=known_32x16, known_128x64=known_128x64, known_1024x64=known_1024x64, unknown=unknown)

def build_arg_parser() -> argparse.ArgumentParser:
    defaults = JEPA1024x64EMAScratchTrainingConfig(dataset_file=[], working_dir='')
    parser = argparse.ArgumentParser(description='Train the tri-scale 1024x64 JEPA pipeline from scratch with 32x16 and 128x64 EMA teachers active from step 0.')
    parser.add_argument('--dataset-file', nargs='+', required=True)
    parser.add_argument('--working-dir', required=True)
    parser.add_argument('--jepa-config-name', default=defaults.jepa_config_name)
    parser.add_argument('--num-low-bands', type=int, default=defaults.num_low_bands)
    parser.add_argument('--num-tokens', type=int, default=defaults.num_tokens)
    parser.add_argument('--token-dim', type=int, default=defaults.token_dim)
    parser.add_argument('--predictor-dim', type=int, default=defaults.predictor_dim)
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
    parser.add_argument('--curriculum-start-mask-rate', type=float, default=defaults.curriculum_start_mask_rate)
    parser.add_argument('--final-min-mask-rate', type=float, default=defaults.final_min_mask_rate)
    parser.add_argument('--final-max-mask-rate', type=float, default=defaults.final_max_mask_rate)
    parser.add_argument('--final-beta-alpha', type=float, default=defaults.final_beta_alpha)
    parser.add_argument('--final-beta-beta', type=float, default=defaults.final_beta_beta)
    parser.add_argument('--alignment-warmup-ratio', type=float, default=defaults.alignment_warmup_ratio)
    parser.add_argument('--cov-fro-weight', type=float, default=defaults.cov_fro_weight)
    parser.add_argument('--teacher-momentum', type=float, default=defaults.teacher_momentum)
    parser.add_argument('--tokenizer-output-norm-32x16', choices=['batchnorm', 'layernorm', 'none'], default=defaults.tokenizer_output_norm_32x16)
    parser.add_argument('--tokenizer-output-norm-128x64', choices=['auto', 'batchnorm', 'layernorm', 'none'], default=defaults.tokenizer_output_norm_128x64)
    parser.add_argument('--tokenizer-output-norm-1024x64', choices=['auto', 'batchnorm', 'layernorm', 'none'], default=defaults.tokenizer_output_norm_1024x64)
    parser.add_argument('--save-every', type=int, default=defaults.save_every)
    parser.add_argument('--wandb-project', default=defaults.wandb_project)
    parser.add_argument('--wandb-run-name', default=defaults.wandb_run_name)
    return parser

def _resolve_model_config(config: JEPA1024x64EMAScratchTrainingConfig) -> tuple[TimeBand1024x64JEPAConfig, dict[str, object]]:
    parts = str(config.jepa_config_name).strip().lower().split('_')
    _, tokenizer_size, encoder_size, predictor_size = parts
    tokenizer_name_32x16_3p5, tokenizer_name_32x16_28 = _TOKENIZER_MAP[tokenizer_size]
    tokenizer_name_128x64_3p5 = _map_32x16_tokenizer_to_128x64(tokenizer_name_32x16_3p5)
    tokenizer_name_128x64_28 = _map_32x16_tokenizer_to_128x64(tokenizer_name_32x16_28)
    output_norm_128 = config.tokenizer_output_norm_32x16 if config.tokenizer_output_norm_128x64 == 'auto' else config.tokenizer_output_norm_128x64
    output_norm_1024 = config.tokenizer_output_norm_32x16 if config.tokenizer_output_norm_1024x64 == 'auto' else config.tokenizer_output_norm_1024x64
    model_config = TimeBand1024x64JEPAConfig(num_low_bands=int(config.num_low_bands), predictor_dim=int(config.predictor_dim), preprocessor_name_32x16='uniform_grid_32x16_quadrant', preprocessor_name_128x64='uniform_grid_128x64', preprocessor_name_1024x64='identity_1024x64', tokenizer_name_32x16_3p5=tokenizer_name_32x16_3p5, tokenizer_name_32x16_28=tokenizer_name_32x16_28, tokenizer_name_128x64_3p5=tokenizer_name_128x64_3p5, tokenizer_name_128x64_28=tokenizer_name_128x64_28, tokenizer_name_1024x64_3p5='1024x64_xl', tokenizer_name_1024x64_28='1024x64_xl', num_tokens=int(config.num_tokens), token_dim=int(config.token_dim), tokenizer_output_norm_32x16=str(config.tokenizer_output_norm_32x16), tokenizer_output_norm_128x64=str(output_norm_128), tokenizer_output_norm_1024x64=str(output_norm_1024), encoder=TimeBandViTConfig(depth=int(_LAYER_MAP[encoder_size]['depth']), num_heads=int(_LAYER_MAP[encoder_size]['num_heads']), mlp_ratio=4.0, dropout=0.0), predictor=TimeBandViTConfig(depth=int(_LAYER_MAP[predictor_size]['depth']), num_heads=int(_LAYER_MAP[predictor_size]['num_heads']), mlp_ratio=4.0, dropout=0.0))
    meta={'resolved_tokenizer_output_norm_32x16': config.tokenizer_output_norm_32x16, 'resolved_tokenizer_output_norm_128x64': output_norm_128, 'resolved_tokenizer_output_norm_1024x64': output_norm_1024}
    return model_config, meta

def _alignment_mse(output: TimeBand1024x64EMAJEPAOutput) -> Tensor:
    return F.mse_loss(output.latent_1024x64_all, output.latent_32x16_teacher_all)

def _compute_loss(output: TimeBand1024x64EMAJEPAOutput, *, cov_fro_weight: float, step: int, total_steps: int, warmup_steps: int) -> LossBreakdown:
    jepa_l1=_jepa_l1(output); cov=_cov_penalty(output); alignment=_alignment_mse(output); w_align,w_jepa=_loss_weights(step,total_steps,warmup_steps); total=float(w_align)*alignment+float(w_jepa)*(jepa_l1+float(cov_fro_weight)*cov)
    return LossBreakdown(total_loss=total, components={'alignment_mse': alignment.detach(), 'jepa_l1': jepa_l1.detach(), 'latent_all_offdiag_cov_fro': cov.detach(), 'jepa_nmse': _jepa_nmse(output).detach(), 'loss_weight_alignment': output.latent_predicted.new_tensor(w_align), 'loss_weight_jepa': output.latent_predicted.new_tensor(w_jepa)})

def train_batch(model: TimeBand1024x64EMAJEPAPipeline, batch_x: Tensor, optimizer: torch.optim.Optimizer, accelerator: Accelerator, *, cov_fro_weight: float, teacher_momentum: float, step: int, total_steps: int, warmup_steps: int, scheduler):
    model.train()
    with accelerator.accumulate(model):
        output=model(batch_x); loss=_compute_loss(output,cov_fro_weight=cov_fro_weight,step=step,total_steps=total_steps,warmup_steps=warmup_steps); accelerator.backward(loss.total_loss); optimizer.step(); did_step=bool(accelerator.sync_gradients)
        if did_step: accelerator.unwrap_model(model).synchronize_teacher(momentum=teacher_momentum)
        if did_step and scheduler is not None: scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return output, loss, did_step

def evaluate_batch(model: TimeBand1024x64EMAJEPAPipeline, batch_x: Tensor, *, cov_fro_weight: float, step: int, total_steps: int, warmup_steps: int):
    model.eval();
    with torch.no_grad(): output=model(batch_x); loss=_compute_loss(output,cov_fro_weight=cov_fro_weight,step=step,total_steps=total_steps,warmup_steps=warmup_steps)
    return output, loss

def run_jepa_1024x64_ema_scratch_training(args: argparse.Namespace | JEPA1024x64EMAScratchTrainingConfig) -> dict[str, object]:
    config = args if isinstance(args, JEPA1024x64EMAScratchTrainingConfig) else JEPA1024x64EMAScratchTrainingConfig.from_namespace(args)
    working_dir=Path(config.working_dir); working_dir.mkdir(parents=True, exist_ok=True); torch.manual_seed(config.seed)
    accelerator=Accelerator(log_with='wandb', gradient_accumulation_steps=config.gradient_accumulation_steps)
    train_loader,test_loader,cross_test_loader=build_time_band_channel_three_way_dataloaders(dataset_file=config.dataset_file,batch_size=config.batch_size,window_length=config.window_length,random_windows_per_sequence=config.random_windows_per_sequence,num_workers=config.num_workers,seed=config.seed,train_paths=config.train_paths,test_paths=config.test_paths)
    model_config,resolved_meta=_resolve_model_config(config)
    mask_generator=CurriculumTriScaleMaskGenerator(start_mask_rate=config.curriculum_start_mask_rate, final_min_mask_rate=config.final_min_mask_rate, final_max_mask_rate=config.final_max_mask_rate, final_beta_alpha=config.final_beta_alpha, final_beta_beta=config.final_beta_beta)
    model=TimeBand1024x64EMAJEPAPipeline(mask_generator=mask_generator, config=model_config)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=config.weight_decay)
    updates_per_epoch=max(1,(len(train_loader)+config.gradient_accumulation_steps-1)//config.gradient_accumulation_steps); total_steps=max(1,config.epochs*updates_per_epoch); warmup_steps=max(1,int(total_steps*config.alignment_warmup_ratio)); scheduler=build_warmup_cosine_scheduler(optimizer,total_steps=total_steps,warmup_ratio=config.alignment_warmup_ratio)
    model,optimizer,train_loader,test_loader,cross_test_loader=accelerator.prepare(model,optimizer,train_loader,test_loader,cross_test_loader)
    payload=config.to_dict(); payload['resolved_model_config']=asdict(model_config); payload['resolved_meta']=resolved_meta; payload['total_steps']=total_steps; payload['warmup_steps']=warmup_steps
    if accelerator.is_main_process: (working_dir/'run_config.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    accelerator.init_trackers(project_name=config.wandb_project, config=payload, init_kwargs={'wandb': {'name': config.wandb_run_name or working_dir.name, 'dir': str(working_dir)}})
    history=[]; best_test_loss=None; best_epoch=None; global_step=0
    for epoch in range(1, config.epochs+1):
        if hasattr(mask_generator, 'set_epoch'): mask_generator.set_epoch(epoch, config.epochs)
        train_total_values=[]; train_component_values={}; train_mask_rates=[]
        for batch_x in train_loader:
            batch_x=batch_x.to(dtype=torch.float32)
            _, loss, did_step = train_batch(model,batch_x,optimizer,accelerator,cov_fro_weight=config.cov_fro_weight,teacher_momentum=config.teacher_momentum,step=global_step,total_steps=total_steps,warmup_steps=warmup_steps,scheduler=scheduler)
            if did_step: global_step += 1
            train_total_values.append(loss.total_loss.detach())
            for name, value in loss.components.items(): train_component_values.setdefault(name, []).append(value.detach())
            current_rate=getattr(accelerator.unwrap_model(model).mask_generator,'last_mask_rate',None)
            if current_rate is not None: train_mask_rates.append(float(current_rate))
        epoch_metrics={'epoch': float(epoch), 'train/total_loss': _mean_metric(accelerator, train_total_values), 'train/lr': float(optimizer.param_groups[0]['lr'])}
        if train_mask_rates: epoch_metrics['train/mask_rate']=float(sum(train_mask_rates)/len(train_mask_rates))
        for name, values in train_component_values.items(): epoch_metrics[f'train/{name}']=_mean_metric(accelerator, values)
        if epoch % 5 == 0:
            test_total_values=[]; test_component_values={}; test_mask_rates=[]; eval_step=min(global_step,total_steps-1)
            for batch_x in test_loader:
                batch_x=batch_x.to(dtype=torch.float32)
                _, loss = evaluate_batch(model,batch_x,cov_fro_weight=config.cov_fro_weight,step=eval_step,total_steps=total_steps,warmup_steps=warmup_steps)
                test_total_values.append(loss.total_loss.detach())
                for name, value in loss.components.items(): test_component_values.setdefault(name, []).append(value.detach())
                current_rate=getattr(accelerator.unwrap_model(model).mask_generator,'last_mask_rate',None)
                if current_rate is not None: test_mask_rates.append(float(current_rate))
            epoch_metrics['test/total_loss']=_mean_metric(accelerator, test_total_values)
            if test_mask_rates: epoch_metrics['test/mask_rate']=float(sum(test_mask_rates)/len(test_mask_rates))
            for name, values in test_component_values.items(): epoch_metrics[f'test/{name}']=_mean_metric(accelerator, values)
        history.append(epoch_metrics); accelerator.log(epoch_metrics, step=epoch)
        current_test_loss=epoch_metrics.get('test/total_loss'); unwrapped_model=accelerator.unwrap_model(model)
        if current_test_loss is not None and (best_test_loss is None or current_test_loss<best_test_loss):
            best_test_loss=current_test_loss; best_epoch=epoch
            if accelerator.is_main_process: unwrapped_model.save(working_dir/'best_model')
        if config.save_every>0 and epoch % config.save_every == 0 and accelerator.is_main_process: unwrapped_model.save(working_dir/f'checkpoint_epoch_{epoch:03d}')
    summary={'best_epoch': best_epoch, 'best_test_loss': best_test_loss, 'history': history}
    if accelerator.is_main_process:
        (working_dir/'results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        accelerator.unwrap_model(model).save(working_dir/'last_model')
    accelerator.end_training(); return summary
