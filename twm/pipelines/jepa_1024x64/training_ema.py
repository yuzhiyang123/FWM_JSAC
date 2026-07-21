from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from twm.datasets import build_time_band_channel_three_way_dataloaders
from .pipeline import TriScaleMaskGenerator
from .pipeline_ema import TimeBand1024x64EMAJEPAOutput, TimeBand1024x64EMAJEPAPipeline
from .training import JEPA1024x64TrainingConfig, LossBreakdown, _cov_penalty, _jepa_l1, _jepa_nmse, _loss_weights, _mean_metric, _resolve_model_config, build_warmup_cosine_scheduler

@dataclass
class JEPA1024x64EMATrainingConfig(JEPA1024x64TrainingConfig):
    teacher_momentum: float = 0.95
    wandb_project: str = 'twm_jepa_1024x64_ema'
    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> 'JEPA1024x64EMATrainingConfig':
        payload = vars(args).copy(); payload['dataset_file']=[str(item) for item in payload.get('dataset_file', [])]; payload['pretrained_32x16_dir']=str(payload.get('pretrained_32x16_dir','')); payload['pretrained_128x64_dir']=str(payload.get('pretrained_128x64_dir','')); payload['working_dir']=str(payload.get('working_dir','')); return cls(**payload)

def build_arg_parser() -> argparse.ArgumentParser:
    defaults = JEPA1024x64EMATrainingConfig()
    parser = argparse.ArgumentParser(description='Train the tri-scale 1024x64 JEPA pipeline with pretrained 32x16 and 128x64 students plus EMA teachers.')
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
    parser.add_argument('--teacher-momentum', type=float, default=defaults.teacher_momentum)
    parser.add_argument('--save-every', type=int, default=defaults.save_every)
    parser.add_argument('--wandb-project', default=defaults.wandb_project)
    parser.add_argument('--wandb-run-name', default=defaults.wandb_run_name)
    return parser

def _alignment_mse(output: TimeBand1024x64EMAJEPAOutput, reference_domain: str) -> Tensor:
    key = str(reference_domain).strip().lower()
    reference = output.latent_32x16_teacher_all if key == '32x16' else output.latent_128x64_teacher_all
    return F.mse_loss(output.latent_1024x64_all, reference)

def _compute_loss(output: TimeBand1024x64EMAJEPAOutput, *, cov_fro_weight: float, reference_domain: str, step: int, total_steps: int, warmup_steps: int) -> LossBreakdown:
    jepa_l1 = _jepa_l1(output)
    cov = _cov_penalty(output)
    alignment = _alignment_mse(output, reference_domain)
    w_align, w_jepa = _loss_weights(step, total_steps, warmup_steps)
    total = float(w_align) * alignment + float(w_jepa) * (jepa_l1 + float(cov_fro_weight) * cov)
    return LossBreakdown(total_loss=total, components={'alignment_mse': alignment.detach(), 'jepa_l1': jepa_l1.detach(), 'latent_all_offdiag_cov_fro': cov.detach(), 'jepa_nmse': _jepa_nmse(output).detach(), 'loss_weight_alignment': output.latent_predicted.new_tensor(w_align), 'loss_weight_jepa': output.latent_predicted.new_tensor(w_jepa)})

def train_batch(model: TimeBand1024x64EMAJEPAPipeline, batch_x: Tensor, optimizer: torch.optim.Optimizer, accelerator: Accelerator, *, cov_fro_weight: float, reference_domain: str, teacher_momentum: float, step: int, total_steps: int, warmup_steps: int, scheduler):
    model.train()
    with accelerator.accumulate(model):
        output=model(batch_x); loss=_compute_loss(output,cov_fro_weight=cov_fro_weight,reference_domain=reference_domain,step=step,total_steps=total_steps,warmup_steps=warmup_steps); accelerator.backward(loss.total_loss); optimizer.step(); did_step=bool(accelerator.sync_gradients)
        if did_step and step >= warmup_steps: accelerator.unwrap_model(model).synchronize_teacher(momentum=teacher_momentum)
        if did_step and scheduler is not None: scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return output, loss, did_step

def evaluate_batch(model: TimeBand1024x64EMAJEPAPipeline, batch_x: Tensor, *, cov_fro_weight: float, reference_domain: str, step: int, total_steps: int, warmup_steps: int):
    model.eval()
    with torch.no_grad(): output=model(batch_x); loss=_compute_loss(output,cov_fro_weight=cov_fro_weight,reference_domain=reference_domain,step=step,total_steps=total_steps,warmup_steps=warmup_steps)
    return output, loss

def run_jepa_1024x64_ema_training(args: argparse.Namespace | JEPA1024x64EMATrainingConfig) -> dict[str, object]:
    config = args if isinstance(args, JEPA1024x64EMATrainingConfig) else JEPA1024x64EMATrainingConfig.from_namespace(args)
    working_dir=Path(config.working_dir); working_dir.mkdir(parents=True, exist_ok=True); torch.manual_seed(config.seed)
    accelerator=Accelerator(log_with='wandb', gradient_accumulation_steps=config.gradient_accumulation_steps)
    train_loader,test_loader,cross_test_loader=build_time_band_channel_three_way_dataloaders(dataset_file=config.dataset_file,batch_size=config.batch_size,window_length=config.window_length,random_windows_per_sequence=config.random_windows_per_sequence,num_workers=config.num_workers,seed=config.seed,train_paths=config.train_paths,test_paths=config.test_paths)
    model_config,resolved_meta=_resolve_model_config(config)
    model=TimeBand1024x64EMAJEPAPipeline(mask_generator=TriScaleMaskGenerator(final_min_mask_rate=config.final_min_mask_rate, final_max_mask_rate=config.final_max_mask_rate, final_beta_alpha=config.final_beta_alpha, final_beta_beta=config.final_beta_beta), config=model_config)
    model.load_pretrained_cores(config.pretrained_32x16_dir, config.pretrained_128x64_dir)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=config.weight_decay)
    updates_per_epoch=max(1,(len(train_loader)+config.gradient_accumulation_steps-1)//config.gradient_accumulation_steps); total_steps=max(1,config.epochs*updates_per_epoch); warmup_steps=max(1,int(total_steps*config.alignment_warmup_ratio)); scheduler=build_warmup_cosine_scheduler(optimizer,total_steps=total_steps,warmup_ratio=config.alignment_warmup_ratio)
    model,optimizer,train_loader,test_loader,cross_test_loader=accelerator.prepare(model,optimizer,train_loader,test_loader,cross_test_loader)
    payload=config.to_dict(); payload['resolved_model_config']=asdict(model_config); payload['resolved_meta']=resolved_meta; payload['total_steps']=total_steps; payload['warmup_steps']=warmup_steps
    if accelerator.is_main_process: (working_dir/'run_config.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    accelerator.init_trackers(project_name=config.wandb_project, config=payload, init_kwargs={'wandb': {'name': config.wandb_run_name or working_dir.name, 'dir': str(working_dir)}})
    history=[]; best_test_loss=None; best_epoch=None; global_step=0
    for epoch in range(1, config.epochs+1):
        train_total_values=[]; train_component_values={}
        for batch_x in train_loader:
            batch_x=batch_x.to(dtype=torch.float32)
            _, loss, did_step = train_batch(model,batch_x,optimizer,accelerator,cov_fro_weight=config.cov_fro_weight,reference_domain=config.reference_domain,teacher_momentum=config.teacher_momentum,step=global_step,total_steps=total_steps,warmup_steps=warmup_steps,scheduler=scheduler)
            if did_step: global_step += 1
            train_total_values.append(loss.total_loss.detach())
            for name, value in loss.components.items(): train_component_values.setdefault(name, []).append(value.detach())
        epoch_metrics={'epoch': float(epoch), 'train/total_loss': _mean_metric(accelerator, train_total_values), 'train/lr': float(optimizer.param_groups[0]['lr'])}
        for name, values in train_component_values.items(): epoch_metrics[f'train/{name}']=_mean_metric(accelerator, values)
        if epoch % 5 == 0:
            test_total_values=[]; test_component_values={}; eval_step=min(global_step,total_steps-1)
            for batch_x in test_loader:
                batch_x=batch_x.to(dtype=torch.float32)
                _, loss = evaluate_batch(model,batch_x,cov_fro_weight=config.cov_fro_weight,reference_domain=config.reference_domain,step=eval_step,total_steps=total_steps,warmup_steps=warmup_steps)
                test_total_values.append(loss.total_loss.detach())
                for name, value in loss.components.items(): test_component_values.setdefault(name, []).append(value.detach())
            epoch_metrics['test/total_loss']=_mean_metric(accelerator, test_total_values)
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
