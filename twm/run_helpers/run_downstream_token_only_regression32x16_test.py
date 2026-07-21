from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

from twm.downstream.channel_pred import ChannelPredConfig
from twm.downstream.channel_pred.training import _latent_metric_triplet
from twm.downstream.channel_pred_regression32x16 import build_channel_pred_token_only_three_way_dataloaders
from twm.downstream.channel_pred_regression32x16 import ChannelPredRegressionTask
from twm.run_helpers.run_downstream_common import ChannelPredTrainingConfig, _build_backbone, _move_batch_to_device, _mean

FIXED_WINDOW_LENGTH = 16
FIXED_RANDOM_WINDOWS_PER_SEQUENCE = 30


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sweep fixed pilot-noise NMSE for a token-only downstream checkpoint, comparing tokenizer-target and JEPA-predicted latent recovery.')
    parser.add_argument('--dataset-file', nargs='+', required=True)
    parser.add_argument('--trained-run-dir', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--checkpoint-name', default='best_model.pt')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--train-paths', type=int, default=466)
    parser.add_argument('--test-paths', type=int, default=83)
    return parser


def _load_base_config(trained_run_dir: Path) -> ChannelPredTrainingConfig:
    payload = json.loads((trained_run_dir / 'run_config.json').read_text(encoding='utf-8'))
    allowed = {field.name for field in fields(ChannelPredTrainingConfig)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return ChannelPredTrainingConfig(**filtered)


def _evaluate_loader_compare(task: ChannelPredRegressionTask, loader, device: torch.device) -> dict[str, float]:
    task.eval()
    target_recovery_mse_values: list[float] = []
    prediction_recovery_mse_values: list[float] = []
    prediction_target_latent_nmse_values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            target_source = batch.current_gt if batch.current_gt is not None else batch.current_channel
            current_preprocessed, target_preprocessed = task.preprocess_channel_pair(batch.current_channel, target_source)
            target_latents, _ = task.encode_channels(batch.current_channel)
            predicted_latents = task._predict_current_latents(batch.history_channel, batch.history_mask, batch.current_channel)
            _, _, prediction_target_latent_nmse = _latent_metric_triplet(predicted_latents, target_latents)

            if batch.subband_mask is None:
                effective_mask = torch.ones(target_latents.shape[:2], dtype=torch.bool, device=target_latents.device)
            else:
                effective_mask = batch.subband_mask.to(dtype=torch.bool)
            pilot_estimate = task._build_pilot_estimate_from_preprocessed(current_preprocessed)
            pilot_estimate = pilot_estimate * effective_mask[..., None, None, None].to(dtype=pilot_estimate.dtype)

            flat_target = target_preprocessed.reshape(-1, *target_preprocessed.shape[-3:])
            flat_pilot = pilot_estimate.reshape(-1, *pilot_estimate.shape[-3:])[:, :2]

            target_condition_tokens = target_latents.reshape(-1, 1, target_latents.shape[-1])
            target_recovered = task.regressor(flat_pilot, target_condition_tokens)
            target_recovery_mse = F.mse_loss(target_recovered, flat_target)

            prediction_condition_tokens = predicted_latents.reshape(-1, 1, predicted_latents.shape[-1])
            prediction_recovered = task.regressor(flat_pilot, prediction_condition_tokens)
            prediction_recovery_mse = F.mse_loss(prediction_recovered, flat_target)

            target_recovery_mse_values.append(float(target_recovery_mse.detach().item()))
            prediction_recovery_mse_values.append(float(prediction_recovery_mse.detach().item()))
            prediction_target_latent_nmse_values.append(float(prediction_target_latent_nmse.detach().item()))

    return {
        'target_recovery_mse': _mean(target_recovery_mse_values),
        'prediction_recovery_mse': _mean(prediction_recovery_mse_values),
        'prediction_target_latent_nmse': _mean(prediction_target_latent_nmse_values),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    trained_run_dir = Path(args.trained_run_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    config = _load_base_config(trained_run_dir)
    config.dataset_file = [str(x) for x in args.dataset_file]
    config.device = str(args.device)
    config.train_paths = int(args.train_paths)
    config.test_paths = int(args.test_paths)
    config.window_length = FIXED_WINDOW_LENGTH
    config.random_windows_per_sequence = FIXED_RANDOM_WINDOWS_PER_SEQUENCE
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        config.num_workers = int(args.num_workers)

    _, test_loader, _ = build_channel_pred_token_only_three_way_dataloaders(
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
    task = ChannelPredRegressionTask.from_stage2_checkpoint(
        backbone,
        pretrained_dir=config.pretrained_dir,
        config=ChannelPredConfig(
            pilot_align_mode='none',
            channel_mode='complex',
            pilot_est_subcarrier_strides=tuple(config.pilot_est_subcarrier_strides),
            pilot_est_noise_mode='fixed_nmse',
            pilot_est_max_noise_std=config.pilot_est_max_noise_std,
            pilot_input_noise_var=0.0,
            pilot_total_noise_var_target=0.0,
            pilot_fixed_nmse=0.0,
            fusion_mode=config.fusion_mode,
        ),
        freeze_backbone=True,
    )
    state_dict = torch.load(trained_run_dir / args.checkpoint_name, map_location='cpu')
    task.load_state_dict(state_dict)
    task = task.to(device)

    fieldnames = ['trained_run_dir', 'method', 'fixed_nmse', 'target_recovery_mse', 'prediction_recovery_mse', 'prediction_target_latent_nmse']
    rows = []
    for k in range(11):
        nmse = k / 10.0
        task.config.pilot_est_noise_mode = 'fixed_nmse'
        task.config.pilot_fixed_nmse = nmse
        task.config.pilot_input_noise_var = 0.0
        task.config.pilot_total_noise_var_target = 0.0
        metrics = _evaluate_loader_compare(task, test_loader, device)
        rows.append({
            'trained_run_dir': str(trained_run_dir),
            'method': 'token_only_compare',
            'fixed_nmse': nmse,
            'target_recovery_mse': metrics['target_recovery_mse'],
            'prediction_recovery_mse': metrics['prediction_recovery_mse'],
            'prediction_target_latent_nmse': metrics['prediction_target_latent_nmse'],
        })

    with output_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {output_csv}')


if __name__ == '__main__':
    main()
