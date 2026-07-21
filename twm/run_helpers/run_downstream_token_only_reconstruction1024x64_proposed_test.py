from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

import torch

from twm.downstream.channel_pred import build_channel_pred_three_way_dataloaders
from twm.downstream.channel_pred_reconstruction1024x64.comm_eval import (
    evaluate_fa_beamforming,
    evaluate_non_fa_ser,
)
from twm.downstream.channel_pred_reconstruction1024x64.training import (
    ChannelPredReconstruction1024x64Config,
    ChannelPredReconstruction1024x64Task,
    build_conditioning_backbone,
)
from twm.run_helpers.run_downstream_reconstruction1024x64 import (
    ChannelPredReconstruction1024x64TrainingConfig,
)
from twm.run_helpers.run_downstream_common import _move_batch_to_device, _mean

FIXED_WINDOW_LENGTH = 16
FIXED_RANDOM_WINDOWS_PER_SEQUENCE = 30


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Sweep fixed pilot-noise NMSE for a shared 1024x64 proposed downstream checkpoint. NoCal reports tok/pred NMSE+SER; Cal reports prediction NMSE+FA.'
    )
    parser.add_argument('--dataset-file', nargs='+', required=True)
    parser.add_argument('--trained-run-dir', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--checkpoint-name', default='best_model.pt')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--train-paths', type=int, default=466)
    parser.add_argument('--test-paths', type=int, default=83)
    parser.add_argument('--num-eval-repeats', type=int, default=1)
    return parser


def _load_base_config(trained_run_dir: Path) -> ChannelPredReconstruction1024x64TrainingConfig:
    payload = json.loads((trained_run_dir / 'run_config.json').read_text(encoding='utf-8'))
    allowed = {field.name for field in fields(ChannelPredReconstruction1024x64TrainingConfig)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return ChannelPredReconstruction1024x64TrainingConfig(**filtered)


def _pilot_data_mask(pilot_estimate: torch.Tensor) -> torch.Tensor:
    if pilot_estimate.shape[2] < 3:
        raise ValueError('Expected pilot_estimate to include inverse-mask channel at index 2')
    return pilot_estimate[:, :, 2, :, 0] > 0.5


def _channel_recovery_nmse(reconstructed: torch.Tensor, target: torch.Tensor) -> float:
    if reconstructed.shape != target.shape:
        raise ValueError(f'Expected matching reconstructed/target shapes, got {tuple(reconstructed.shape)} vs {tuple(target.shape)}')
    mse = torch.nn.functional.mse_loss(reconstructed, target)
    denom = target.pow(2).mean().clamp_min(1e-12)
    return float((mse / denom).item())


def _sample_fixed_known_rb_mask(batch_size: int, num_bands: int, known_rb_count: int, device: torch.device) -> torch.Tensor:
    count = max(0, min(int(known_rb_count), int(num_bands)))
    mask = torch.zeros(batch_size, num_bands, device=device, dtype=torch.bool)
    if count <= 0:
        return mask
    if count >= num_bands:
        mask.fill_(True)
        return mask
    order = torch.rand(batch_size, num_bands, device=device).argsort(dim=1)
    chosen = order[:, :count]
    mask.scatter_(1, chosen, True)
    return mask


def _beam_metrics_batch(reconstructed: torch.Tensor, target: torch.Tensor, rb_mask: torch.Tensor | None) -> tuple[float, float]:
    avg_power_values: list[float] = []
    ratio_values: list[float] = []
    for b in range(reconstructed.shape[0]):
        for n in range(reconstructed.shape[1]):
            if rb_mask is not None and not bool(rb_mask[b, n].item()):
                continue
            out = evaluate_fa_beamforming(reconstructed[b, n], target[b, n])
            avg_power_values.append(out.avg_rx_power)
            ratio_values.append(out.power_ratio)
    if not avg_power_values:
        return float('nan'), float('nan')
    return _mean(avg_power_values), _mean(ratio_values)


def _evaluate_loader_compare(task: ChannelPredReconstruction1024x64Task, loader, device: torch.device, *, num_eval_repeats: int, fixed_nmse: float, known_rb_count: int | None = None) -> dict[str, float]:
    task.eval()
    mode = str(task.config.pilot_align_mode).strip().lower()
    repeat_values: dict[str, list[float]] = {}
    with torch.no_grad():
        for _repeat_idx in range(int(num_eval_repeats)):
            per_repeat: dict[str, list[float]] = {}
            for batch in loader:
                batch = _move_batch_to_device(batch, device)
                target_source = batch.current_gt if batch.current_gt is not None else batch.current_channel
                current_preprocessed, target_preprocessed = task.preprocess_channel_pair(batch.current_channel, target_source)
                target_latents, _ = task.encode_channels(batch.current_channel)
                predicted_latents = task._predict_current_latents(batch.history_channel, batch.history_mask, batch.current_channel)
                if mode == 'none':
                    if batch.subband_mask is None:
                        effective_mask = torch.ones(target_latents.shape[:2], dtype=torch.bool, device=target_latents.device)
                    else:
                        effective_mask = batch.subband_mask.to(dtype=torch.bool)
                else:
                    effective_mask = _sample_fixed_known_rb_mask(
                        target_latents.shape[0],
                        target_latents.shape[1],
                        target_latents.shape[1] if known_rb_count is None else int(known_rb_count),
                        target_latents.device,
                    )
                pilot_estimate = task._build_pilot_estimate_from_preprocessed(current_preprocessed)
                pilot_estimate = pilot_estimate * effective_mask[..., None, None, None].to(dtype=pilot_estimate.dtype)
                flat_pilot = pilot_estimate.reshape(-1, *pilot_estimate.shape[-3:])[:, :2]
                data_mask = _pilot_data_mask(pilot_estimate)
                B, N = target_preprocessed.shape[:2]
                H, W = target_preprocessed.shape[-2:]
                tok_condition_tokens = target_latents.reshape(-1, 1, target_latents.shape[-1])
                pred_condition_tokens = predicted_latents.reshape(-1, 1, predicted_latents.shape[-1])
                if mode == 'none':
                    tok_recovered = task.regressor(flat_pilot, tok_condition_tokens).view(B, N, 2, H, W)
                    pred_recovered = task.regressor(flat_pilot, pred_condition_tokens).view(B, N, 2, H, W)
                    tok_qpsk = evaluate_non_fa_ser(tok_recovered, target_preprocessed, modulation='qpsk', noise_var=float(fixed_nmse), data_mask=data_mask)
                    tok_qam16 = evaluate_non_fa_ser(tok_recovered, target_preprocessed, modulation='16qam', noise_var=float(fixed_nmse), data_mask=data_mask)
                    pred_qpsk = evaluate_non_fa_ser(pred_recovered, target_preprocessed, modulation='qpsk', noise_var=float(fixed_nmse), data_mask=data_mask)
                    pred_qam16 = evaluate_non_fa_ser(pred_recovered, target_preprocessed, modulation='16qam', noise_var=float(fixed_nmse), data_mask=data_mask)
                    per_repeat.setdefault('tok_recovery_nmse', []).append(_channel_recovery_nmse(tok_recovered, target_preprocessed))
                    per_repeat.setdefault('prediction_recovery_nmse', []).append(_channel_recovery_nmse(pred_recovered, target_preprocessed))
                    per_repeat.setdefault('tok_ser_qpsk', []).append(tok_qpsk.ser)
                    per_repeat.setdefault('tok_ser_16qam', []).append(tok_qam16.ser)
                    per_repeat.setdefault('prediction_ser_qpsk', []).append(pred_qpsk.ser)
                    per_repeat.setdefault('prediction_ser_16qam', []).append(pred_qam16.ser)
                else:
                    pred_calibrated, _ = task.pilot_align(predicted_latents, pilot_estimate[:, :, :2], effective_mask)
                    pred_recovered = task.regressor(flat_pilot, pred_calibrated.reshape(-1, 1, pred_calibrated.shape[-1])).view(B, N, 2, H, W)
                    target_rb_mask = ~effective_mask.to(dtype=torch.bool)
                    pred_avg, pred_ratio = _beam_metrics_batch(pred_recovered, target_preprocessed, target_rb_mask)
                    per_repeat.setdefault('prediction_recovery_nmse', []).append(_channel_recovery_nmse(pred_recovered, target_preprocessed))
                    per_repeat.setdefault('prediction_fa_avg_rx_power', []).append(pred_avg)
                    per_repeat.setdefault('prediction_fa_power_ratio', []).append(pred_ratio)
            for key, values in per_repeat.items():
                repeat_values.setdefault(key, []).append(_mean(values))
    return {key: _mean(values) for key, values in repeat_values.items()}


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

    _, test_loader, _ = build_channel_pred_three_way_dataloaders(
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
            pilot_align_input_mode=getattr(config, 'pilot_align_input_mode', 'tok'),
            pilot_est_subcarrier_strides=tuple(config.pilot_est_subcarrier_strides),
            pilot_est_noise_mode='fixed_nmse',
            pilot_fixed_nmse=0.0,
            eval_history_mask_rate=config.eval_history_mask_rate,
        ),
        freeze_backbone=config.freeze_backbone,
    )
    state_dict = torch.load(trained_run_dir / args.checkpoint_name, map_location='cpu')
    task.load_state_dict(state_dict)
    task = task.to(device)

    mode = str(task.config.pilot_align_mode).strip().lower()
    if mode == 'none':
        fieldnames = [
            'trained_run_dir', 'pretrained_dir', 'resolved_backbone_kind', 'resolved_reference_domain', 'method', 'fixed_nmse',
            'tok_recovery_nmse', 'prediction_recovery_nmse',
            'tok_ser_qpsk', 'tok_ser_16qam', 'prediction_ser_qpsk', 'prediction_ser_16qam',
        ]
    else:
        fieldnames = [
            'trained_run_dir', 'pretrained_dir', 'resolved_backbone_kind', 'resolved_reference_domain', 'method', 'fixed_nmse', 'known_rb_count',
            'prediction_recovery_nmse', 'prediction_fa_avg_rx_power', 'prediction_fa_power_ratio',
        ]
    rows = []
    if mode == 'none':
        known_counts = [None]
    else:
        known_counts = list(range(1, 8))
    for known_rb_count in known_counts:
        for k in range(11):
            nmse = k / 10.0
            task.config.pilot_est_noise_mode = 'fixed_nmse'
            task.config.pilot_fixed_nmse = nmse
            task.config.pilot_input_noise_var = 0.0
            task.config.pilot_total_noise_var_target = 0.0
            metrics = _evaluate_loader_compare(
                task,
                test_loader,
                device,
                num_eval_repeats=args.num_eval_repeats,
                fixed_nmse=nmse,
                known_rb_count=known_rb_count,
            )
            row = {
                'trained_run_dir': str(trained_run_dir),
                'pretrained_dir': str(config.pretrained_dir),
                'resolved_backbone_kind': backbone_kind,
                'resolved_reference_domain': reference_domain,
                'method': 'token_only_proposed',
                'fixed_nmse': nmse,
            }
            if known_rb_count is not None:
                row['known_rb_count'] = int(known_rb_count)
            row.update(metrics)
            rows.append(row)

    with output_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {output_csv}')


if __name__ == '__main__':
    main()
