
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
import wandb

from twm.downstream.channel_pred import ChannelPredConfig, build_channel_pred_three_way_dataloaders
from twm.downstream.channel_pred_regression32x16 import ChannelPredRegressionTask
from twm.run_helpers.run_downstream_common import (
    ChannelPredTrainingConfig,
    _accumulate_channel_pred_metrics,
    _build_backbone,
    _mean,
    _move_batch_to_device,
    _save_model_state,
)
from twm.pipelines.jepa_32x16 import build_warmup_cosine_scheduler


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = ChannelPredTrainingConfig(wandb_project="twm_downstream_regression32x16_open")
    parser = argparse.ArgumentParser(description="Train the non-diffusion sequential channel prediction regressor with a frozen JEPA backbone.")
    parser.add_argument("--dataset-file", nargs="+", required=True)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--jepa-config-name", default=defaults.jepa_config_name)
    parser.add_argument("--num-low-bands", type=int, default=defaults.num_low_bands)
    parser.add_argument("--num-tokens", type=int, default=defaults.num_tokens)
    parser.add_argument("--token-dim", type=int, default=defaults.token_dim)
    parser.add_argument("--predictor-dim", type=int, default=defaults.predictor_dim)
    parser.add_argument("--tokenizer-output-norm", choices=["batchnorm", "layernorm", "none"], default=defaults.tokenizer_output_norm)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--train-paths", type=int, default=defaults.train_paths)
    parser.add_argument("--test-paths", type=int, default=defaults.test_paths)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--window-length", type=int, default=defaults.window_length)
    parser.add_argument("--random-windows-per-sequence", type=int, default=defaults.random_windows_per_sequence)
    parser.add_argument("--eval-every-epochs", type=int, default=defaults.eval_every_epochs)
    parser.add_argument("--pilot-est-subcarrier-strides", nargs="+", type=int, default=list(defaults.pilot_est_subcarrier_strides))
    parser.add_argument("--pilot-est-max-noise-std", type=float, default=defaults.pilot_est_max_noise_std)
    parser.add_argument("--pilot-input-noise-var", type=float, default=defaults.pilot_input_noise_var)
    parser.add_argument("--pilot-total-noise-var-target", type=float, default=defaults.pilot_total_noise_var_target)
    parser.add_argument("--fusion-mode", choices=["pilot_only", "split_cross_attn", "gated_fusion", "latent_residual", "late_fusion"], default=defaults.fusion_mode)
    parser.add_argument("--wandb-project", default=defaults.wandb_project)
    parser.add_argument("--wandb-run-name", default=defaults.wandb_run_name)
    parser.add_argument("--failed-jepa-list", default=defaults.failed_jepa_list)
    parser.add_argument("--latent-nmse-fail-threshold", type=float, default=defaults.latent_nmse_fail_threshold)
    return parser


def _config_from_namespace(args: argparse.Namespace) -> ChannelPredTrainingConfig:
    payload = vars(args).copy()
    payload["dataset_file"] = [str(x) for x in payload["dataset_file"]]
    payload["pretrained_dir"] = str(payload["pretrained_dir"])
    payload["working_dir"] = str(payload["working_dir"])
    payload["pilot_est_subcarrier_strides"] = tuple(int(x) for x in payload["pilot_est_subcarrier_strides"])
    return ChannelPredTrainingConfig(**payload)


def _evaluate_loader(task: ChannelPredRegressionTask, loader, device: torch.device) -> dict[str, float]:
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
        "loss": _mean(losses),
        "x_loss": _mean(x_losses),
        "recon_loss": _mean(recon),
        "practical_recon_loss": _mean(practical),
        "latent_target_power": _mean(latent_target_power),
        "latent_nmse": _mean(latent_nmse),
    }
    if gt_recon:
        metrics["gt_recon_loss"] = _mean(gt_recon)
    return metrics


def run_downstream_regression32x16_training(args: argparse.Namespace | ChannelPredTrainingConfig) -> dict[str, object]:
    config = args if isinstance(args, ChannelPredTrainingConfig) else _config_from_namespace(args)
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
    task = ChannelPredRegressionTask.from_stage2_checkpoint(
        backbone,
        pretrained_dir=config.pretrained_dir,
        config=ChannelPredConfig(
            pilot_align_mode="none",
            channel_mode="complex",
            pilot_est_subcarrier_strides=config.pilot_est_subcarrier_strides,
            pilot_est_max_noise_std=config.pilot_est_max_noise_std,
            pilot_input_noise_var=config.pilot_input_noise_var,
            pilot_total_noise_var_target=config.pilot_total_noise_var_target,
            fusion_mode=config.fusion_mode,
        ),
        freeze_backbone=True,
    ).to(device)

    trainable_parameters = [p for p in task.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(train_loader))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.1)

    failed_jepa_list_path = config.failed_jepa_list or str((working_dir.parent / "failed_jepa_runs.json"))

    config_payload = asdict(config)
    config_payload["task_type"] = "downstream_regression32x16"
    config_payload["reporting_paths"] = {
        "working_dir": str(working_dir),
        "pretrained_dir": str(config.pretrained_dir),
        "dataset_file": list(config.dataset_file),
        "failed_jepa_list": failed_jepa_list_path,
    }
    (working_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2) + '\n', encoding="utf-8")

    wandb_init_kwargs = {"name": config.wandb_run_name or working_dir.name, "dir": str(working_dir), "config": config_payload}
    wandb_entity = os.environ.get("WANDB_ENTITY", "").strip()
    if wandb_entity:
        wandb_init_kwargs["entity"] = wandb_entity
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
            train_gt_recon: list[float] = []
            train_latent_target_power: list[float] = []
            train_latent_nmse: list[float] = []
            for batch in train_loader:
                batch = _move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                result = task.training_step(batch.history_channel, batch.history_mask, batch.current_channel, subband_mask=batch.subband_mask, current_gt=batch.current_gt)
                result.loss.backward()
                optimizer.step()
                scheduler.step()
                _accumulate_channel_pred_metrics(result, train_losses, train_x, train_recon, train_practical, train_gt_recon, train_latent_target_power, train_latent_nmse)
            epoch_metrics = {
                "epoch": float(epoch),
                "train/loss": _mean(train_losses),
                "train/x_loss": _mean(train_x),
                "train/recon_loss": _mean(train_recon),
                "train/practical_recon_loss": _mean(train_practical),
                "train/latent_target_power": _mean(train_latent_target_power),
                "train/latent_nmse": _mean(train_latent_nmse),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
            }
            if train_gt_recon:
                epoch_metrics["train/gt_recon_loss"] = _mean(train_gt_recon)
            if (epoch % eval_every_epochs) == 0:
                test_metrics = _evaluate_loader(task, test_loader, device)
                for key, value in test_metrics.items():
                    epoch_metrics[f"test/{key}"] = value
                current_test_loss = epoch_metrics["test/loss"]
                if best_test_loss is None or current_test_loss < best_test_loss:
                    best_test_loss = current_test_loss
                    best_epoch = epoch
                    _save_model_state(task, working_dir / "best_model.pt")
            history.append(epoch_metrics)
            wandb.log(epoch_metrics, step=epoch)
        _save_model_state(task, working_dir / "last_model.pt")
        if best_epoch is None or not (working_dir / "best_model.pt").exists():
            best_epoch = config.epochs
            _save_model_state(task, working_dir / "best_model.pt")
        summary = {"best_epoch": best_epoch, "best_test_loss": best_test_loss, "history": history}
        (working_dir / "results.json").write_text(json.dumps(summary, indent=2) + '\n', encoding="utf-8")
        if wandb_run is not None:
            wandb_run.summary["best_epoch"] = best_epoch
            wandb_run.summary["best_test_loss"] = best_test_loss
            wandb_run.summary["working_dir"] = str(working_dir)
            wandb_run.summary["pretrained_dir"] = str(config.pretrained_dir)
        return summary
    finally:
        if wandb.run is not None:
            wandb.finish()


def main() -> None:
    args = build_arg_parser().parse_args()
    run_downstream_regression32x16_training(args)


if __name__ == "__main__":
    main()
