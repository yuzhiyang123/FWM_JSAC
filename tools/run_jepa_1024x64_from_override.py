from __future__ import annotations

import argparse
from pathlib import Path

from twm.pipelines.jepa_override_configs import get_override_spec
from twm.pipelines.jepa_1024x64.training import JEPA1024x64TrainingConfig, run_jepa_1024x64_training
from twm.pipelines.jepa_1024x64.training_ema import JEPA1024x64EMATrainingConfig, run_jepa_1024x64_ema_training
from twm.pipelines.jepa_1024x64.training_ema_scratch import JEPA1024x64EMAScratchTrainingConfig, run_jepa_1024x64_ema_scratch_training



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch one jepa_1024x64 run from a 32x16 override config id.")
    parser.add_argument("--method", choices=["transfer_ref32", "transfer_ref128", "ema_ref32", "ema_ref128", "scratch"], required=True)
    parser.add_argument("--config-id", type=int, required=True)
    parser.add_argument("--dataset-file", nargs="+", required=True)
    parser.add_argument("--working-dir-root", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--random-windows-per-sequence", type=int, default=1)
    parser.add_argument("--train-paths", type=int, default=466)
    parser.add_argument("--test-paths", type=int, default=83)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--teacher-momentum", type=float, default=0.95)
    parser.add_argument("--pretrain-32-root")
    parser.add_argument("--pretrain-128-transfer-root")
    parser.add_argument("--pretrain-128-ema-root")
    parser.add_argument("--run-name-prefix", default="open1024")
    return parser


def _resolve_32_best_dir(root: str, config_id: int, run_name: str) -> Path:
    path = Path(root) / f"{config_id}_{run_name}" / "best_model"
    if not path.exists():
        raise SystemExit(f"Missing 32x16 pretrained dir: {path}")
    return path


def _resolve_128_best_dir(root: str, config_id: int, run_name: str, suffix: str) -> Path:
    path = Path(root) / f"{config_id}_{run_name}__{suffix}" / "best_model"
    if not path.exists():
        raise SystemExit(f"Missing 128x64 pretrained dir: {path}")
    return path


def main() -> None:
    args = build_arg_parser().parse_args()
    spec = get_override_spec(args.config_id)
    basic = spec.basic
    run_name = spec.name
    working_dir_root = Path(args.working_dir_root)
    working_dir_root.mkdir(parents=True, exist_ok=True)

    if args.method == "scratch":
        cfg = JEPA1024x64EMAScratchTrainingConfig(
            dataset_file=[str(x) for x in args.dataset_file],
            working_dir=str(working_dir_root / f"{args.config_id}_{run_name}__jepa1024_scratch"),
            jepa_config_name=basic.jepa_config.preset_name,
            num_low_bands=5,
            num_tokens=basic.jepa_config.num_tokens,
            token_dim=basic.jepa_config.token_dim,
            predictor_dim=basic.jepa_config.predictor_dim,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epochs=args.epochs,
            num_workers=args.num_workers,
            window_length=args.window_length,
            random_windows_per_sequence=args.random_windows_per_sequence,
            train_paths=args.train_paths,
            test_paths=args.test_paths,
            seed=args.seed,
            device=args.device,
            teacher_momentum=args.teacher_momentum,
            tokenizer_output_norm_32x16=str(basic.tokenizer_output_norm or "batchnorm"),
            save_every=args.save_every,
            wandb_project=args.wandb_project,
            wandb_run_name=f"{args.run_name_prefix}_scratch_{args.config_id}_{run_name}",
        )
        run_jepa_1024x64_ema_scratch_training(cfg)
        return

    if not args.pretrain_32_root:
        raise SystemExit("--pretrain-32-root is required for pretrained methods")
    pre32 = _resolve_32_best_dir(args.pretrain_32_root, args.config_id, run_name)

    if args.method in {"transfer_ref32", "transfer_ref128"}:
        if not args.pretrain_128_transfer_root:
            raise SystemExit("--pretrain-128-transfer-root is required for transfer methods")
        pre128 = _resolve_128_best_dir(args.pretrain_128_transfer_root, args.config_id, run_name, 'jepa128_transfer')
        ref = '32x16' if args.method.endswith('ref32') else '128x64'
        suffix = 'jepa1024_transfer_ref32' if ref == '32x16' else 'jepa1024_transfer_ref128'
        cfg = JEPA1024x64TrainingConfig(
            dataset_file=[str(x) for x in args.dataset_file],
            pretrained_32x16_dir=str(pre32),
            pretrained_128x64_dir=str(pre128),
            working_dir=str(working_dir_root / f"{args.config_id}_{run_name}__{suffix}"),
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epochs=args.epochs,
            num_workers=args.num_workers,
            window_length=args.window_length,
            random_windows_per_sequence=args.random_windows_per_sequence,
            train_paths=args.train_paths,
            test_paths=args.test_paths,
            seed=args.seed,
            device=args.device,
            save_every=args.save_every,
            wandb_project=args.wandb_project,
            wandb_run_name=f"{args.run_name_prefix}_{suffix}_{args.config_id}_{run_name}",
            reference_domain=ref,
        )
        run_jepa_1024x64_training(cfg)
        return

    if not args.pretrain_128_ema_root:
        raise SystemExit("--pretrain-128-ema-root is required for ema methods")
    pre128 = _resolve_128_best_dir(args.pretrain_128_ema_root, args.config_id, run_name, 'jepa128_ema32')
    ref = '32x16' if args.method.endswith('ref32') else '128x64'
    suffix = 'jepa1024_ema_ref32' if ref == '32x16' else 'jepa1024_ema_ref128'
    cfg = JEPA1024x64EMATrainingConfig(
        dataset_file=[str(x) for x in args.dataset_file],
        pretrained_32x16_dir=str(pre32),
        pretrained_128x64_dir=str(pre128),
        working_dir=str(working_dir_root / f"{args.config_id}_{run_name}__{suffix}"),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        num_workers=args.num_workers,
        window_length=args.window_length,
        random_windows_per_sequence=args.random_windows_per_sequence,
        train_paths=args.train_paths,
        test_paths=args.test_paths,
        seed=args.seed,
        device=args.device,
        teacher_momentum=args.teacher_momentum,
        save_every=args.save_every,
        wandb_project=args.wandb_project,
        wandb_run_name=f"{args.run_name_prefix}_{suffix}_{args.config_id}_{run_name}",
        reference_domain=ref,
    )
    run_jepa_1024x64_ema_training(cfg)


if __name__ == "__main__":
    main()
