from __future__ import annotations

import argparse
from pathlib import Path

from twm.pipelines.jepa_override_configs import get_override_spec
from twm.pipelines.jepa_128x64.training import JEPA128x64TrainingConfig, run_jepa_128x64_training
from twm.pipelines.jepa_128x64.training_ema32 import JEPA128x64EMA32TrainingConfig, run_jepa_128x64_ema32_training
from twm.pipelines.jepa_128x64.training_ema32_scratch import JEPA128x64EMA32ScratchTrainingConfig, run_jepa_128x64_ema32_scratch_training


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch one jepa_128x64 run from a 32x16 override config id.")
    parser.add_argument("--method", choices=["transfer", "ema32", "scratch"], required=True)
    parser.add_argument("--config-id", type=int, required=True)
    parser.add_argument("--dataset-file", nargs="+", required=True)
    parser.add_argument("--working-dir-root", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
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
    parser.add_argument("--pretrain-root")
    parser.add_argument("--run-name-prefix", default="open128")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    spec = get_override_spec(args.config_id)
    basic = spec.basic
    run_name = spec.name
    working_dir_root = Path(args.working_dir_root)
    try:
        working_dir_root.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        if not working_dir_root.is_dir():
            raise

    if args.method == "transfer":
        if not args.pretrain_root:
            raise SystemExit("--pretrain-root is required for transfer method")
        pretrain_dir = Path(args.pretrain_root) / f"{args.config_id}_{run_name}" / "best_model"
        if not pretrain_dir.exists():
            raise SystemExit(f"Missing pretrained dir: {pretrain_dir}")
        cfg = JEPA128x64TrainingConfig(
            dataset_file=[str(x) for x in args.dataset_file],
            pretrained_32x16_dir=str(pretrain_dir),
            working_dir=str(working_dir_root / f"{args.config_id}_{run_name}__jepa128_transfer"),
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
            wandb_run_name=f"{args.run_name_prefix}_transfer_{args.config_id}_{run_name}",
        )
        run_jepa_128x64_training(cfg)
        return

    if args.method == "ema32":
        if not args.pretrain_root:
            raise SystemExit("--pretrain-root is required for ema32 method")
        pretrain_dir = Path(args.pretrain_root) / f"{args.config_id}_{run_name}" / "best_model"
        if not pretrain_dir.exists():
            raise SystemExit(f"Missing pretrained dir: {pretrain_dir}")
        cfg = JEPA128x64EMA32TrainingConfig(
            dataset_file=[str(x) for x in args.dataset_file],
            pretrained_32x16_dir=str(pretrain_dir),
            working_dir=str(working_dir_root / f"{args.config_id}_{run_name}__jepa128_ema32"),
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
            wandb_run_name=f"{args.run_name_prefix}_ema32_{args.config_id}_{run_name}",
        )
        run_jepa_128x64_ema32_training(cfg)
        return

    cfg = JEPA128x64EMA32ScratchTrainingConfig(
        dataset_file=[str(x) for x in args.dataset_file],
        working_dir=str(working_dir_root / f"{args.config_id}_{run_name}__jepa128_ema32_scratch"),
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
    run_jepa_128x64_ema32_scratch_training(cfg)


if __name__ == "__main__":
    main()
