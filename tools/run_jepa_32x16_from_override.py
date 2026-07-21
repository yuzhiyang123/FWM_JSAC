from __future__ import annotations

import argparse
from pathlib import Path

from twm.pipelines.jepa_32x16 import JEPATrainingConfig, run_jepa_training
from twm.pipelines.jepa_override_configs import get_named_override_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch one 32x16 JEPA run from an override config id.")
    parser.add_argument("--config-id", type=int, required=True)
    parser.add_argument("--dataset-file", nargs="+", required=True)
    parser.add_argument("--working-dir-root", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--run-name-prefix", default="open32")
    parser.add_argument("--num-low-bands", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--random-windows-per-sequence", type=int, default=1)
    parser.add_argument("--train-paths", type=int, default=466)
    parser.add_argument("--test-paths", type=int, default=83)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    working_dir_root = Path(args.working_dir_root)
    working_dir_root.mkdir(parents=True, exist_ok=True)
    base = JEPATrainingConfig(
        dataset_file=[str(x) for x in args.dataset_file],
        working_dir="",
        num_low_bands=args.num_low_bands,
        batch_size=args.batch_size,
        lr=args.lr,
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
        wandb_run_name="",
    )
    run_name, cfg = get_named_override_config(base, args.config_id)
    cfg.working_dir = str(working_dir_root / f"{args.config_id}_{run_name}")
    cfg.wandb_run_name = f"{args.run_name_prefix}_{args.config_id}_{run_name}"
    run_jepa_training(cfg)


if __name__ == "__main__":
    main()
