from __future__ import annotations

import argparse

from twm.pipelines.jepa_32x16 import JEPATrainingConfig, run_jepa_training


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = JEPATrainingConfig()
    parser = argparse.ArgumentParser(description='Train the reconstructed twm JEPA pipeline.')
    parser.add_argument('--dataset-file', nargs='+', required=True)
    parser.add_argument('--working-dir', required=True)
    parser.add_argument('--jepa-config-name', default=defaults.jepa_config.preset_name)
    parser.add_argument('--num-low-bands', type=int, default=5)
    parser.add_argument('--num-tokens', type=int, default=defaults.jepa_config.num_tokens)
    parser.add_argument('--token-dim', type=int, default=defaults.jepa_config.token_dim)
    parser.add_argument('--predictor-dim', type=int, default=defaults.jepa_config.predictor_dim)
    parser.add_argument('--batch-size', type=int, default=defaults.batch_size)
    parser.add_argument('--lr', type=float, default=defaults.lr)
    parser.add_argument('--epochs', type=int, default=defaults.epochs)
    parser.add_argument('--num-workers', type=int, default=defaults.num_workers)
    parser.add_argument('--window-length', type=int, default=defaults.window_length)
    parser.add_argument('--random-windows-per-sequence', type=int, default=defaults.random_windows_per_sequence)
    parser.add_argument('--test-ratio', type=float, default=defaults.test_ratio)
    parser.add_argument('--train-paths', type=int, default=defaults.train_paths)
    parser.add_argument('--test-paths', type=int, default=defaults.test_paths)
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--device', default=defaults.device)
    parser.add_argument('--weight-decay', type=float, default=defaults.weight_decay)
    parser.add_argument('--teacher-momentum', type=float, default=defaults.teacher_momentum)
    parser.add_argument('--masked-target-source', choices=['teacher_tokenizer', 'student_tokenizer', 'teacher_encoder'], default=defaults.masked_target_source)
    parser.add_argument('--tokenizer-output-norm', choices=['batchnorm', 'layernorm', 'none'], default=defaults.tokenizer_output_norm)
    parser.add_argument('--mask-style', choices=['random', 'random_except_last', 'curriculum_random_except_last'], default=defaults.mask_style)
    parser.add_argument('--mask-rate', type=float, default=defaults.mask_rate)
    parser.add_argument('--curriculum-start-mask-rate', type=float, default=defaults.curriculum_start_mask_rate)
    parser.add_argument('--curriculum-end-mask-rate', type=float, default=defaults.curriculum_end_mask_rate)
    parser.add_argument('--curriculum-ramp-epochs', type=int, default=defaults.curriculum_ramp_epochs)
    parser.add_argument('--curriculum-sample-jitter', type=float, default=defaults.curriculum_sample_jitter)
    parser.add_argument('--final-min-mask-rate', type=float, default=defaults.final_min_mask_rate)
    parser.add_argument('--final-max-mask-rate', type=float, default=defaults.final_max_mask_rate)
    parser.add_argument('--final-beta-alpha', type=float, default=defaults.final_beta_alpha)
    parser.add_argument('--final-beta-beta', type=float, default=defaults.final_beta_beta)
    parser.add_argument('--jepa-loss-weight', type=float, default=defaults.jepa_loss_weight)
    parser.add_argument('--remained-nt-vicreg-weight', type=float, default=defaults.remained_nt_vicreg_weight)
    parser.add_argument('--remained-nt-vicreg-floor', type=float, default=defaults.remained_nt_vicreg_floor)
    parser.add_argument('--remained-cov-fro-weight', type=float, default=defaults.remained_cov_fro_weight)
    parser.add_argument('--alignment-loss-weight', type=float, default=defaults.alignment_loss_weight)
    parser.add_argument('--recovery-loss-weight', type=float, default=defaults.recovery_loss_weight)
    parser.add_argument('--amplitude-recovery-loss-weight', type=float, default=defaults.amplitude_recovery_loss_weight)
    parser.add_argument('--diffusion-loss-weight', type=float, default=defaults.diffusion_loss_weight)
    parser.add_argument('--save-every', type=int, default=defaults.save_every)
    parser.add_argument('--stage2-epochs', type=int, default=defaults.stage2_epochs)
    parser.add_argument('--stage2-freeze-tokenizers', type=int, default=defaults.stage2_freeze_tokenizers)
    parser.add_argument('--stage2-jepa-only', type=int, default=defaults.stage2_jepa_only)
    parser.add_argument('--wandb-project', default=defaults.wandb_project)
    parser.add_argument('--wandb-run-name', default=defaults.wandb_run_name)
    parser.add_argument('--estimate-main-task', type=int, default=defaults.estimate_main_task)
    parser.add_argument('--estimate-steps', type=int, default=defaults.estimate_steps)
    parser.add_argument('--estimate-warmup-steps', type=int, default=defaults.estimate_warmup_steps)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_jepa_training(args)


if __name__ == '__main__':
    main()
