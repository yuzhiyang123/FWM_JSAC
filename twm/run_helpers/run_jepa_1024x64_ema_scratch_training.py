from __future__ import annotations
from twm.pipelines.jepa_1024x64.training_ema_scratch import build_arg_parser, run_jepa_1024x64_ema_scratch_training

def main() -> None:
    args = build_arg_parser().parse_args()
    run_jepa_1024x64_ema_scratch_training(args)

if __name__ == "__main__":
    main()
