from __future__ import annotations

from twm.pipelines.jepa_128x64.training_ema32_scratch import build_arg_parser, run_jepa_128x64_ema32_scratch_training


def main() -> None:
    args = build_arg_parser().parse_args()
    run_jepa_128x64_ema32_scratch_training(args)


if __name__ == "__main__":
    main()
