#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK:-1}
#SBATCH --mem=${MEMORY:-64G}
#SBATCH --time=${WALLTIME:-36:00:00}
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --gres=${SLURM_GRES:-gpu:1}

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_TRAIN="${DATASET_TRAIN:-/path/to/train_dataset}"
DATASET_TEST="${DATASET_TEST:-/path/to/test_dataset}"
WORKING_DIR="${WORKING_DIR:-/path/to/output_run_dir}"
PRETRAINED_DIR="${PRETRAINED_DIR:-/path/to/pretrained_jepa_dir}"
if [[ "$DATASET_TRAIN" == /path/to/* || "$DATASET_TEST" == /path/to/* || "$WORKING_DIR" == /path/to/* || "$PRETRAINED_DIR" == /path/to/* ]]; then
  echo "Please set DATASET_TRAIN, DATASET_TEST, WORKING_DIR, and PRETRAINED_DIR before submission." >&2
  exit 1
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$WORKING_DIR"
"$PYTHON_BIN" -u -m twm.run_helpers.run_downstream_token_only_reconstruction1024x64 \
  --dataset-file "$DATASET_TRAIN" "$DATASET_TEST" \
  --pretrained-dir "$PRETRAINED_DIR" \
  --working-dir "$WORKING_DIR" \
  --train-paths "${TRAIN_PATHS:-466}" \
  --test-paths "${TEST_PATHS:-83}" \
  --epochs "${EPOCHS:-300}" \
  --eval-every-epochs "${EVAL_EVERY_EPOCHS:-10}" \
  --pilot-align-mode "${PILOT_ALIGN_MODE:-none}" \
  --pilot-est-subcarrier-strides "${PILOT_STRIDE:-128}" \
  --condition-source "${CONDITION_SOURCE:-token_only}" \
  --backbone-init "${BACKBONE_INIT:-pretrained}" \
  --train-mode "${TRAIN_MODE:-auto}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --wandb-project "${WANDB_PROJECT:-twm_downstream_token_only_reconstruction1024x64_open}" \
  --wandb-run-name "${WANDB_RUN_NAME:-}"
