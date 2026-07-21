#!/bin/bash
#SBATCH --job-name=jepa1024_e128
#SBATCH --output=logs/jepa1024_ema_ref128_%A_%a.out
#SBATCH --error=logs/jepa1024_ema_ref128_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/path/to/prepared_dataset_root/01chicago}"
TRAIN_DATASET_DIR="$DATA_ROOT/shards/01chicago/01chicago_gt_64ant_train_npy"
TEST_DATASET_DIR="$DATA_ROOT/shards/01chicago/01chicago_gt_64ant_test_npy"

if [[ "$DATA_ROOT" == /path/to/* ]]; then
  echo "Please set DATA_ROOT to the prepared dataset root." >&2
  exit 1
fi
if [[ ! -d "$TRAIN_DATASET_DIR" || ! -d "$TEST_DATASET_DIR" ]]; then
  echo "Dataset shard directories are missing under DATA_ROOT=$DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/logs"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

SELECTED_CONFIG_IDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
TASK_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "$TASK_INDEX" ]]; then echo "TASK_INDEX is required." >&2; exit 1; fi
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#SELECTED_CONFIG_IDS[@]} )); then echo "TASK_INDEX out of range" >&2; exit 1; fi
CONFIG_ID="${SELECTED_CONFIG_IDS[$TASK_INDEX]}"
PRETRAIN_32_ROOT="${PRETRAIN_32_ROOT:-/path/to/output_run_root/jepa_32x16}"
if [[ "$PRETRAIN_32_ROOT" == /path/to/* ]]; then
  echo "Please set PRETRAIN_32_ROOT." >&2
  exit 1
fi
PRETRAIN_128_EMA_ROOT="${PRETRAIN_128_EMA_ROOT:-/path/to/output_run_root/jepa_128x64_ema32}"
if [[ "$PRETRAIN_128_EMA_ROOT" == /path/to/* ]]; then
  echo "Please set PRETRAIN_128_EMA_ROOT." >&2
  exit 1
fi
WORKING_DIR_ROOT="${WORKING_DIR_ROOT:-/path/to/output_run_root/jepa_1024x64_ema_ref128}"
WANDB_PROJECT="${WANDB_PROJECT:-twm_jepa_1024x64_ema_ref128_open}"

"$PYTHON_BIN" tools/run_jepa_1024x64_from_override.py \
  --method ema_ref128 \
  --config-id "$CONFIG_ID" \
  --dataset-file "$TRAIN_DATASET_DIR" "$TEST_DATASET_DIR" \
  --pretrain-32-root "$PRETRAIN_32_ROOT" \
  --pretrain-128-ema-root "$PRETRAIN_128_EMA_ROOT" \
  --working-dir-root "$WORKING_DIR_ROOT" \
  --wandb-project "$WANDB_PROJECT" \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --epochs 500 \
  --num-workers 0 \
  --window-length 16 \
  --random-windows-per-sequence 1 \
  --train-paths 466 \
  --test-paths 83 \
  --seed 42 \
  --device cuda \
  --save-every 20 \
  --teacher-momentum 0.95 \
  --run-name-prefix open1024
