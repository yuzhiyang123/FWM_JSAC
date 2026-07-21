#!/bin/bash
#SBATCH --job-name=jepa32_open
#SBATCH --output=logs/jepa32_open_%A_%a.out
#SBATCH --error=logs/jepa32_open_%A_%a.err
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

WORKING_DIR_ROOT="${WORKING_DIR_ROOT:-/path/to/output_run_root/jepa_32x16}"
WANDB_PROJECT="${WANDB_PROJECT:-twm_jepa_32x16_open}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-open32}"
CONFIG_ID="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "$CONFIG_ID" ]]; then
  echo "CONFIG_ID is required." >&2
  exit 1
fi

"$PYTHON_BIN" tools/run_jepa_32x16_from_override.py \
  --config-id "$CONFIG_ID" \
  --dataset-file "$TRAIN_DATASET_DIR" "$TEST_DATASET_DIR" \
  --working-dir-root "$WORKING_DIR_ROOT" \
  --wandb-project "$WANDB_PROJECT" \
  --run-name-prefix "$RUN_NAME_PREFIX" \
  --num-low-bands 5 \
  --batch-size 32 \
  --lr 1e-4 \
  --epochs 1000 \
  --num-workers 0 \
  --window-length 16 \
  --random-windows-per-sequence 1 \
  --train-paths 466 \
  --test-paths 83 \
  --seed 42 \
  --device cuda \
  --save-every 20
