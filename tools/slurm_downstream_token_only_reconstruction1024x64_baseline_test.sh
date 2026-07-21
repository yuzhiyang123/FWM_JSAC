#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK:-1}
#SBATCH --mem=${MEMORY:-64G}
#SBATCH --time=${WALLTIME:-12:00:00}
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
TRAINED_RUN_DIR="${TRAINED_RUN_DIR:-/path/to/trained_run_dir}"
OUTPUT_CSV="${OUTPUT_CSV:-/path/to/output_results.csv}"
if [[ "$DATASET_TRAIN" == /path/to/* || "$DATASET_TEST" == /path/to/* || "$TRAINED_RUN_DIR" == /path/to/* || "$OUTPUT_CSV" == /path/to/* ]]; then
  echo "Please set DATASET_TRAIN, DATASET_TEST, TRAINED_RUN_DIR, and OUTPUT_CSV before submission." >&2
  exit 1
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$(dirname -- "$OUTPUT_CSV")"
"$PYTHON_BIN" -u -m twm.run_helpers.run_downstream_token_only_reconstruction1024x64_baseline_test \
  --dataset-file "$DATASET_TRAIN" "$DATASET_TEST" \
  --trained-run-dir "$TRAINED_RUN_DIR" \
  --output-csv "$OUTPUT_CSV" \
  --train-paths "${TRAIN_PATHS:-466}" \
  --test-paths "${TEST_PATHS:-83}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --num-eval-repeats "${NUM_EVAL_REPEATS:-20}" \
  --device "${DEVICE:-cuda}"
