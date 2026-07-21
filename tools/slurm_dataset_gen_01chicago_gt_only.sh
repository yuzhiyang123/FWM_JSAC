#!/bin/bash
#SBATCH --job-name=twm_01chi_gt_only
#SBATCH --output=logs/twm_01chi_gt_only_%j.out
#SBATCH --error=logs/twm_01chi_gt_only_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

RAW_DATA_ROOT="${RAW_DATA_ROOT:-/path/to/raw_dataset_root}"
RUN_ROOT="${RUN_ROOT:-/path/to/output_run_root}"
DRJIT_ROOT="${DRJIT_ROOT:-$RAW_DATA_ROOT/.drjit}"
SCENARIO_NAME="${SCENARIO_NAME:-01chicago}"
XYZ_PATHS_FILE="$RAW_DATA_ROOT/$SCENARIO_NAME/xyz_paths.json"
SCENARIO_RUN_ROOT="$RUN_ROOT/$SCENARIO_NAME"
SNAPSHOT_PATH="$SCENARIO_RUN_ROOT/${SCENARIO_NAME}_snapshot.json.gz"
TRAIN_SNAPSHOT_PATH="$SCENARIO_RUN_ROOT/${SCENARIO_NAME}_snapshot_train.json.gz"
TEST_SNAPSHOT_PATH="$SCENARIO_RUN_ROOT/${SCENARIO_NAME}_snapshot_test.json.gz"
SPLIT_REPORT_PATH="$SCENARIO_RUN_ROOT/${SCENARIO_NAME}_split_report.json"
SHARD_ROOT="$SCENARIO_RUN_ROOT/shards/$SCENARIO_NAME"
TRAIN_SHARD_DIR="$SHARD_ROOT/${SCENARIO_NAME}_gt_64ant_train_npy"
TEST_SHARD_DIR="$SHARD_ROOT/${SCENARIO_NAME}_gt_64ant_test_npy"
TRAIN_GAIN_REPORT="$SCENARIO_RUN_ROOT/${SCENARIO_NAME}_train_channel_gain_report.json"
TEST_GAIN_REPORT="$SCENARIO_RUN_ROOT/${SCENARIO_NAME}_test_channel_gain_report.json"

NUM_ANT="64"
ANT_TYPE="rectangle"
DEVICE="cpu"
BATCH_SIZE="128"
MAX_DEPTH="5"
TRAIN_PATHS="512"
TEST_PATHS="88"
SPLIT_SEED="42"
MIN_POINT_POWER="1e-10"

if [[ "$RAW_DATA_ROOT" == "/path/to/raw_dataset_root" || "$RUN_ROOT" == "/path/to/output_run_root" ]]; then
  echo "Please set RAW_DATA_ROOT and RUN_ROOT before submitting this script." >&2
  exit 1
fi

if [[ ! -d "$RAW_DATA_ROOT/$SCENARIO_NAME" ]]; then
  echo "Scenario directory does not exist: $RAW_DATA_ROOT/$SCENARIO_NAME" >&2
  exit 1
fi

if [[ ! -f "$XYZ_PATHS_FILE" ]]; then
  echo "XYZ paths file does not exist: $XYZ_PATHS_FILE" >&2
  exit 1
fi

mkdir -p "$SCENARIO_RUN_ROOT"
mkdir -p "$SHARD_ROOT"
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$DRJIT_ROOT"

export DRJIT_LIBLLVM_PATH="$DRJIT_ROOT/libLLVM.so"
export DRJIT_LIBLLVVM_PATH="$DRJIT_ROOT/libLLVM.so"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

cd "$REPO_ROOT"

echo "==== generate snapshots ===="
echo "scenario:      $SCENARIO_NAME"
echo "xyz_paths:     $XYZ_PATHS_FILE"
echo "snapshot_path: $SNAPSHOT_PATH"
echo "max_depth:     $MAX_DEPTH"
echo "batch_size:    $BATCH_SIZE"
echo "min_point_power: $MIN_POINT_POWER"
"$PYTHON_BIN" -m twm.datasets.data_gen_sionna \
  --raw-data-dir "$RAW_DATA_ROOT" \
  --scenario-name "$SCENARIO_NAME" \
  --xyz-paths-file "$XYZ_PATHS_FILE" \
  --output-path "$SNAPSHOT_PATH" \
  --batch-size "$BATCH_SIZE" \
  --max-depth "$MAX_DEPTH"

echo "==== split snapshots into train/test ===="
"$PYTHON_BIN" -m twm.datasets.split_snapshot_dataset \
  --input-file "$SNAPSHOT_PATH" \
  --train-output-file "$TRAIN_SNAPSHOT_PATH" \
  --test-output-file "$TEST_SNAPSHOT_PATH" \
  --train-paths "$TRAIN_PATHS" \
  --test-paths "$TEST_PATHS" \
  --seed "$SPLIT_SEED" \
  --report-path "$SPLIT_REPORT_PATH"

echo "==== build train gt shards ===="
"$PYTHON_BIN" -m twm.datasets.fixed_band_dataset \
  --snapshot-file "$TRAIN_SNAPSHOT_PATH" \
  --output-path "$TRAIN_SHARD_DIR" \
  --output-format npy_dir \
  --report-path "$TRAIN_GAIN_REPORT" \
  --min-point-power "$MIN_POINT_POWER" \
  --num-ant "$NUM_ANT" \
  --ant-type "$ANT_TYPE" \
  --device "$DEVICE"

echo "==== build test gt shards ===="
"$PYTHON_BIN" -m twm.datasets.fixed_band_dataset \
  --snapshot-file "$TEST_SNAPSHOT_PATH" \
  --output-path "$TEST_SHARD_DIR" \
  --output-format npy_dir \
  --report-path "$TEST_GAIN_REPORT" \
  --min-point-power "$MIN_POINT_POWER" \
  --num-ant "$NUM_ANT" \
  --ant-type "$ANT_TYPE" \
  --device "$DEVICE"

echo "dataset generation finished"
echo "snapshot:      $SNAPSHOT_PATH"
echo "train_snapshot:$TRAIN_SNAPSHOT_PATH"
echo "test_snapshot: $TEST_SNAPSHOT_PATH"
echo "train_shards:  $TRAIN_SHARD_DIR"
echo "test_shards:   $TEST_SHARD_DIR"
echo "split_report:  $SPLIT_REPORT_PATH"
echo "train_gain:    $TRAIN_GAIN_REPORT"
echo "test_gain:     $TEST_GAIN_REPORT"
