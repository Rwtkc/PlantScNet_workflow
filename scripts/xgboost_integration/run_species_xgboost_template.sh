#!/usr/bin/env bash
set -euo pipefail

SPECIES=""
SAMPLES_DIR=""
GOLD=""
OUT_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS_DIR="${SCRIPTS_DIR:-$SCRIPT_DIR}"
MATRIX_DIR=""
NEGATIVE_RATIO="${NEGATIVE_RATIO:-5}"
FOLDS="${FOLDS:-5}"
SEED="${SEED:-42}"
N_TRIALS="${N_TRIALS:-100}"
EARLY_STOPPING="${EARLY_STOPPING:-50}"
CHUNKSIZE="${CHUNKSIZE:-100000}"
DEVICE="${DEVICE:-cuda}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --species) SPECIES="$2"; shift 2 ;;
    --samples-dir) SAMPLES_DIR="$2"; shift 2 ;;
    --gold) GOLD="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --scripts-dir) SCRIPTS_DIR="$2"; shift 2 ;;
    --matrix-dir) MATRIX_DIR="$2"; shift 2 ;;
    --negative-ratio) NEGATIVE_RATIO="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --n-trials) N_TRIALS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$SPECIES" ]] || { echo "--species is required" >&2; exit 1; }
[[ -d "$SAMPLES_DIR" ]] || { echo "Missing samples dir: $SAMPLES_DIR" >&2; exit 1; }
[[ -s "$GOLD" ]] || { echo "Missing reference labels: $GOLD" >&2; exit 1; }
[[ -n "$OUT_DIR" ]] || { echo "--out-dir is required" >&2; exit 1; }
[[ -d "$SCRIPTS_DIR" ]] || { echo "Missing scripts dir: $SCRIPTS_DIR" >&2; exit 1; }

mkdir -p "$OUT_DIR"
MATRIX_DIR="${MATRIX_DIR:-$OUT_DIR/matrices}"
mkdir -p "$MATRIX_DIR"

STEP1="$SCRIPTS_DIR/Step1-CreateRobustDataset.py"
STEP2="$SCRIPTS_DIR/Step2-TPE_byessearch-v2.py"
STEP3="$SCRIPTS_DIR/Step3-makeModel-v2.py"
STEP4="$SCRIPTS_DIR/Step4-PredictLargeData-v2.py"
MERGE="${MERGE:-$SCRIPTS_DIR/merge_tf_target_matrix_sqlite.py}"
ENHANCE="${ENHANCE:-$SCRIPTS_DIR/build_enhanced_tf_target_matrix.py}"
PROJECT_FEATURES="${PROJECT_FEATURES:-$SCRIPTS_DIR/build_project_group_features.py}"

for f in "$STEP1" "$STEP2" "$STEP3" "$STEP4"; do
  [[ -s "$f" ]] || { echo "Missing XGBoost step script: $f" >&2; exit 1; }
done

RAW_MATRIX="$MATRIX_DIR/tf_target_matrix.tsv"
ENHANCED_MATRIX="$MATRIX_DIR/tf_target_matrix.enhanced.tsv"
PROJECT_MATRIX="$MATRIX_DIR/tf_target_matrix.project_enhanced.tsv"

if [[ -s "$MERGE" ]]; then
  python "$MERGE" \
    --input-dir "$SAMPLES_DIR" \
    --pattern "*.tf_target.txt" \
    --output-tsv "$RAW_MATRIX" \
    --batch-size 50000 \
    --no-uppercase
elif [[ -s "$SCRIPTS_DIR/merge_tf_target_for_xgboost.py" ]]; then
  python "$SCRIPTS_DIR/merge_tf_target_for_xgboost.py" \
    --input_dir "$SAMPLES_DIR" \
    --pattern "*.tf_target.txt" \
    --output_tsv "$RAW_MATRIX"
else
  echo "No matrix merge script found. Create $RAW_MATRIX before running this wrapper." >&2
  exit 1
fi

if [[ -s "$ENHANCE" ]]; then
  python "$ENHANCE" \
    --input-tsv "$RAW_MATRIX" \
    --output-tsv "$ENHANCED_MATRIX" \
    --chunksize "$CHUNKSIZE"
else
  cp "$RAW_MATRIX" "$ENHANCED_MATRIX"
fi

if [[ -s "$PROJECT_FEATURES" ]]; then
  python "$PROJECT_FEATURES" \
    --input-tsv "$ENHANCED_MATRIX" \
    --output-tsv "$PROJECT_MATRIX" \
    --chunksize "$CHUNKSIZE"
else
  cp "$ENHANCED_MATRIX" "$PROJECT_MATRIX"
fi

python "$STEP1" \
  --feature_matrix "$PROJECT_MATRIX" \
  --gold_standard "$GOLD" \
  --output_dir "$OUT_DIR/step1" \
  --negative_ratio "$NEGATIVE_RATIO" \
  --folds "$FOLDS" \
  --random_seed "$SEED" \
  --chunksize 200000 \
  --batch_size 50000

python "$STEP2" \
  --train_dataset "$OUT_DIR/step1/train_dataset.tsv" \
  --fold_indices "$OUT_DIR/step1/fold_indices.tsv" \
  --output_dir "$OUT_DIR/step2" \
  --n_trials "$N_TRIALS" \
  --early_stopping_rounds "$EARLY_STOPPING" \
  --device "$DEVICE" \
  --seed "$SEED"

python "$STEP3" \
  --train_dataset "$OUT_DIR/step1/train_dataset.tsv" \
  --fold_indices "$OUT_DIR/step1/fold_indices.tsv" \
  --best_params_json "$OUT_DIR/step2/best_params.json" \
  --output_dir "$OUT_DIR/step3" \
  --early_stopping_rounds "$EARLY_STOPPING"

python "$STEP4" \
  --feature_matrix "$PROJECT_MATRIX" \
  --model_path "$OUT_DIR/step3/final_integrated_model.json" \
  --feature_columns_file "$OUT_DIR/step3/integrated_feature_columns.txt" \
  --output_tsv "$OUT_DIR/step4/final_regulatory_with_probability.tsv"

echo "Species-level network written to: $OUT_DIR/step4/final_regulatory_with_probability.tsv"
