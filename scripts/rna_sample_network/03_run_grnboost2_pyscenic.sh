#!/usr/bin/env bash
set -euo pipefail

WORKDIR="result"
TF_LIST=""
RANKINGS=""
MOTIF2TF=""
WORKERS="${WORKERS:-10}"
SEED="${SEED:-777}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
POST1="$SCRIPT_DIR/GRN_postProcess1.py"
POST2="$SCRIPT_DIR/GRN_postProcess2_Rwtkc_args.R"
GROUP="${GROUP:-seurat_clusters}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) WORKDIR="$2"; shift 2 ;;
    --tf-list) TF_LIST="$2"; shift 2 ;;
    --rankings) RANKINGS="$2"; shift 2 ;;
    --motif2tf) MOTIF2TF="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --group) GROUP="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

for f in "$WORKDIR/exprMat.loom" "$TF_LIST" "$RANKINGS" "$MOTIF2TF"; do
  [[ -s "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
done

TF_LIST="$(realpath "$TF_LIST")"
RANKINGS="$(realpath "$RANKINGS")"
MOTIF2TF="$(realpath "$MOTIF2TF")"
POST1="$(realpath "$POST1")"
POST2="$(realpath "$POST2")"

cd "$WORKDIR"

if [[ ! -s "$POST1" ]]; then
  echo "Missing postprocess script: $POST1." >&2
  exit 1
fi
if [[ ! -s "$POST2" ]]; then
  echo "Missing postprocess script: $POST2." >&2
  exit 1
fi

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

arboreto_with_multiprocessing.py \
  ./exprMat.loom \
  "$TF_LIST" \
  --method grnboost2 \
  --output ./adj.tsv \
  --num_workers "$WORKERS" \
  --seed "$SEED"

pyscenic ctx \
  ./adj.tsv \
  "$RANKINGS" \
  --annotations_fname "$MOTIF2TF" \
  --expression_mtx_fname ./exprMat.loom \
  --no_pruning \
  --output ./reg.tsv \
  --num_workers "$WORKERS"

pyscenic aucell \
  ./exprMat.loom \
  ./reg.tsv \
  --output ./pyscenicOutput.loom \
  --num_workers "$WORKERS" \
  --seed "$SEED"

python3 "$POST1" ./pyscenicOutput.loom ./reg.tsv "$WORKERS" 1 .
Rscript --vanilla "$POST2" --scenicOutput . --group "$GROUP" -o .

test -s ./tf_target.txt
echo "RNA sample-level network written to: $(pwd)/tf_target.txt"
