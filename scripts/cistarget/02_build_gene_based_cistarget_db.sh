#!/usr/bin/env bash
set -euo pipefail

FASTA=""
MOTIF_DIR=""
MOTIF_LIST=""
CISTARGET_SCRIPT=""
OUT_PREFIX="output_db_genes/species_cistarget_db_genes"
THREADS="${THREADS:-30}"
GENE_EXTRACT_REGEX="${GENE_EXTRACT_REGEX:-#.*$}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fasta) FASTA="$2"; shift 2 ;;
    --motif-dir) MOTIF_DIR="$2"; shift 2 ;;
    --motif-list) MOTIF_LIST="$2"; shift 2 ;;
    --script) CISTARGET_SCRIPT="$2"; shift 2 ;;
    --out-prefix) OUT_PREFIX="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --gene-extract-regex) GENE_EXTRACT_REGEX="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

for f in "$FASTA" "$MOTIF_LIST" "$CISTARGET_SCRIPT"; do
  [[ -s "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
done
[[ -d "$MOTIF_DIR" ]] || { echo "Missing motif directory: $MOTIF_DIR" >&2; exit 1; }

mkdir -p "$(dirname "$OUT_PREFIX")"

python3 "$CISTARGET_SCRIPT" \
  -f "$FASTA" \
  -M "$MOTIF_DIR" \
  -m "$MOTIF_LIST" \
  -o "$OUT_PREFIX" \
  -t "$THREADS" \
  -g "$GENE_EXTRACT_REGEX"

echo "cisTarget database prefix: $OUT_PREFIX"
