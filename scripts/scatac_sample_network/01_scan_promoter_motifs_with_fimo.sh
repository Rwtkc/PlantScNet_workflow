#!/usr/bin/env bash
set -euo pipefail

PROMOTER_FASTA=""
PROMOTER_BED=""
MOTIF_MEME=""
OUTDIR="resources/ath/fimo"
OUTPUT_BED="resources/ath/motif_hits.bed"
THRESH="${THRESH:-1e-4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --promoter-fasta) PROMOTER_FASTA="$2"; shift 2 ;;
    --promoter-bed) PROMOTER_BED="$2"; shift 2 ;;
    --motif-meme) MOTIF_MEME="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --output-bed) OUTPUT_BED="$2"; shift 2 ;;
    --thresh) THRESH="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -s "$PROMOTER_FASTA" ]] || { echo "Missing promoter FASTA: $PROMOTER_FASTA" >&2; exit 1; }
[[ -s "$PROMOTER_BED" ]] || { echo "Missing promoter BED: $PROMOTER_BED" >&2; exit 1; }
[[ -s "$MOTIF_MEME" ]] || { echo "Missing motif MEME file: $MOTIF_MEME" >&2; exit 1; }

command -v fimo >/dev/null 2>&1 || { echo "fimo not found. Install MEME Suite first." >&2; exit 1; }

mkdir -p "$OUTDIR"
mkdir -p "$(dirname "$OUTPUT_BED")"

fimo \
  --oc "$OUTDIR" \
  --thresh "$THRESH" \
  --verbosity 1 \
  "$MOTIF_MEME" \
  "$PROMOTER_FASTA"

python3 - "$OUTDIR/fimo.tsv" "$PROMOTER_BED" "$OUTPUT_BED" <<'PY'
import csv
import sys
from pathlib import Path

fimo_tsv = Path(sys.argv[1])
promoter_bed = Path(sys.argv[2])
output_bed = Path(sys.argv[3])

promoters = {}
with promoter_bed.open() as handle:
    for line in handle:
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 4:
            continue
        chrom = fields[0]
        start = int(fields[1])
        end = int(fields[2])
        gene = fields[3]
        strand = fields[5] if len(fields) >= 6 and fields[5] in {"+", "-"} else "+"
        promoters[gene] = (chrom, start, end, strand)

rows = []
with fimo_tsv.open() as handle:
    reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
    for row in reader:
        motif_id = (row.get("motif_id") or "").strip()
        gene = (row.get("sequence_name") or "").strip()
        if not motif_id or gene not in promoters:
            continue
        chrom, promoter_start, promoter_end, promoter_strand = promoters[gene]
        start_1 = int(row["start"])
        stop_1 = int(row["stop"])

        if promoter_strand == "-":
            hit_start = promoter_end - stop_1
            hit_end = promoter_end - start_1 + 1
        else:
            hit_start = promoter_start + start_1 - 1
            hit_end = promoter_start + stop_1

        score = row.get("score", ".") or "."
        fimo_strand = row.get("strand", ".") or "."
        rows.append((chrom, hit_start, hit_end, motif_id, score, fimo_strand, gene))

rows.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[6]))
with output_bed.open("w") as out:
    for row in rows:
        out.write("\t".join(map(str, row)) + "\n")

print(f"motif_hits={len(rows)} output={output_bed}")
if not rows:
    raise SystemExit("No motif hits were written; check motif IDs, promoter FASTA headers and FIMO output.")
PY

echo "Motif hit BED: $OUTPUT_BED"
