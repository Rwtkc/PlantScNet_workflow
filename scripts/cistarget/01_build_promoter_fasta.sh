#!/usr/bin/env bash
set -euo pipefail

GTF=""
GENOME=""
OUT_PREFIX="promoters_2kb"
UPSTREAM=2000
GENE_ID_REGEX='gene_id "([^"]+)"'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gtf|--gff|--annotation) GTF="$2"; shift 2 ;;
    --genome) GENOME="$2"; shift 2 ;;
    --out-prefix) OUT_PREFIX="$2"; shift 2 ;;
    --upstream) UPSTREAM="$2"; shift 2 ;;
    --gene-id-regex) GENE_ID_REGEX="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -s "$GTF" ]] || { echo "Missing annotation file: $GTF" >&2; exit 1; }
[[ -s "$GENOME" ]] || { echo "Missing genome FASTA: $GENOME" >&2; exit 1; }

command -v samtools >/dev/null 2>&1 || { echo "samtools not found" >&2; exit 1; }
command -v bedtools >/dev/null 2>&1 || { echo "bedtools not found" >&2; exit 1; }

WORKDIR="$(dirname "$OUT_PREFIX")"
mkdir -p "$WORKDIR"

GENOME_FOR_RUN="$GENOME"
ANNOTATION_FOR_RUN="$GTF"

if [[ "$GENOME" == *.gz ]]; then
  GENOME_FOR_RUN="$WORKDIR/$(basename "${GENOME%.gz}")"
  if [[ ! -s "$GENOME_FOR_RUN" ]]; then
    gzip -dc "$GENOME" > "$GENOME_FOR_RUN"
  fi
fi

if [[ "$GTF" == *.gz ]]; then
  ANNOTATION_FOR_RUN="$WORKDIR/$(basename "${GTF%.gz}")"
  if [[ ! -s "$ANNOTATION_FOR_RUN" ]]; then
    gzip -dc "$GTF" > "$ANNOTATION_FOR_RUN"
  fi
fi

samtools faidx "$GENOME_FOR_RUN"

python3 - "$ANNOTATION_FOR_RUN" "$OUT_PREFIX.genes.bed" "$GENE_ID_REGEX" <<'PY'
import re
import sys
from pathlib import Path

annotation = Path(sys.argv[1])
out_bed = Path(sys.argv[2])
gene_id_regex = re.compile(sys.argv[3])

rows = []
with annotation.open() as fh:
    for line in fh:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9 or parts[2] != "gene":
            continue
        match = gene_id_regex.search(parts[8])
        if not match:
            continue
        chrom = parts[0]
        start0 = max(0, int(parts[3]) - 1)
        end1 = int(parts[4])
        gene = match.group(1)
        strand = parts[6] if parts[6] in {"+", "-"} else "+"
        rows.append((chrom, start0, end1, gene, ".", strand))

with out_bed.open("w") as out:
    for row in rows:
        out.write("\t".join(map(str, row)) + "\n")

print(f"gene_rows={len(rows)} output={out_bed}")
if len(rows) == 0:
    raise SystemExit("No gene rows parsed; check --gene-id-regex and annotation format")
PY

bedtools flank \
  -i "$OUT_PREFIX.genes.bed" \
  -g "$GENOME_FOR_RUN.fai" \
  -l "$UPSTREAM" \
  -r 0 \
  -s \
  > "$OUT_PREFIX.bed"

bedtools getfasta \
  -fi "$GENOME_FOR_RUN" \
  -bed "$OUT_PREFIX.bed" \
  -name \
  -s \
  > "$OUT_PREFIX.fa.raw"

python3 - "$OUT_PREFIX.fa.raw" "$OUT_PREFIX.fa" <<'PY'
import re
import sys
from pathlib import Path

inp = Path(sys.argv[1])
out = Path(sys.argv[2])
with inp.open() as fi, out.open("w") as fo:
    for line in fi:
        if line.startswith(">"):
            line = re.sub(r"::.*", "", line)
        fo.write(line)
PY

echo "Promoter BED: $OUT_PREFIX.bed"
echo "Promoter FASTA: $OUT_PREFIX.fa"
