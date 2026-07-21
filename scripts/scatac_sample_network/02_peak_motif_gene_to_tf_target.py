#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


def normalize_header(name):
    return name.strip().lstrip("#")


def load_motif_tf_map(path):
    motif_to_tfs = defaultdict(set)
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        reader.fieldnames = [normalize_header(x) for x in (reader.fieldnames or [])]
        for row in reader:
            row = {normalize_header(k): (v or "").strip() for k, v in row.items()}
            motif = row.get("motif_id", "")
            tf = row.get("TF") or row.get("gene_id") or row.get("gene_name")
            if motif and tf:
                motif_to_tfs[motif].add(tf)
    return motif_to_tfs


def iter_intersect_lines(a_bed, b_bed, sorted_intersect=False):
    cmd = ["bedtools", "intersect", "-wa", "-wb"]
    if sorted_intersect:
        cmd.append("-sorted")
    cmd.extend(["-a", a_bed, "-b", b_bed])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip("\n")
    stderr = proc.stderr.read() if proc.stderr else ""
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"bedtools intersect failed for {a_bed} vs {b_bed}: {stderr}")


def count_bed_columns(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                return len(line.rstrip("\n").split("\t"))
    raise ValueError(f"No BED records found in {path}")


def collect_peak_map(a_bed, b_bed, b_value_col_in_b, sorted_intersect=False):
    a_cols = count_bed_columns(a_bed)
    value_col = a_cols + b_value_col_in_b
    peak_map = defaultdict(set)
    for line in iter_intersect_lines(a_bed, b_bed, sorted_intersect=sorted_intersect):
        fields = line.split("\t")
        if len(fields) <= value_col:
            continue
        peak = f"{fields[0]}:{fields[1]}-{fields[2]}"
        value = fields[value_col].strip()
        if value:
            peak_map[peak].add(value)
    return peak_map


def write_rows(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def process_bed(bed_path, motif_bed, promoter_bed, motif_tf_file, outdir, sorted_intersect=False):
    basename = os.path.basename(bed_path)
    cluster_id = os.path.splitext(basename)[0]

    motif_to_tfs = load_motif_tf_map(motif_tf_file)
    peak_to_motifs = collect_peak_map(bed_path, motif_bed, 3, sorted_intersect=sorted_intersect)
    peak_to_genes = collect_peak_map(bed_path, promoter_bed, 3, sorted_intersect=sorted_intersect)

    motif_target_rows = []
    tf_target_rows = []
    collapsed = defaultdict(lambda: {"motifs": set(), "peaks": set()})

    for peak in sorted(set(peak_to_motifs) & set(peak_to_genes)):
        for motif in sorted(peak_to_motifs[peak]):
            for gene in sorted(peak_to_genes[peak]):
                motif_target_rows.append((cluster_id, peak, motif, gene))
                for tf in sorted(motif_to_tfs.get(motif, [])):
                    tf_target_rows.append((cluster_id, peak, motif, tf, gene))
                    collapsed[(tf, gene)]["motifs"].add(motif)
                    collapsed[(tf, gene)]["peaks"].add(peak)

    collapsed_rows = []
    for (tf, gene), payload in sorted(collapsed.items()):
        collapsed_rows.append(
            (
                cluster_id,
                tf,
                gene,
                len(payload["motifs"]),
                len(payload["peaks"]),
                ",".join(sorted(payload["motifs"])),
                ",".join(sorted(payload["peaks"])),
            )
        )

    write_rows(
        os.path.join(outdir, f"{cluster_id}_motif_target.tsv"),
        ["cluster_id", "peak", "motif", "target_gene"],
        motif_target_rows,
    )
    write_rows(
        os.path.join(outdir, f"{cluster_id}_tf_target.tsv"),
        ["cluster_id", "peak", "motif", "TF", "target_gene"],
        tf_target_rows,
    )
    write_rows(
        os.path.join(outdir, f"{cluster_id}_tf_target_collapsed.tsv"),
        ["cluster_id", "TF", "target_gene", "n_support_motifs", "n_support_peaks", "motifs", "peaks"],
        collapsed_rows,
    )

    return (
        cluster_id,
        len(peak_to_motifs),
        len(peak_to_genes),
        len(set(peak_to_motifs) & set(peak_to_genes)),
        len(motif_target_rows),
        len(tf_target_rows),
        len(collapsed_rows),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing cluster-enriched BED files")
    parser.add_argument("--motif-bed", required=True, help="BED file with motif hits; fourth column is motif_id")
    parser.add_argument("--promoter-bed", required=True, help="Promoter BED; fourth column is target gene ID")
    parser.add_argument("--motif-tf-file", required=True, help="motif-to-TF table, e.g. ath.tbl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sorted-intersect", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bed_files = sorted(
        os.path.join(args.input_dir, name)
        for name in os.listdir(args.input_dir)
        if name.endswith(".bed")
    )
    if not bed_files:
        raise SystemExit(f"No .bed files found in {args.input_dir}")

    tasks = [
        (bed, args.motif_bed, args.promoter_bed, args.motif_tf_file, args.output_dir, args.sorted_intersect)
        for bed in bed_files
    ]

    if args.workers <= 1:
        results = [process_bed(*task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(process_bed, *task): task[0] for task in tasks}
            for future in as_completed(future_map):
                results.append(future.result())

    results = sorted(results, key=lambda x: x[0])
    write_rows(
        os.path.join(args.output_dir, "tf_target_manifest.tsv"),
        [
            "cluster_id",
            "peaks_with_motif",
            "peaks_with_promoter_gene",
            "peaks_with_both",
            "motif_target_rows",
            "tf_target_rows",
            "collapsed_tf_target_edges",
        ],
        results,
    )

    merged_path = os.path.join(args.output_dir, "high_confidence_tf_target_with_clusters.tsv")
    with open(merged_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["cluster_id", "TF", "target_gene", "n_support_motifs", "n_support_peaks"])
        for name in os.listdir(args.output_dir):
            if not name.endswith("_tf_target_collapsed.tsv"):
                continue
            with open(os.path.join(args.output_dir, name), "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    writer.writerow([
                        row["cluster_id"],
                        row["TF"],
                        row["target_gene"],
                        row["n_support_motifs"],
                        row["n_support_peaks"],
                    ])

    print(merged_path)


if __name__ == "__main__":
    main()
