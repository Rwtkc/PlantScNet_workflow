#!/usr/bin/env python3

import argparse
import csv
import math
import re
from collections import defaultdict


REP_RE = re.compile(r"(rep\d+)", re.IGNORECASE)


def detect_header(row):
    lowered = [x.strip().lower() for x in row]
    return lowered[:5] == ["cluster_id", "tf", "target_gene", "n_support_motifs", "n_support_peaks"]


def parse_input(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        first = next(reader, None)
        if first is None:
            return rows
        if not detect_header(first) and len(first) >= 5:
            rows.append(first[:5])
        for row in reader:
            if len(row) >= 5:
                rows.append(row[:5])
    return rows


def rep_from_cluster_id(cluster_id):
    match = REP_RE.search(cluster_id)
    return match.group(1) if match else "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    agg = defaultdict(
        lambda: {
            "clusters": set(),
            "reps": set(),
            "total_support_motifs": 0,
            "total_support_peaks": 0,
            "max_support_motifs": 0,
            "max_support_peaks": 0,
        }
    )

    for cluster_id, tf, target_gene, n_motifs, n_peaks in parse_input(args.input):
        n_motifs = int(n_motifs)
        n_peaks = int(n_peaks)
        bucket = agg[(tf, target_gene)]
        bucket["clusters"].add(cluster_id)
        bucket["reps"].add(rep_from_cluster_id(cluster_id))
        bucket["total_support_motifs"] += n_motifs
        bucket["total_support_peaks"] += n_peaks
        bucket["max_support_motifs"] = max(bucket["max_support_motifs"], n_motifs)
        bucket["max_support_peaks"] = max(bucket["max_support_peaks"], n_peaks)

    ranked = []
    for (tf, target_gene), payload in agg.items():
        cluster_support = len(payload["clusters"])
        total_support_peaks = payload["total_support_peaks"]
        importance_score = cluster_support * math.log2(1 + total_support_peaks)
        ranked.append(
            {
                "TF": tf,
                "target": target_gene,
                "importance_score": importance_score,
                "rep_support": len(payload["reps"]),
                "cluster_support": cluster_support,
                "total_support_peaks": total_support_peaks,
                "total_support_motifs": payload["total_support_motifs"],
                "max_support_peaks": payload["max_support_peaks"],
                "max_support_motifs": payload["max_support_motifs"],
                "cluster_list": ",".join(sorted(payload["clusters"])),
            }
        )

    ranked.sort(
        key=lambda x: (
            -x["importance_score"],
            -x["rep_support"],
            -x["cluster_support"],
            -x["total_support_peaks"],
            -x["max_support_peaks"],
            -x["total_support_motifs"],
            -x["max_support_motifs"],
            x["TF"],
            x["target"],
        )
    )

    full_out = f"{args.output_prefix}_ranked_full.tsv"
    simple_out = f"{args.output_prefix}_ranked_scplant_like.tsv"

    with open(full_out, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow([
            "rank",
            "TF",
            "target",
            "importance_score",
            "rep_support",
            "cluster_support",
            "total_support_peaks",
            "total_support_motifs",
            "max_support_peaks",
            "max_support_motifs",
            "cluster_list",
        ])
        for i, row in enumerate(ranked, start=1):
            writer.writerow([
                i,
                row["TF"],
                row["target"],
                f"{row['importance_score']:.6f}",
                row["rep_support"],
                row["cluster_support"],
                row["total_support_peaks"],
                row["total_support_motifs"],
                row["max_support_peaks"],
                row["max_support_motifs"],
                row["cluster_list"],
            ])

    with open(simple_out, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["TF", "target", "importance_score"])
        for row in ranked:
            writer.writerow([row["TF"], row["target"], f"{row['importance_score']:.6f}"])

    print(full_out)
    print(simple_out)


if __name__ == "__main__":
    main()
