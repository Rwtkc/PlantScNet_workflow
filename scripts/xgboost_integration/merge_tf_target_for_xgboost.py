#!/usr/bin/env python3
"""Merge per-sample TF-target files into a sample-by-edge matrix.

Input files are expected to be tab-separated with columns:
TF    target    importance_score

Output matrix is tab-separated:
TF    target    <sample_1>    <sample_2> ...
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sqlite3
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge tf_target_*.txt files into a matrix (rows=edge, cols=sample)."
    )
    parser.add_argument(
        "--input_dir",
        default="./data/ath/result",
        help="Directory containing per-sample tf_target files.",
    )
    parser.add_argument(
        "--pattern",
        default="tf_target_*.txt",
        help="Glob pattern for input files.",
    )
    parser.add_argument(
        "--output_tsv",
        default="./data/ath/tf_target_matrix.tsv",
        help="Output matrix path.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["max", "sum"],
        default="max",
        help="How to combine duplicate TF-target rows inside the same sample file.",
    )
    parser.add_argument(
        "--missing_value",
        type=float,
        default=0.0,
        help="Fill value when a sample has no score for an edge.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50000,
        help="SQLite insert batch size.",
    )
    parser.add_argument(
        "--db_path",
        default="",
        help="Optional sqlite path. Default: <output_tsv>.tmp.sqlite",
    )
    parser.add_argument(
        "--keep_db",
        action="store_true",
        help="Keep temporary sqlite file after writing output.",
    )
    return parser.parse_args()


def list_input_files(input_dir: str, pattern: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files found in '{input_dir}' with pattern '{pattern}'."
        )
    return paths


def sample_id_from_path(path: str) -> str:
    name = os.path.basename(path)
    if name.startswith("tf_target_"):
        name = name[len("tf_target_") :]
    if name.endswith(".txt"):
        name = name[:-4]
    return name


def iter_tf_target(path: str) -> Iterable[Tuple[str, str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"TF", "target", "importance_score"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"File '{path}' must contain columns: TF, target, importance_score"
            )
        for row in reader:
            tf = row["TF"].strip()
            target = row["target"].strip()
            if not tf or not target:
                continue
            try:
                score = float(row["importance_score"])
            except (TypeError, ValueError):
                continue
            yield tf, target, score


def aggregate_sample_edges(path: str, aggregation: str) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for tf, target, score in iter_tf_target(path):
        key = (tf, target)
        if key not in out:
            out[key] = score
            continue
        if aggregation == "max":
            if score > out[key]:
                out[key] = score
        else:
            out[key] += score
    return out


def init_db(db_path: str) -> sqlite3.Connection:
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute(
        """
        CREATE TABLE edges (
            tf TEXT NOT NULL,
            target TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (tf, target, sample_idx)
        );
        """
    )
    conn.execute("CREATE INDEX idx_edges_pair ON edges(tf, target);")
    return conn


def load_into_db(
    conn: sqlite3.Connection,
    input_files: List[str],
    aggregation: str,
    batch_size: int,
) -> List[str]:
    sample_ids: List[str] = []
    insert_sql = "INSERT INTO edges(tf, target, sample_idx, score) VALUES (?, ?, ?, ?)"

    for sample_idx, path in enumerate(input_files):
        sample_id = sample_id_from_path(path)
        sample_ids.append(sample_id)

        aggregated = aggregate_sample_edges(path, aggregation)
        batch: List[Tuple[str, str, int, float]] = []
        cur = conn.cursor()
        for (tf, target), score in aggregated.items():
            batch.append((tf, target, sample_idx, score))
            if len(batch) >= batch_size:
                cur.executemany(insert_sql, batch)
                batch.clear()
        if batch:
            cur.executemany(insert_sql, batch)
        conn.commit()
        print(
            f"[load] {sample_idx + 1}/{len(input_files)} "
            f"{os.path.basename(path)} edges={len(aggregated)}"
        )

    return sample_ids


def _format_number(value: float) -> str:
    return f"{value:.12g}"


def write_matrix(
    conn: sqlite3.Connection,
    sample_ids: List[str],
    output_tsv: str,
    missing_value: float,
) -> None:
    out_dir = os.path.dirname(output_tsv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    query = """
        SELECT tf, target, sample_idx, score
        FROM edges
        ORDER BY tf, target, sample_idx
    """

    missing_text = _format_number(missing_value)
    n_samples = len(sample_ids)
    with open(output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["TF", "target", *sample_ids])

        current_pair: Tuple[str, str] | None = None
        values: List[str] = [missing_text] * n_samples
        row_count = 0

        for tf, target, sample_idx, score in conn.execute(query):
            pair = (tf, target)
            if current_pair != pair:
                if current_pair is not None:
                    writer.writerow([current_pair[0], current_pair[1], *values])
                    row_count += 1
                current_pair = pair
                values = [missing_text] * n_samples

            values[sample_idx] = _format_number(score)

        if current_pair is not None:
            writer.writerow([current_pair[0], current_pair[1], *values])
            row_count += 1

    print(f"[write] rows={row_count}, samples={n_samples}, output={output_tsv}")


def build_matrix(
    input_dir: str,
    pattern: str,
    output_tsv: str,
    aggregation: str = "max",
    missing_value: float = 0.0,
    batch_size: int = 50000,
    db_path: str | None = None,
    keep_db: bool = False,
) -> None:
    input_files = list_input_files(input_dir, pattern)

    db_is_temp = db_path is None
    if db_path is None:
        db_path = output_tsv + ".tmp.sqlite"

    conn = init_db(db_path)
    try:
        sample_ids = load_into_db(conn, input_files, aggregation, batch_size)
        write_matrix(conn, sample_ids, output_tsv, missing_value)
    finally:
        conn.close()

    if db_is_temp and not keep_db and os.path.exists(db_path):
        os.remove(db_path)


def main() -> None:
    args = parse_args()
    db_path = args.db_path if args.db_path else None
    build_matrix(
        input_dir=args.input_dir,
        pattern=args.pattern,
        output_tsv=args.output_tsv,
        aggregation=args.aggregation,
        missing_value=args.missing_value,
        batch_size=args.batch_size,
        db_path=db_path,
        keep_db=args.keep_db,
    )


if __name__ == "__main__":
    main()
