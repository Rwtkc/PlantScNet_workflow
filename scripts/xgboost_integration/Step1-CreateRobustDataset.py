#!/usr/bin/env python3
"""Build a robust train dataset from feature matrix and gold-standard edges."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sqlite3
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a robust train dataset: positives from gold standard, "
            "negatives sampled from non-gold edges."
        )
    )
    parser.add_argument(
        "--feature_matrix",
        required=True,
        help="Path to tf_target_matrix.tsv",
    )
    parser.add_argument(
        "--gold_standard",
        required=True,
        help="Path to ath_verified_regulatory_relations.tsv",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--negative_ratio",
        type=int,
        default=5,
        help="Negatives per positive (default: 5).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified folds (default: 5).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for sampling/splits.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Feature matrix read chunk size.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50000,
        help="SQLite batch insert size.",
    )
    parser.add_argument(
        "--temp_db",
        default="",
        help="Optional sqlite path. Default: <output_dir>/step1_cache.sqlite",
    )
    parser.add_argument(
        "--keep_temp_db",
        action="store_true",
        help="Keep sqlite cache after Step1 finishes.",
    )
    return parser.parse_args()


def _require_columns(cols: Sequence[str], required: Sequence[str], file_path: str) -> None:
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{file_path} missing required columns: {missing}")


def _iter_tsv_rows(path: str) -> Iterable[List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row:
                yield row


def build_gold_table(conn: sqlite3.Connection, gold_path: str, batch_size: int) -> int:
    conn.execute("DROP TABLE IF EXISTS gold_edges")
    conn.execute(
        """
        CREATE TABLE gold_edges (
            tf TEXT NOT NULL,
            target TEXT NOT NULL,
            PRIMARY KEY (tf, target)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gold_tf_target ON gold_edges(tf, target)")

    rows_iter = _iter_tsv_rows(gold_path)
    header = next(rows_iter)
    _require_columns(header, ["TF", "target"], gold_path)
    idx_tf = header.index("TF")
    idx_target = header.index("target")

    insert_sql = "INSERT OR IGNORE INTO gold_edges(tf, target) VALUES (?, ?)"
    batch: List[tuple[str, str]] = []
    for row in rows_iter:
        if len(row) <= max(idx_tf, idx_target):
            continue
        tf = row[idx_tf].strip()
        target = row[idx_target].strip()
        if not tf or not target:
            continue
        batch.append((tf, target))
        if len(batch) >= batch_size:
            conn.executemany(insert_sql, batch)
            batch.clear()
    if batch:
        conn.executemany(insert_sql, batch)
    conn.commit()

    return int(conn.execute("SELECT COUNT(*) FROM gold_edges").fetchone()[0])


def label_rows(
    conn: sqlite3.Connection,
    feature_matrix_path: str,
    chunksize: int,
    batch_size: int,
) -> tuple[int, list[str]]:
    conn.execute("DROP TABLE IF EXISTS labels")
    conn.execute(
        """
        CREATE TABLE labels (
            row_id INTEGER PRIMARY KEY,
            label INTEGER NOT NULL
        );
        """
    )

    total_rows = 0
    feature_cols: list[str] | None = None
    chunk_id = 0
    for chunk in pd.read_csv(feature_matrix_path, sep="\t", chunksize=chunksize):
        chunk_id += 1
        _require_columns(chunk.columns.tolist(), ["TF", "target"], feature_matrix_path)

        if feature_cols is None:
            feature_cols = [c for c in chunk.columns if c not in ("TF", "target")]
            if not feature_cols:
                raise ValueError("Feature matrix has no sample feature columns.")

        n = len(chunk)
        row_ids = np.arange(total_rows, total_rows + n, dtype=np.int64)
        pairs = list(
            zip(
                row_ids.tolist(),
                chunk["TF"].astype(str).tolist(),
                chunk["target"].astype(str).tolist(),
            )
        )

        conn.execute("DROP TABLE IF EXISTS chunk_edges")
        conn.execute(
            """
            CREATE TEMP TABLE chunk_edges (
                row_id INTEGER PRIMARY KEY,
                tf TEXT NOT NULL,
                target TEXT NOT NULL
            );
            """
        )
        insert_sql = "INSERT INTO chunk_edges(row_id, tf, target) VALUES (?, ?, ?)"
        for i in range(0, len(pairs), batch_size):
            conn.executemany(insert_sql, pairs[i : i + batch_size])

        conn.execute(
            """
            INSERT INTO labels(row_id, label)
            SELECT c.row_id,
                   CASE WHEN g.tf IS NULL THEN 0 ELSE 1 END AS label
            FROM chunk_edges c
            LEFT JOIN gold_edges g
              ON g.tf = c.tf AND g.target = c.target
            ORDER BY c.row_id
            """
        )
        conn.commit()

        total_rows += n
        print(f"[label] chunk={chunk_id} rows={n} total_rows={total_rows}")

    if feature_cols is None:
        raise ValueError("Feature matrix is empty.")
    return total_rows, feature_cols


def reservoir_sample_negative_row_ids(
    conn: sqlite3.Connection, target_size: int, seed: int
) -> list[int]:
    rng = random.Random(seed)
    selected: list[int] = []
    seen = 0
    cur = conn.execute("SELECT row_id FROM labels WHERE label = 0 ORDER BY row_id")
    for (row_id,) in cur:
        seen += 1
        if len(selected) < target_size:
            selected.append(int(row_id))
        else:
            j = rng.randint(1, seen)
            if j <= target_size:
                selected[j - 1] = int(row_id)
    selected.sort()
    return selected


def export_train_dataset(
    conn: sqlite3.Connection,
    feature_matrix_path: str,
    out_path: str,
    negative_ratio: int,
    seed: int,
    chunksize: int,
) -> tuple[int, int, int]:
    pos_count = int(conn.execute("SELECT COUNT(*) FROM labels WHERE label = 1").fetchone()[0])
    neg_count = int(conn.execute("SELECT COUNT(*) FROM labels WHERE label = 0").fetchone()[0])
    if pos_count == 0:
        raise ValueError("No positive edges found: gold standard has no overlap with feature matrix.")

    target_neg = min(neg_count, pos_count * negative_ratio)
    sampled_neg = reservoir_sample_negative_row_ids(conn, target_neg, seed)
    sampled_neg_set = set(sampled_neg)

    positive_ids = {
        int(r[0]) for r in conn.execute("SELECT row_id FROM labels WHERE label = 1")
    }
    selected_ids = positive_ids | sampled_neg_set
    selected_array = np.array(sorted(selected_ids), dtype=np.int64)

    total_rows = 0
    written_rows = 0
    header_written = False
    for chunk in pd.read_csv(feature_matrix_path, sep="\t", chunksize=chunksize):
        n = len(chunk)
        row_ids = np.arange(total_rows, total_rows + n, dtype=np.int64)
        keep_mask = np.isin(row_ids, selected_array)
        if keep_mask.any():
            out_chunk = chunk.loc[keep_mask].copy()
            kept_row_ids = row_ids[keep_mask]
            out_chunk.insert(0, "source_row_id", kept_row_ids)
            out_chunk["label"] = [
                1 if int(rid) in positive_ids else 0 for rid in kept_row_ids.tolist()
            ]
            mode = "w" if not header_written else "a"
            out_chunk.to_csv(
                out_path,
                sep="\t",
                index=False,
                mode=mode,
                header=not header_written,
            )
            header_written = True
            written_rows += len(out_chunk)
        total_rows += n

    return pos_count, target_neg, written_rows


def build_fold_indices(
    train_dataset_path: str, fold_path: str, n_folds: int, seed: int
) -> None:
    df = pd.read_csv(train_dataset_path, sep="\t", usecols=["source_row_id", "label"])
    y = df["label"].to_numpy()
    if len(df) < n_folds:
        raise ValueError(f"Dataset size {len(df)} is smaller than n_folds={n_folds}.")
    if np.unique(y).size < 2:
        raise ValueError("Need both positive and negative samples for StratifiedKFold.")

    rng = random.Random(seed)
    fold = np.zeros(len(df), dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0].tolist()
        rng.shuffle(idx)
        for i, pos in enumerate(idx):
            fold[pos] = i % n_folds

    out = df.copy()
    out["fold"] = fold
    out.to_csv(fold_path, sep="\t", index=False)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset_path = os.path.join(args.output_dir, "train_dataset.tsv")
    fold_path = os.path.join(args.output_dir, "fold_indices.tsv")
    summary_path = os.path.join(args.output_dir, "step1_summary.json")

    temp_db = args.temp_db or os.path.join(args.output_dir, "step1_cache.sqlite")
    db_is_default = not bool(args.temp_db)

    conn = sqlite3.connect(temp_db)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    try:
        gold_size = build_gold_table(conn, args.gold_standard, args.batch_size)
        total_rows, feature_cols = label_rows(
            conn,
            args.feature_matrix,
            chunksize=args.chunksize,
            batch_size=args.batch_size,
        )
        pos_count, neg_count, written_rows = export_train_dataset(
            conn,
            feature_matrix_path=args.feature_matrix,
            out_path=train_dataset_path,
            negative_ratio=args.negative_ratio,
            seed=args.random_seed,
            chunksize=args.chunksize,
        )
    finally:
        conn.close()

    build_fold_indices(
        train_dataset_path=train_dataset_path,
        fold_path=fold_path,
        n_folds=args.folds,
        seed=args.random_seed,
    )

    summary = {
        "feature_matrix": args.feature_matrix,
        "gold_standard": args.gold_standard,
        "feature_columns": len(feature_cols),
        "feature_rows": int(total_rows),
        "gold_edges": int(gold_size),
        "positive_rows": int(pos_count),
        "negative_rows_sampled": int(neg_count),
        "train_rows_written": int(written_rows),
        "folds": int(args.folds),
        "negative_ratio": int(args.negative_ratio),
        "random_seed": int(args.random_seed),
        "train_dataset_path": train_dataset_path,
        "fold_indices_path": fold_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if db_is_default and not args.keep_temp_db and os.path.exists(temp_db):
        os.remove(temp_db)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
