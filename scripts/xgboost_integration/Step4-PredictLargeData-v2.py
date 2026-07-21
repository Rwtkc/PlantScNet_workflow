#!/usr/bin/env python3
"""Predict probability for all edges and export sorted result."""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict all TF-target edges and export Probability-ranked TSV."
    )
    parser.add_argument("--feature_matrix", required=True, help="Path to tf_target_matrix.tsv")
    parser.add_argument("--model_path", required=True, help="Path to final_integrated_model.json")
    parser.add_argument("--output_tsv", required=True, help="Output TSV with Probability")
    parser.add_argument(
        "--feature_columns_file",
        default="",
        help="Optional integrated_feature_columns.txt from Step3 for strict column order.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Prediction chunk size",
    )
    parser.add_argument(
        "--temp_db",
        default="",
        help="Optional sqlite path for temporary sorting cache.",
    )
    parser.add_argument(
        "--keep_temp_db",
        action="store_true",
        help="Keep temp sqlite cache after export.",
    )
    return parser.parse_args()


def _require_runtime_deps():
    try:
        import xgboost  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Missing runtime dependency. Install xgboost on the server before running Step4."
        ) from e


def _load_feature_columns(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        cols = [line.strip() for line in f if line.strip()]
    if not cols:
        raise ValueError(f"feature_columns_file is empty: {path}")
    return cols


def main() -> None:
    args = parse_args()
    _require_runtime_deps()
    import xgboost as xgb

    out_dir = os.path.dirname(args.output_tsv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    temp_db = args.temp_db or (args.output_tsv + ".tmp.sqlite")
    db_is_default = not bool(args.temp_db)
    if os.path.exists(temp_db):
        os.remove(temp_db)

    conn = sqlite3.connect(temp_db)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        """
        CREATE TABLE preds (
            tf TEXT NOT NULL,
            target TEXT NOT NULL,
            prob REAL NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX idx_preds_prob ON preds(prob);")

    model = xgb.XGBClassifier()
    model.load_model(args.model_path)

    expected_cols = _load_feature_columns(args.feature_columns_file) if args.feature_columns_file else None

    total_rows = 0
    for i, chunk in enumerate(pd.read_csv(args.feature_matrix, sep="\t", chunksize=args.chunksize), start=1):
        if "TF" not in chunk.columns or "target" not in chunk.columns:
            raise ValueError("feature_matrix must include TF and target columns.")

        if expected_cols is None:
            feature_cols = [c for c in chunk.columns if c not in ("TF", "target")]
        else:
            missing = [c for c in expected_cols if c not in chunk.columns]
            if missing:
                raise ValueError(f"Missing expected feature columns in input: {missing[:5]}")
            feature_cols = expected_cols

        x = chunk[feature_cols].to_numpy(dtype=np.float32)
        probs = model.predict_proba(x)[:, 1]

        rows = list(
            zip(
                chunk["TF"].astype(str).tolist(),
                chunk["target"].astype(str).tolist(),
                probs.astype(float).tolist(),
            )
        )
        conn.executemany("INSERT INTO preds(tf, target, prob) VALUES (?, ?, ?)", rows)
        conn.commit()

        total_rows += len(chunk)
        print(f"[predict] chunk={i} rows={len(chunk)} total_rows={total_rows}")

    with open(args.output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["TF", "target", "Probability"])
        cur = conn.execute("SELECT tf, target, prob FROM preds ORDER BY prob DESC")
        for tf, target, prob in cur:
            writer.writerow([tf, target, f"{prob:.12g}"])

    conn.close()
    if db_is_default and not args.keep_temp_db and os.path.exists(temp_db):
        os.remove(temp_db)

    print(f"output={args.output_tsv}")
    print(f"rows={total_rows}")


if __name__ == "__main__":
    main()
