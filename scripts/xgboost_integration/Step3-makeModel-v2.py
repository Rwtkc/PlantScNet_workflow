#!/usr/bin/env python3
"""Evaluate integrated/single-sample models and train final integrated model."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate integrated and single-sample models with fixed folds, "
            "plot AUC/AP figures, and train final integrated model."
        )
    )
    parser.add_argument("--train_dataset", required=True, help="Step1 train_dataset.tsv")
    parser.add_argument("--fold_indices", required=True, help="Step1 fold_indices.tsv")
    parser.add_argument("--best_params_json", required=True, help="Step2 best_params.json")
    parser.add_argument("--output_dir", required=True, help="Step3 output directory")
    parser.add_argument("--top_k", type=int, default=5, help="Top single-sample count by AUC")
    parser.add_argument(
        "--early_stopping_rounds", type=int, default=50, help="Early stopping rounds"
    )
    return parser.parse_args()


def _require_runtime_deps():
    try:
        import matplotlib  # noqa: F401
        import xgboost  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Missing runtime dependency. Install xgboost and matplotlib on the server "
            "before running Step3."
        ) from e


def roc_curve_np(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = y_true.astype(np.int64)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    pos = np.sum(y_sorted == 1)
    neg = np.sum(y_sorted == 0)
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / pos
    fpr = fp / neg
    return np.concatenate(([0.0], fpr)), np.concatenate(([0.0], tpr))


def pr_curve_np(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = y_true.astype(np.int64)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    pos = np.sum(y_sorted == 1)
    if pos == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    return np.concatenate(([0.0], recall)), np.concatenate(([1.0], precision))


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = roc_curve_np(y_true, y_score)
    return float(np.trapz(tpr, fpr))


def average_precision_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    recall, precision = pr_curve_np(y_true, y_score)
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def compute_random_pr_baseline(y_true: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == 1))


def build_auc_compare_curve_points(
    integrated_auc: float, top_df: pd.DataFrame
) -> Tuple[List[str], List[float]]:
    labels = ["Integrated"] + top_df["sample"].astype(str).tolist()
    values = [float(integrated_auc)] + top_df["mean_auc"].astype(float).tolist()
    return labels, values


def build_mean_roc_curve(
    roc_curves: List[Tuple[np.ndarray, np.ndarray]], n_points: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, n_points)
    if not roc_curves:
        return grid, np.zeros_like(grid)
    all_tpr = [np.interp(grid, fpr, tpr) for fpr, tpr in roc_curves]
    mean_tpr = np.mean(all_tpr, axis=0)
    return grid, mean_tpr


def build_auc_compare_roc_series(
    integrated_result: Dict, top_curve_results: List[Dict]
) -> List[Dict]:
    series = []
    x_int, y_int = build_mean_roc_curve(integrated_result["roc_curves"])
    series.append(
        {
            "label": f"Integrated (AUC={float(integrated_result['mean_auc']):.3f})",
            "x": x_int,
            "y": y_int,
            "linewidth": 2.6,
            "color": "black",
        }
    )
    for item in top_curve_results:
        x, y = build_mean_roc_curve(item["roc_curves"])
        series.append(
            {
                "label": f"{item['sample']} (AUC={float(item['mean_auc']):.3f})",
                "x": x,
                "y": y,
                "linewidth": 1.9,
                "color": None,
            }
        )
    return series


def load_data(train_dataset_path: str, fold_indices_path: str) -> pd.DataFrame:
    df = pd.read_csv(train_dataset_path, sep="\t")
    folds = pd.read_csv(fold_indices_path, sep="\t")
    merged = df.merge(folds[["source_row_id", "fold"]], on="source_row_id", how="inner")
    if merged.empty:
        raise ValueError("Merged train/folds dataset is empty.")
    return merged


def evaluate_cv_model(
    xgb,
    data: pd.DataFrame,
    feature_cols: List[str],
    model_params: Dict,
    early_stopping_rounds: int,
) -> Dict:
    fold_ids = sorted(data["fold"].unique().tolist())
    fold_metrics = []
    roc_curves = []
    pr_curves = []

    for fold_id in fold_ids:
        tr = data[data["fold"] != fold_id]
        va = data[data["fold"] == fold_id]
        x_train = tr[feature_cols].to_numpy(dtype=np.float32)
        y_train = tr["label"].to_numpy(dtype=np.int32)
        x_valid = va[feature_cols].to_numpy(dtype=np.float32)
        y_valid = va["label"].to_numpy(dtype=np.int32)

        model = xgb.XGBClassifier(**model_params, early_stopping_rounds=early_stopping_rounds)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        y_prob = model.predict_proba(x_valid)[:, 1]

        auc = roc_auc_score_np(y_valid, y_prob)
        ap = average_precision_score_np(y_valid, y_prob)
        fpr, tpr = roc_curve_np(y_valid, y_prob)
        recall, precision = pr_curve_np(y_valid, y_prob)
        roc_curves.append((fpr, tpr))
        pr_curves.append((recall, precision))
        fold_metrics.append(
            {
                "fold": int(fold_id),
                "auc": float(auc),
                "ap": float(ap),
                "best_iteration": int(getattr(model, "best_iteration", -1)),
            }
        )

    mean_auc = float(np.mean([m["auc"] for m in fold_metrics]))
    mean_ap = float(np.mean([m["ap"] for m in fold_metrics]))
    return {
        "feature_cols": feature_cols,
        "fold_metrics": fold_metrics,
        "mean_auc": mean_auc,
        "mean_ap": mean_ap,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
    }


def plot_integrated_roc(result: Dict, out_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for i, (fpr, tpr) in enumerate(result["roc_curves"], start=1):
        plt.plot(fpr, tpr, alpha=0.35, label=f"Fold {i}")

    grid = np.linspace(0, 1, 200)
    mean_tpr = []
    for fpr, tpr in result["roc_curves"]:
        mean_tpr.append(np.interp(grid, fpr, tpr))
    mean_tpr = np.mean(mean_tpr, axis=0)

    plt.plot(grid, mean_tpr, color="black", linewidth=2, label=f"Mean AUC={result['mean_auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Integrated Model ROC (5-fold CV)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_integrated_pr(result: Dict, out_path: str, random_baseline: float) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for i, (recall, precision) in enumerate(result["pr_curves"], start=1):
        plt.plot(recall, precision, alpha=0.35, label=f"Fold {i}")

    grid = np.linspace(0, 1, 200)
    mean_precision = []
    for recall, precision in result["pr_curves"]:
        mean_precision.append(np.interp(grid, recall, precision, left=1.0, right=precision[-1]))
    mean_precision = np.mean(mean_precision, axis=0)

    plt.plot(
        grid,
        mean_precision,
        color="black",
        linewidth=2,
        label=f"Mean AP={result['mean_ap']:.4f}",
    )
    plt.axhline(
        y=random_baseline,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Random baseline={random_baseline:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Integrated Model Precision-Recall (5-fold CV)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_auc_compare(integrated_result: Dict, top_curve_results: List[Dict], out_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    series = build_auc_compare_roc_series(integrated_result, top_curve_results)
    for line in series:
        plt.plot(
            line["x"],
            line["y"],
            linewidth=line["linewidth"],
            color=line["color"],
            label=line["label"],
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Integrated + Top5 Single-Sample ROC Curves")
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    _require_runtime_deps()
    import xgboost as xgb

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args.train_dataset, args.fold_indices)

    with open(args.best_params_json, "r", encoding="utf-8") as f:
        best = json.load(f)
    model_params = dict(best["best_params"])
    integrated_feature_cols = best.get("feature_columns")
    if not integrated_feature_cols:
        integrated_feature_cols = [
            c for c in data.columns if c not in ("source_row_id", "TF", "target", "label", "fold")
        ]

    integrated_result = evaluate_cv_model(
        xgb=xgb,
        data=data,
        feature_cols=integrated_feature_cols,
        model_params=model_params,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    integrated_fold_df = pd.DataFrame(integrated_result["fold_metrics"])
    integrated_fold_df.to_csv(
        os.path.join(args.output_dir, "integrated_cv_fold_metrics.tsv"),
        sep="\t",
        index=False,
    )

    sample_cols = [
        c for c in data.columns if c not in ("source_row_id", "TF", "target", "label", "fold")
    ]
    single_rows = []
    for i, col in enumerate(sample_cols, start=1):
        res = evaluate_cv_model(
            xgb=xgb,
            data=data,
            feature_cols=[col],
            model_params=model_params,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        single_rows.append(
            {
                "sample": col,
                "mean_auc": res["mean_auc"],
                "mean_ap": res["mean_ap"],
            }
        )
        print(f"[single] {i}/{len(sample_cols)} sample={col} auc={res['mean_auc']:.4f}")

    single_df = pd.DataFrame(single_rows).sort_values("mean_auc", ascending=False)
    single_df.to_csv(
        os.path.join(args.output_dir, "single_sample_cv_metrics.tsv"),
        sep="\t",
        index=False,
    )
    top_df = single_df.head(args.top_k).copy()
    top_df.to_csv(
        os.path.join(args.output_dir, "single_sample_topk_by_auc.tsv"),
        sep="\t",
        index=False,
    )

    top_curve_results = []
    for sample in top_df["sample"].tolist():
        sample_result = evaluate_cv_model(
            xgb=xgb,
            data=data,
            feature_cols=[sample],
            model_params=model_params,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        top_curve_results.append(
            {
                "sample": sample,
                "mean_auc": sample_result["mean_auc"],
                "roc_curves": sample_result["roc_curves"],
            }
        )
    random_baseline = compute_random_pr_baseline(data["label"].to_numpy(dtype=np.int32))

    plot_integrated_roc(
        integrated_result, os.path.join(args.output_dir, "fig_roc_integrated.png")
    )
    plot_integrated_pr(
        integrated_result,
        os.path.join(args.output_dir, "fig_pr_integrated.png"),
        random_baseline=random_baseline,
    )
    plot_auc_compare(
        integrated_result,
        top_curve_results,
        os.path.join(args.output_dir, "fig_auc_compare_integrated_vs_top5single.png"),
    )

    x_full = data[integrated_feature_cols].to_numpy(dtype=np.float32)
    y_full = data["label"].to_numpy(dtype=np.int32)
    final_model = xgb.XGBClassifier(**model_params)
    final_model.fit(x_full, y_full, verbose=False)

    final_model_path = os.path.join(args.output_dir, "final_integrated_model.json")
    final_model.save_model(final_model_path)
    feature_col_file = os.path.join(args.output_dir, "integrated_feature_columns.txt")
    with open(feature_col_file, "w", encoding="utf-8") as f:
        for col in integrated_feature_cols:
            f.write(col + "\n")

    summary = {
        "integrated_mean_auc": integrated_result["mean_auc"],
        "integrated_mean_ap": integrated_result["mean_ap"],
        "pr_random_baseline": random_baseline,
        "top_k": int(args.top_k),
        "top_k_samples": top_df["sample"].tolist(),
        "final_model_path": final_model_path,
        "feature_columns_file": feature_col_file,
    }
    with open(os.path.join(args.output_dir, "step3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
