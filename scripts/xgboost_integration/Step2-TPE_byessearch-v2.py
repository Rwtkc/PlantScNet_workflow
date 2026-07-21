#!/usr/bin/env python3
"""Tune integrated XGBoost model with Optuna using fixed 5-fold splits."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune integrated model hyperparameters with Optuna + fixed folds."
    )
    parser.add_argument("--train_dataset", required=True, help="Step1 train_dataset.tsv")
    parser.add_argument("--fold_indices", required=True, help="Step1 fold_indices.tsv")
    parser.add_argument("--output_dir", required=True, help="Output directory for Step2")
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna trial count")
    parser.add_argument(
        "--early_stopping_rounds", type=int, default=50, help="Early stopping rounds"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="XGBoost device",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=0,
        help="XGBoost n_jobs (0 means use all threads).",
    )
    return parser.parse_args()


def _require_runtime_deps():
    try:
        import optuna  # noqa: F401
        import xgboost  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Missing runtime dependency. Install xgboost and optuna on the server "
            "before running Step2."
        ) from e


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    pos = int(y_true.sum())
    neg = int((1 - y_true).sum())
    if pos == 0 or neg == 0:
        return 0.5

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    rank_sum_pos = float(ranks[y_true == 1].sum())
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def average_precision_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    pos = int(y_true.sum())
    if pos == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos

    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return float(ap)


def load_data(train_dataset_path: str, fold_indices_path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(train_dataset_path, sep="\t")
    folds = pd.read_csv(fold_indices_path, sep="\t")

    required_train = {"source_row_id", "TF", "target", "label"}
    missing_train = required_train - set(df.columns)
    if missing_train:
        raise ValueError(f"train_dataset missing columns: {sorted(missing_train)}")

    required_folds = {"source_row_id", "fold"}
    missing_folds = required_folds - set(folds.columns)
    if missing_folds:
        raise ValueError(f"fold_indices missing columns: {sorted(missing_folds)}")

    merged = df.merge(folds[["source_row_id", "fold"]], on="source_row_id", how="inner")
    if merged.empty:
        raise ValueError("Merged dataset is empty. Check Step1 outputs.")

    feature_cols = [
        c for c in merged.columns if c not in ("source_row_id", "TF", "target", "label", "fold")
    ]
    if not feature_cols:
        raise ValueError("No feature columns found in train dataset.")
    return merged, feature_cols


def evaluate_cv_auc_ap(
    xgb,
    data: pd.DataFrame,
    feature_cols: List[str],
    params: Dict,
    early_stopping_rounds: int,
) -> Tuple[float, float]:
    auc_scores: List[float] = []
    ap_scores: List[float] = []
    fold_ids = sorted(data["fold"].unique().tolist())
    for fold_id in fold_ids:
        tr = data[data["fold"] != fold_id]
        va = data[data["fold"] == fold_id]

        x_train = tr[feature_cols].to_numpy(dtype=np.float32)
        y_train = tr["label"].to_numpy(dtype=np.int32)
        x_valid = va[feature_cols].to_numpy(dtype=np.float32)
        y_valid = va["label"].to_numpy(dtype=np.int32)

        model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        y_prob = model.predict_proba(x_valid)[:, 1]

        auc_scores.append(roc_auc_score_np(y_valid, y_prob))
        ap_scores.append(average_precision_score_np(y_valid, y_prob))

    return float(np.mean(auc_scores)), float(np.mean(ap_scores))


def main() -> None:
    args = parse_args()
    _require_runtime_deps()
    import optuna
    import xgboost as xgb

    os.makedirs(args.output_dir, exist_ok=True)
    data, feature_cols = load_data(args.train_dataset, args.fold_indices)

    pos_count = int((data["label"] == 1).sum())
    neg_count = int((data["label"] == 0).sum())
    scale_pos_weight_default = float(neg_count / max(pos_count, 1))

    fixed = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": args.device,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
    }

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 2e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                max(1.0, scale_pos_weight_default * 0.5),
                max(2.0, scale_pos_weight_default * 1.5),
            ),
        }
        params = {**fixed, **trial_params}
        mean_auc, mean_ap = evaluate_cv_auc_ap(
            xgb=xgb,
            data=data,
            feature_cols=feature_cols,
            params=params,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        trial.set_user_attr("mean_auc", mean_auc)
        trial.set_user_attr("mean_ap", mean_ap)
        return mean_auc

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    best_trial = study.best_trial
    best_params = {**fixed, **best_trial.params}
    result = {
        "best_value_mean_auc": float(best_trial.value),
        "best_trial_number": int(best_trial.number),
        "best_params": best_params,
        "best_mean_ap": float(best_trial.user_attrs.get("mean_ap", 0.0)),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "feature_columns": feature_cols,
        "n_trials": int(args.n_trials),
        "seed": int(args.seed),
        "device": args.device,
    }

    best_params_path = os.path.join(args.output_dir, "best_params.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    rows = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "objective_mean_auc": t.value,
            "mean_auc": t.user_attrs.get("mean_auc"),
            "mean_ap": t.user_attrs.get("mean_ap"),
            "state": str(t.state),
        }
        row.update(t.params)
        rows.append(row)
    trials_path = os.path.join(args.output_dir, "optuna_trials.tsv")
    pd.DataFrame(rows).to_csv(trials_path, sep="\t", index=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
