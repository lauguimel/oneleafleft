"""Hyperparameter tuning for XGBoost core model via Optuna.

Optimizes validation PR-AUC (the main bottleneck) on core features
(hansen + spatial + infra).

Usage:
    conda activate deforest
    python scripts/tune_xgboost.py                         # 50 trials
    python scripts/tune_xgboost.py --n-trials 100
    python scripts/tune_xgboost.py --dataset data/features_250k_20260228.parquet

Output:
    data/optuna_results_YYYYMMDD.json   — best params + trial history
    data/tuned_model_YYYYMMDD.json      — best XGBoost model
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from train_xgboost import (
    load_dataset,
    find_latest_parquet,
    compute_scale_pos_weight,
    optimal_threshold,
    evaluate,
    run_shap,
)
from ablation_study import resolve_group

OUTPUT_DIR = PROJECT_DIR / "data"
CORE_GROUPS = ["hansen", "spatial", "infra"]


def main(dataset_path: Path | None, n_trials: int) -> None:
    tag = date.today().strftime("%Y%m%d")

    print("=" * 60)
    print(f"OPTUNA TUNING — {n_trials} trials, objective=val PR-AUC")
    print("=" * 60)

    # ── Load data ───────────────────────────────────────────────────────────
    print("\n[1] Loading dataset...")
    if dataset_path is None:
        dataset_path = find_latest_parquet(OUTPUT_DIR)
    train_df, val_df, test_df, all_feature_cols = load_dataset(dataset_path)

    core_cols = []
    for g in CORE_GROUPS:
        core_cols.extend(resolve_group(g, all_feature_cols))
    core_cols = list(dict.fromkeys(core_cols))
    print(f"  Core features: {len(core_cols)}")

    X_train = train_df[core_cols].values
    y_train = train_df["target"].values.astype(int)
    X_val = val_df[core_cols].values
    y_val = val_df["target"].values.astype(int)
    X_test = test_df[core_cols].values
    y_test = test_df["target"].values.astype(int)

    spw = compute_scale_pos_weight(y_train)

    # ── Optuna objective ────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        }

        model = xgb.XGBClassifier(
            n_estimators=2000,
            scale_pos_weight=spw,
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
            **params,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_proba)

        # Report intermediate value for pruning
        trial.set_user_attr("val_auc_roc", float(roc_auc_score(y_val, y_proba)))
        trial.set_user_attr("val_pr_auc", float(pr_auc))
        trial.set_user_attr("best_iteration", model.best_iteration)

        return pr_auc

    # ── Run optimization ────────────────────────────────────────────────────
    print(f"\n[2] Running {n_trials} Optuna trials...")
    study = optuna.create_study(direction="maximize", study_name="xgb_core_prauc")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n  Best trial #{best.number}:")
    print(f"    Val PR-AUC: {best.value:.4f}")
    print(f"    Val AUC-ROC: {best.user_attrs['val_auc_roc']:.4f}")
    print(f"    Best iteration: {best.user_attrs['best_iteration']}")
    print(f"    Params: {json.dumps(best.params, indent=4)}")

    # ── Retrain with best params ────────────────────────────────────────────
    print("\n[3] Retraining with best params...")
    best_model = xgb.XGBClassifier(
        n_estimators=2000,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        **best.params,
    )
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Evaluate ────────────────────────────────────────────────────────────
    print("\n[4] Evaluation with best model:")
    threshold = optimal_threshold(best_model, X_val, y_val)
    results = [
        evaluate(best_model, X_train, y_train, "train", threshold),
        evaluate(best_model, X_val, y_val, "val", threshold),
        evaluate(best_model, X_test, y_test, "test", threshold),
    ]

    # ── Save ────────────────────────────────────────────────────────────────
    model_path = OUTPUT_DIR / f"tuned_model_{tag}.json"
    best_model.save_model(str(model_path))
    print(f"\n[5] Model saved: {model_path.name}")

    # Trial history
    trial_history = []
    for t in study.trials:
        trial_history.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "user_attrs": t.user_attrs,
        })

    results_data = {
        "dataset": str(dataset_path),
        "n_trials": n_trials,
        "best_trial": best.number,
        "best_params": best.params,
        "best_val_pr_auc": best.value,
        "best_val_auc_roc": best.user_attrs["val_auc_roc"],
        "feature_cols": core_cols,
        "n_features": len(core_cols),
        "scale_pos_weight": spw,
        "threshold": threshold,
        "splits": results,
        "trials": trial_history,
    }
    results_path = OUTPUT_DIR / f"optuna_results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results saved: {results_path.name}")

    # ── SHAP ────────────────────────────────────────────────────────────────
    print("\n[6] SHAP analysis:")
    n_shap = min(10_000, len(X_test))
    idx = np.random.default_rng(42).choice(len(X_test), n_shap, replace=False)
    run_shap(best_model, X_test[idx], core_cols, OUTPUT_DIR, f"tuned_{tag}")

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print(f"  Best Val PR-AUC: {best.value:.4f}")
    print(f"  Test AUC-ROC:    {results[2]['auc_roc']:.4f}")
    print(f"  Test PR-AUC:     {results[2]['auc_pr']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuning for XGBoost core model")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    main(args.dataset, args.n_trials)
