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

sys.path.insert(0, str(PROJECT_DIR / "src"))
from evaluation.spatial_cv import SpatialBlockKFold, project_km  # noqa: E402


def _ensure_xy_km(df):
    """Add x_km/y_km columns from lon/lat if missing (in-place safe)."""
    if "x_km" in df.columns and "y_km" in df.columns:
        return df
    if "lon" not in df.columns or "lat" not in df.columns:
        raise KeyError("Need either (x_km,y_km) or (lon,lat) on the dataframe.")
    x_km, y_km = project_km(df["lon"].to_numpy(), df["lat"].to_numpy())
    df = df.copy()
    df["x_km"] = x_km
    df["y_km"] = y_km
    return df

OUTPUT_DIR = PROJECT_DIR / "data"
CORE_GROUPS = ["hansen", "spatial", "infra"]


def _suggest_params(trial: "optuna.Trial") -> dict:
    """Suggest an XGBoost hyperparameter set for one Optuna trial.

    Args:
        trial: Active Optuna trial.

    Returns:
        Dict of hyperparameters compatible with ``xgb.XGBClassifier``.
    """
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    }


def make_spatial_cv_objective(
    train_df,
    feature_cols: list[str],
    target_col: str = "target",
    n_splits: int = 5,
    guard_km: float = 5.0,
    x_col: str = "x_km",
    y_col: str = "y_km",
    group_col: str = "block_id",
    block_km: float | None = 50.0,
    n_estimators: int = 2000,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
):
    """Build an Optuna objective using spatial block K-fold CV.

    Averages validation PR-AUC across ``n_splits`` spatial folds with an
    asymmetric ``guard_km`` buffer.

    Args:
        train_df: Training dataframe containing features, target, coordinates
            and block id.
        feature_cols: Names of feature columns to use.
        target_col: Name of the binary target column.
        n_splits: Number of spatial folds.
        guard_km: Guard-band width in km.
        x_col: Projected x coordinate column (km).
        y_col: Projected y coordinate column (km).
        group_col: Block id column name (used when ``block_km`` is None).
        block_km: Coarse block size. If set, overrides ``group_col``.
        n_estimators: Max XGBoost boosting rounds per fold.
        early_stopping_rounds: Early stopping patience.
        random_state: Seed.

    Returns:
        A function ``objective(trial)`` suitable for ``study.optimize``.
    """
    import pandas as pd  # local import to keep module import cheap

    splitter = SpatialBlockKFold(
        n_splits=n_splits,
        guard_km=guard_km,
        x_col=x_col,
        y_col=y_col,
        group_col=group_col,
        block_km=block_km,
        random_state=random_state,
    )
    folds = list(splitter.split(train_df))
    X_all = train_df[feature_cols].to_numpy()
    y_all = train_df[target_col].to_numpy().astype(int)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        fold_scores = []
        best_iters = []
        for tr_idx, val_idx in folds:
            if len(tr_idx) == 0 or len(val_idx) == 0:
                continue
            y_tr = y_all[tr_idx]
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                continue
            spw = compute_scale_pos_weight(y_tr)
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                scale_pos_weight=spw,
                tree_method="hist",
                eval_metric="aucpr",
                early_stopping_rounds=early_stopping_rounds,
                random_state=random_state,
                n_jobs=-1,
                **params,
            )
            model.fit(
                X_all[tr_idx], y_tr,
                eval_set=[(X_all[val_idx], y_all[val_idx])],
                verbose=False,
            )
            proba = model.predict_proba(X_all[val_idx])[:, 1]
            fold_scores.append(float(average_precision_score(y_all[val_idx], proba)))
            best_iters.append(int(getattr(model, "best_iteration", 0) or 0))

        if not fold_scores:
            return 0.0
        mean_pr = float(np.mean(fold_scores))
        trial.set_user_attr("fold_pr_aucs", fold_scores)
        trial.set_user_attr("mean_best_iteration", float(np.mean(best_iters)) if best_iters else 0.0)
        trial.set_user_attr("val_pr_auc", mean_pr)
        return mean_pr

    return objective


def run_spatial_cv_study(
    train_df,
    feature_cols: list[str],
    n_trials: int = 1,
    output_path: Path | None = None,
    guard_km: float = 5.0,
    n_splits: int = 5,
    block_km: float | None = 50.0,
    target_col: str = "target",
    study_name: str = "xgb_spatial_cv",
) -> dict:
    """Run an Optuna study using spatial-CV objective and persist results.

    Args:
        train_df: Training dataframe with features, target, coords, block id.
        feature_cols: Feature column names.
        n_trials: Number of Optuna trials.
        output_path: Optional path for the JSON results file.
        guard_km: Guard band width in km.
        n_splits: Number of spatial folds.
        block_km: Coarse block size (None to use ``block_id`` column).
        target_col: Target column name.
        study_name: Optuna study name.

    Returns:
        Dict with ``cv_mode``, ``best_params``, ``best_val_pr_auc``, and
        ``n_trials``; also written to ``output_path`` if provided.
    """
    objective = make_spatial_cv_objective(
        train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        n_splits=n_splits,
        guard_km=guard_km,
        block_km=block_km,
    )
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    results = {
        "cv_mode": "spatial",
        "n_trials": n_trials,
        "n_splits": n_splits,
        "guard_km": guard_km,
        "block_km": block_km,
        "best_trial": best.number,
        "best_params": best.params,
        "best_val_pr_auc": float(best.value) if best.value is not None else None,
        "fold_pr_aucs": best.user_attrs.get("fold_pr_aucs", []),
        "feature_cols": list(feature_cols),
        "n_features": len(feature_cols),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


def main(
    dataset_path: Path | None,
    n_trials: int,
    cv: str = "random",
    guard_km: float = 5.0,
    n_splits: int = 5,
) -> None:
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
    print(f"\n[2] Running {n_trials} Optuna trials (cv={cv})...")
    if cv == "spatial":
        train_df = _ensure_xy_km(train_df)
        spatial_obj = make_spatial_cv_objective(
            train_df,
            feature_cols=core_cols,
            n_splits=n_splits,
            guard_km=guard_km,
        )
        study = optuna.create_study(direction="maximize", study_name="xgb_spatial_cv")
        study.optimize(spatial_obj, n_trials=n_trials, show_progress_bar=True)
    else:
        study = optuna.create_study(direction="maximize", study_name="xgb_core_prauc")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n  Best trial #{best.number}:")
    print(f"    Val PR-AUC: {best.value:.4f}")
    print(f"    Val AUC-ROC: {best.user_attrs.get('val_auc_roc', float('nan')):.4f}")
    print(f"    Best iteration: {best.user_attrs.get('best_iteration', 'n/a')}")
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
        "cv_mode": cv,
        "guard_km": guard_km if cv == "spatial" else None,
        "n_splits": n_splits if cv == "spatial" else None,
        "dataset": str(dataset_path),
        "n_trials": n_trials,
        "best_trial": best.number,
        "best_params": best.params,
        "best_val_pr_auc": best.value,
        "best_val_auc_roc": best.user_attrs.get("val_auc_roc"),
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
    parser.add_argument("--cv", choices=["random", "spatial"], default="random",
                        help="CV mode: 'random' (legacy single train/val) or 'spatial' block K-fold")
    parser.add_argument("--guard-km", type=float, default=5.0)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    main(args.dataset, args.n_trials, cv=args.cv, guard_km=args.guard_km, n_splits=args.n_splits)
