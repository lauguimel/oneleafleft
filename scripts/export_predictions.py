"""Export val predictions + SHAP values for the Streamlit app.

Creates:
    data/app/predictions_val.parquet — lon/lat/target/proba + top SHAP drivers
    data/app/shap_values.parquet     — full SHAP matrix (10K sample)
    data/app/model_info.json         — model metadata

Usage:
    conda activate deforest
    python scripts/export_predictions.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from train_xgboost import load_dataset, optimal_threshold

DATA_DIR = PROJECT_DIR / "data"
APP_DIR = DATA_DIR / "app"
APP_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_test" / "features_traintest_20260307.parquet"
VAL_PATH = DATA_DIR / "val" / "features_val_20260307.parquet"
MODEL_PATH = DATA_DIR / "model_20260307.json"


def main():
    print("=" * 60)
    print("EXPORT PREDICTIONS FOR STREAMLIT")
    print("=" * 60)

    # Load model
    print("\n[1] Loading model...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    print(f"  Model: {MODEL_PATH.name} ({model.n_estimators} estimators)")

    # Load datasets
    print("\n[2] Loading datasets...")
    train_df, val_df, _, feature_cols = load_dataset(TRAIN_PATH, val_path=VAL_PATH)
    print(f"  Train: {len(train_df):,} rows, Val: {len(val_df):,} rows")
    print(f"  Features: {len(feature_cols)}")

    # Predict on val
    print("\n[3] Predicting on val...")
    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values.astype(int)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Optimal threshold
    X_tr = train_df[feature_cols].values
    y_tr = train_df["target"].values.astype(int)
    threshold = optimal_threshold(model, X_val, y_val)
    print(f"  Threshold: {threshold:.4f}")

    # SHAP values (sample for speed)
    print("\n[4] Computing SHAP values (10K sample)...")
    import shap
    n_shap = min(10_000, len(X_val))
    rng = np.random.default_rng(42)
    shap_idx = rng.choice(len(X_val), n_shap, replace=False)

    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val[shap_idx])
    elapsed = time.time() - t0
    print(f"  SHAP computed in {elapsed:.1f}s")

    # Top 5 SHAP drivers per point (for all val points)
    print("\n[5] Computing top SHAP drivers (all val)...")
    t0 = time.time()
    shap_all = explainer.shap_values(X_val)
    elapsed = time.time() - t0
    print(f"  Full SHAP in {elapsed:.1f}s")

    top_k = 5
    top_features = []
    top_shap_vals = []
    for i in range(len(X_val)):
        abs_shap = np.abs(shap_all[i])
        top_idx = np.argsort(abs_shap)[-top_k:][::-1]
        top_features.append([feature_cols[j] for j in top_idx])
        top_shap_vals.append([float(shap_all[i][j]) for j in top_idx])

    # Build predictions DataFrame
    print("\n[6] Building predictions DataFrame...")
    pred_df = pd.DataFrame({
        "lon": val_df["lon"].values,
        "lat": val_df["lat"].values,
        "target": y_val,
        "proba": y_proba,
        "predicted": (y_proba >= threshold).astype(int),
    })
    # Add top SHAP drivers
    for k in range(top_k):
        pred_df[f"shap_feat_{k+1}"] = [tf[k] for tf in top_features]
        pred_df[f"shap_val_{k+1}"] = [tv[k] for tv in top_shap_vals]

    pred_path = APP_DIR / "predictions_val.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"  Saved: {pred_path} ({len(pred_df):,} rows)")

    # SHAP sample DataFrame
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df["_idx"] = shap_idx
    shap_path = APP_DIR / "shap_values_sample.parquet"
    shap_df.to_parquet(shap_path, index=False)
    print(f"  Saved: {shap_path} ({len(shap_df):,} rows)")

    # Feature importance (mean |SHAP|)
    mean_shap = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_all).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    imp_path = APP_DIR / "feature_importance.csv"
    mean_shap.to_csv(imp_path, index=False)
    print(f"  Saved: {imp_path}")

    # Model info
    from sklearn.metrics import roc_auc_score, average_precision_score
    info = {
        "model_path": str(MODEL_PATH),
        "n_estimators": int(model.n_estimators) if model.n_estimators else None,
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") and model.best_iteration is not None else None,
        "n_features": len(feature_cols),
        "threshold": float(threshold),
        "val_n": len(y_val),
        "val_pos": int(y_val.sum()),
        "val_auc_roc": float(roc_auc_score(y_val, y_proba)),
        "val_pr_auc": float(average_precision_score(y_val, y_proba)),
        "feature_cols": feature_cols,
    }
    info_path = APP_DIR / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved: {info_path}")

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print(f"  Val AUC-ROC: {info['val_auc_roc']:.4f}")
    print(f"  Val PR-AUC: {info['val_pr_auc']:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
