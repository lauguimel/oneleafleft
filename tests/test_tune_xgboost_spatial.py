"""Tests for spatial-CV Optuna tuning in ``scripts/tune_xgboost.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(PROJECT_DIR / "src"))


def _synthetic_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x_km = rng.uniform(0, 200, n)
    y_km = rng.uniform(0, 200, n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    logits = 0.8 * f1 - 0.5 * f2 + 0.01 * x_km
    p = 1.0 / (1.0 + np.exp(-logits))
    target = (rng.uniform(size=n) < p).astype(int)
    # block_id only used if block_km=None; we still include it.
    block_id = (np.floor(x_km / 50).astype(int) * 10
                + np.floor(y_km / 50).astype(int))
    return pd.DataFrame(
        {
            "x_km": x_km,
            "y_km": y_km,
            "block_id": block_id,
            "f1": f1,
            "f2": f2,
            "target": target,
        }
    )


def test_spatial_cv_objective_runs_one_trial(tmp_path: Path) -> None:
    from tune_xgboost import run_spatial_cv_study

    df = _synthetic_frame()
    out = tmp_path / "optuna_spatial.json"
    results = run_spatial_cv_study(
        train_df=df,
        feature_cols=["f1", "f2"],
        n_trials=1,
        output_path=out,
        guard_km=2.0,
        n_splits=3,
        block_km=50.0,
    )

    assert results["cv_mode"] == "spatial"
    assert results["n_trials"] == 1
    assert "best_params" in results and isinstance(results["best_params"], dict)
    # Sanity check that expected hyperparameter names were sampled.
    for key in ("max_depth", "learning_rate", "subsample"):
        assert key in results["best_params"]
    assert results["best_val_pr_auc"] is not None
    assert 0.0 <= results["best_val_pr_auc"] <= 1.0

    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk["cv_mode"] == "spatial"
    assert on_disk["best_params"] == results["best_params"]
