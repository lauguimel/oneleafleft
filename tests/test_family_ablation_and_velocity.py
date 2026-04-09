"""Tests for family ablation and contagion velocity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.family_ablation import (
    compute_contagion_velocity,
    family_ablation,
)


def _fake_family_classifier(name: str) -> str:
    if name.startswith("famA_"):
        return "famA"
    if name.startswith("famB_"):
        return "famB"
    return "other"


def test_family_ablation_and_velocity_on_synthetic(tmp_path):
    rng = np.random.default_rng(42)
    n = 200
    famA = rng.normal(size=(n, 3))
    famB = rng.normal(size=(n, 3))
    # Target driven mostly by famA
    logits = 1.5 * famA[:, 0] - 0.8 * famA[:, 1] + 0.2 * famB[:, 0]
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(int)

    df = pd.DataFrame(
        np.hstack([famA, famB]),
        columns=[f"famA_{i}" for i in range(3)] + [f"famB_{i}" for i in range(3)],
    )
    df["target"] = y
    df_train = df.iloc[:150].reset_index(drop=True)
    df_test = df.iloc[150:].reset_index(drop=True)

    params = {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "verbosity": 0,
    }

    result = family_ablation(
        df_train, df_test, "target",
        _fake_family_classifier, ["famA", "famB"], params,
    )

    assert len(result) == 2
    assert set(result.columns) == {
        "family", "n_features_dropped", "pr_auc",
        "pr_auc_lo", "pr_auc_hi", "delta_pr_auc",
    }
    assert np.isfinite(result["pr_auc"]).all()
    assert np.isfinite(result["delta_pr_auc"]).all()

    # --- contagion velocity ---
    n_pts = 50
    rng2 = np.random.default_rng(0)
    pts = pd.DataFrame({
        "year": np.concatenate([np.full(n_pts, 2020), np.full(n_pts, 2021)]),
        "x": rng2.uniform(0, 10000, 2 * n_pts),
        "y": rng2.uniform(0, 10000, 2 * n_pts),
        "target": np.ones(2 * n_pts, dtype=int),
    })
    vel = compute_contagion_velocity(pts, "year", "x", "y", "target")
    assert vel["n"] > 0
    for k in ("p50", "p75", "p90", "p95", "p99", "mean"):
        assert np.isfinite(vel[k])
