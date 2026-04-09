"""Family ablation and contagion velocity utilities.

Functions to (1) train XGBoost with groups of features dropped and measure
PR-AUC impact with bootstrap CIs, and (2) compute contagion velocity as the
distribution of year-over-year nearest-neighbour distances between positive
points.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier


def train_and_eval(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    params: dict,
) -> dict:
    """Train XGBoost and return PR-AUC / ROC-AUC on the test set.

    Args:
        X_train: Training features.
        y_train: Training binary labels.
        X_test: Test features.
        y_test: Test binary labels.
        params: XGBClassifier hyperparameters.

    Returns:
        Dict with keys ``pr_auc``, ``roc_auc``, ``y_score``.
    """
    clf = XGBClassifier(**params)
    clf.fit(X_train.values, np.asarray(y_train))
    y_score = clf.predict_proba(X_test.values)[:, 1]
    return {
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "y_score": y_score,
    }


def bootstrap_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% CI for PR-AUC.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        Tuple ``(lo, hi)`` for the 2.5th and 97.5th percentiles.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(average_precision_score(yt, ys))
    if not scores:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return (float(lo), float(hi))


def family_ablation(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    family_classifier: Callable[[str], str],
    families_to_drop: list[str],
    params: dict,
) -> pd.DataFrame:
    """Run a family ablation: drop each family in turn and train XGBoost.

    Args:
        df_train: Training dataframe with features + target.
        df_test: Test dataframe with features + target.
        target: Name of the target column.
        family_classifier: Callable mapping feature name -> family string.
        families_to_drop: List of families to ablate one at a time.
        params: XGBClassifier hyperparameters.

    Returns:
        DataFrame with columns ``[family, n_features_dropped, pr_auc,
        pr_auc_lo, pr_auc_hi, delta_pr_auc]``. Delta is relative to the full
        baseline trained on all features.
    """
    # Keep numeric features only — XGBoost cannot consume strings.
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c != target]
    X_tr_full = df_train[feat_cols]
    X_te_full = df_test[feat_cols]
    y_tr = df_train[target].values
    y_te = df_test[target].values

    base = train_and_eval(X_tr_full, y_tr, X_te_full, y_te, params)
    base_pr = base["pr_auc"]

    rows = []
    for fam in families_to_drop:
        dropped = [c for c in feat_cols if family_classifier(c) == fam]
        kept = [c for c in feat_cols if c not in dropped]
        if not kept:
            rows.append((fam, len(dropped), float("nan"), float("nan"),
                         float("nan"), float("nan")))
            continue
        res = train_and_eval(df_train[kept], y_tr, df_test[kept], y_te, params)
        lo, hi = bootstrap_pr_auc(y_te, res["y_score"])
        rows.append((fam, len(dropped), res["pr_auc"], lo, hi,
                     res["pr_auc"] - base_pr))

    return pd.DataFrame(
        rows,
        columns=["family", "n_features_dropped", "pr_auc", "pr_auc_lo",
                 "pr_auc_hi", "delta_pr_auc"],
    )


def compute_contagion_velocity(
    points: pd.DataFrame,
    year_col: str,
    x_col: str,
    y_col: str,
    target_col: str = "target",
) -> dict:
    """Compute contagion velocity from year-over-year nearest-neighbour distances.

    For each positive point (target=1) in year ``t``, find the nearest
    positive point in year ``t-1``. Coordinates must be projected in metres.

    Args:
        points: DataFrame with year, x, y, and target columns.
        year_col: Name of the year column.
        x_col: Name of the projected x coordinate column (metres).
        y_col: Name of the projected y coordinate column (metres).
        target_col: Name of the binary target column. Defaults to ``target``.

    Returns:
        Dict with keys ``p50, p75, p90, p95, p99, mean, n`` (distances in
        metres).
    """
    pos = points[points[target_col] == 1]
    years = sorted(pos[year_col].unique())
    dists: list[float] = []
    for t in years:
        prev = pos[pos[year_col] == t - 1]
        curr = pos[pos[year_col] == t]
        if len(prev) == 0 or len(curr) == 0:
            continue
        tree = cKDTree(prev[[x_col, y_col]].values)
        d, _ = tree.query(curr[[x_col, y_col]].values, k=1)
        dists.extend(d.tolist())

    if not dists:
        return {"p50": float("nan"), "p75": float("nan"), "p90": float("nan"),
                "p95": float("nan"), "p99": float("nan"),
                "mean": float("nan"), "n": 0}

    arr = np.asarray(dists)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "n": int(len(arr)),
    }
