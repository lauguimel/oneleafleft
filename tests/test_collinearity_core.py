"""Tests for src.evaluation.collinearity core pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.collinearity import (  # noqa: E402
    classify_feature_family,
    cluster_features,
    compute_correlations,
    compute_vif,
    extract_top_pairs,
)


def _make_synthetic(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.05, size=n)  # ~0.99 correlated with a
    c = rng.normal(size=n)
    d = rng.normal(size=n)
    e = rng.normal(size=n)
    f = rng.normal(size=n)
    return pd.DataFrame({"f_a": a, "f_b": b, "f_c": c, "f_d": d, "f_e": e, "f_f": f})


def test_correlation_vif_cluster_pipeline() -> None:
    df = _make_synthetic()

    corr = compute_correlations(df, method="pearson")
    assert corr.shape == (6, 6)
    # Sanity: the collinear pair sits near ~0.99.
    assert corr.loc["f_a", "f_b"] > 0.97

    # (a) Top pairs contains (f_a, f_b).
    top = extract_top_pairs(corr, n=5)
    pair_set = {frozenset((r.feature_a, r.feature_b)) for r in top.itertuples()}
    assert frozenset(("f_a", "f_b")) in pair_set

    # (b) VIF of a redundant feature > 10.
    vif = compute_vif(df)
    assert vif.loc["f_b"] > 10
    assert vif.loc["f_a"] > 10

    # (c) Cluster extractor at threshold 0.9 groups them together.
    clusters = cluster_features(corr, threshold=0.9)
    found = False
    for members in clusters.values():
        if "f_a" in members and "f_b" in members:
            found = True
            break
    assert found, f"f_a and f_b not clustered together: {clusters}"


def test_classify_feature_family() -> None:
    assert classify_feature_family("worldpop_2020") == "demography"
    assert classify_feature_family("chirps_precip_mean") == "climate"
    assert classify_feature_family("srtm_elevation") == "topography"
    assert classify_feature_family("dist_to_road") == "contagion"
    assert classify_feature_family("ndvi_d3yr_anomaly") == "temporal_profile"
    assert classify_feature_family("random_col") == "other"
