"""Tests for representative selection and pair annotation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.collinearity import (
    annotate_pairs,
    cluster_features,
    select_representatives,
)


def _synthetic_corr() -> pd.DataFrame:
    """Build a synthetic correlation matrix with 3 clusters + 2 isolated."""
    feats = [
        # Cluster 1: climate trio
        "chirps_mean", "chirps_anomaly", "era5_precip",
        # Cluster 2: demography pair
        "pop_density", "worldpop_total",
        # Cluster 3: contagion trio
        "defo_rate_500m", "defo_rate_1000m", "defo_rate_2000m",
        # Isolated
        "srtm_elev", "slope_mean",
    ]
    n = len(feats)
    m = np.eye(n)
    # cluster 1 indices 0,1,2
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            if i != j:
                m[i, j] = 0.95
    # cluster 2 indices 3,4
    m[3, 4] = m[4, 3] = 0.97
    # cluster 3 indices 5,6,7
    for i in [5, 6, 7]:
        for j in [5, 6, 7]:
            if i != j:
                m[i, j] = 0.93
    # low cross-correlations elsewhere (already 0)
    return pd.DataFrame(m, index=feats, columns=feats)


def test_select_representatives_keeps_one_per_cluster():
    corr = _synthetic_corr()
    clusters = cluster_features(corr, threshold=0.9)
    # Expect 3 multi-feature clusters + 2 isolated = 5 total clusters.
    multi = [c for c in clusters.values() if len(c) > 1]
    isolated = [c for c in clusters.values() if len(c) == 1]
    assert len(multi) == 3
    assert len(isolated) == 2

    reps = select_representatives(clusters, corr)
    assert len(reps) == len(clusters)
    # Exactly one representative per cluster.
    for members in clusters.values():
        picked = [r for r in reps if r in members]
        assert len(picked) == 1
    # Isolated features retained.
    assert "srtm_elev" in reps
    assert "slope_mean" in reps


def test_annotate_pairs_adds_columns_and_justifications():
    df = pd.DataFrame(
        {
            "feature_a": ["pop_density", "defo_rate_500m", "chirps_anomaly_2020", "srtm_elev"],
            "feature_b": ["ntl_mean", "defo_rate_2000m", "chirps_mean_2020", "slope_mean"],
            "corr": [0.9, 0.95, 0.88, 0.2],
            "abs_corr": [0.9, 0.95, 0.88, 0.2],
        }
    )
    out = annotate_pairs(df)
    assert {"family_a", "family_b", "physical_justification"} <= set(out.columns)
    assert out.loc[0, "physical_justification"] == "human activity proxy redundancy"
    assert out.loc[1, "physical_justification"] == "neighbouring deforestation pressure at multiple radii"
    assert out.loc[2, "physical_justification"] == "climate trend redundancy"
