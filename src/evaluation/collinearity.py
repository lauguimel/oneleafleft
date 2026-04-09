"""Collinearity audit utilities.

Pure functions to compute correlations, VIF, hierarchical clusters, and
extract top correlated pairs from a numeric feature matrix.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Family classification: ordered (first match wins).
_FAMILY_PATTERNS: List[tuple[str, re.Pattern]] = [
    ("demography", re.compile(r"(pop|worldpop|density|urban|settlement|ghsl)", re.I)),
    ("climate", re.compile(r"(chirps|era5|precip|temp|tmin|tmax|rain|evap|wind|humid|spi|pdsi)", re.I)),
    ("topography", re.compile(r"(srtm|elev|slope|aspect|dem|altitude|terrain|ruggedness)", re.I)),
    ("contagion", re.compile(r"(buffer|neighbor|contagion|dist_|distance|nearest|adjacent|spatial_lag)", re.I)),
    ("temporal_profile", re.compile(r"(delta|d1yr|d3yr|d10yr|anomaly|volatility|trend|_lag\d|_diff\d)", re.I)),
]


def classify_feature_family(name: str) -> str:
    """Classify a feature name into a family by regex on its name.

    Args:
        name: Feature column name.

    Returns:
        One of: 'demography', 'climate', 'topography', 'contagion',
        'temporal_profile', 'other'.
    """
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(name):
            return family
    return "other"


def compute_correlations(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Compute a correlation matrix on numeric columns.

    Args:
        df: Input dataframe.
        method: 'pearson' or 'spearman'.

    Returns:
        Square correlation DataFrame (numeric columns only).
    """
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method=method)


def compute_vif(df: pd.DataFrame) -> pd.Series:
    """Compute Variance Inflation Factor for each numeric column.

    Args:
        df: Input dataframe with numeric features.

    Returns:
        Series indexed by feature name with VIF values. NaN VIFs are
        replaced with np.inf (fully collinear).
    """
    numeric = df.select_dtypes(include=[np.number]).dropna()
    # Drop zero-variance columns
    numeric = numeric.loc[:, numeric.std(ddof=0) > 0]
    X = numeric.values.astype(float)
    vifs = {}
    for i, col in enumerate(numeric.columns):
        try:
            v = variance_inflation_factor(X, i)
        except Exception:
            v = np.inf
        vifs[col] = v if np.isfinite(v) else np.inf
    return pd.Series(vifs, name="VIF")


def cluster_features(corr: pd.DataFrame, threshold: float) -> Dict[int, List[str]]:
    """Hierarchical clustering on distance = 1 - |corr|.

    Args:
        corr: Square correlation matrix.
        threshold: Absolute correlation threshold; features with |rho| above
            this are grouped together (distance cut at 1 - threshold).

    Returns:
        Dict mapping cluster id -> list of feature names in that cluster.
    """
    features = list(corr.columns)
    if len(features) < 2:
        return {1: features} if features else {}
    dist = 1.0 - corr.abs().values
    np.fill_diagonal(dist, 0.0)
    # Ensure symmetry and non-negativity
    dist = np.clip((dist + dist.T) / 2.0, 0.0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    clusters: Dict[int, List[str]] = {}
    for feat, lab in zip(features, labels):
        clusters.setdefault(int(lab), []).append(feat)
    return clusters


def extract_top_pairs(corr: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Extract the top-n most correlated pairs (by |rho|).

    Args:
        corr: Square correlation matrix.
        n: Number of pairs to return.

    Returns:
        DataFrame with columns ['feature_a', 'feature_b', 'corr', 'abs_corr'],
        sorted by abs_corr descending.
    """
    c = corr.copy()
    cols = c.columns.tolist()
    mat = c.values
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            rows.append((cols[i], cols[j], float(v), abs(float(v))))
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "corr", "abs_corr"])
    out = out.sort_values("abs_corr", ascending=False).head(n).reset_index(drop=True)
    return out


DEFAULT_FAMILY_PRIORITY: List[str] = [
    "contagion",
    "demography",
    "climate",
    "topography",
    "temporal_profile",
    "other",
]


def select_representatives(
    clusters: Dict[int, List[str]],
    corr: pd.DataFrame,
    family_priority: List[str] | None = None,
) -> List[str]:
    """Select one representative feature per cluster.

    For each multi-feature cluster, keeps the feature with the lowest mean
    absolute correlation against features OUTSIDE its cluster (most
    "self-contained"). Ties are broken by family priority then alphabetically.
    Isolated features (cluster size 1) are always retained.

    Args:
        clusters: Mapping cluster_id -> list of feature names.
        corr: Square correlation DataFrame covering all features in clusters.
        family_priority: Ordered list of family names; earlier = preferred.
            Defaults to DEFAULT_FAMILY_PRIORITY.

    Returns:
        List of selected feature names (one per cluster), sorted alphabetically.
    """
    priority = family_priority or DEFAULT_FAMILY_PRIORITY
    prio_rank = {fam: i for i, fam in enumerate(priority)}
    abs_corr = corr.abs()
    selected: List[str] = []
    for _, members in clusters.items():
        if len(members) == 1:
            selected.append(members[0])
            continue
        members_set = set(members)
        outside = [c for c in abs_corr.columns if c not in members_set]
        scored = []
        for feat in members:
            if outside:
                mean_out = float(abs_corr.loc[feat, outside].mean())
            else:
                mean_out = 0.0
            fam = classify_feature_family(feat)
            rank = prio_rank.get(fam, len(priority))
            scored.append((mean_out, rank, feat))
        scored.sort()
        selected.append(scored[0][2])
    return sorted(selected)


_JUSTIFICATION_RULES: List[tuple[re.Pattern, re.Pattern, str]] = [
    (
        re.compile(r"^pop", re.I),
        re.compile(r"ntl", re.I),
        "human activity proxy redundancy",
    ),
    (
        re.compile(r"ntl", re.I),
        re.compile(r"^pop", re.I),
        "human activity proxy redundancy",
    ),
    (
        re.compile(r"defo_rate_.*m", re.I),
        re.compile(r"defo_rate_.*m", re.I),
        "neighbouring deforestation pressure at multiple radii",
    ),
    (
        re.compile(r"chirps_anomaly", re.I),
        re.compile(r"chirps", re.I),
        "climate trend redundancy",
    ),
    (
        re.compile(r"chirps", re.I),
        re.compile(r"chirps_anomaly", re.I),
        "climate trend redundancy",
    ),
]


def _physical_justification(a: str, b: str, fam_a: str, fam_b: str) -> str:
    """Return a short physical justification string for a redundant pair."""
    for pat_a, pat_b, text in _JUSTIFICATION_RULES:
        if pat_a.search(a) and pat_b.search(b):
            return text
    if fam_a == fam_b and fam_a != "other":
        return "same-family redundancy"
    return ""


def annotate_pairs(top_pairs: pd.DataFrame) -> pd.DataFrame:
    """Annotate a top-correlated-pairs DataFrame with families + justifications.

    Args:
        top_pairs: DataFrame with columns ['feature_a', 'feature_b', ...].

    Returns:
        Copy of ``top_pairs`` with added columns ``family_a``, ``family_b``,
        ``physical_justification``.
    """
    out = top_pairs.copy()
    out["family_a"] = out["feature_a"].map(classify_feature_family)
    out["family_b"] = out["feature_b"].map(classify_feature_family)
    out["physical_justification"] = [
        _physical_justification(a, b, fa, fb)
        for a, b, fa, fb in zip(
            out["feature_a"], out["feature_b"], out["family_a"], out["family_b"]
        )
    ]
    return out
