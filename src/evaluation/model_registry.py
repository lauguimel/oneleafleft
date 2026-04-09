"""Shared helpers to parse and classify saved XGBoost model artifacts.

Model artifacts in this project are produced by three training scripts:

- ``scripts/tune_xgboost.py``       → ``tuned_model_<tag>.json`` + ``optuna_results_<tag>.json``
- ``scripts/train_xgboost.py``      → ``model_<tag>.json``        + ``results_<tag>.json``
- ``scripts/train_core_model.py``   → ``core_model_<tag>.json``   + ``core_results_<tag>.json``

This module centralises filename parsing, sibling-results lookup, hyperparameter
fingerprinting, and tuned-vs-default classification so that both the audit
script and downstream evaluation code can share one implementation.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable


# Filename prefix → (kind, sibling results prefix)
_PREFIX_MAP: dict[str, tuple[str, str]] = {
    "tuned_model_": ("tuned_xgboost", "optuna_results_"),
    "core_model_": ("train_xgboost", "core_results_"),
    "model_": ("train_xgboost", "results_"),
}

# XGBoost sklearn defaults we consider "untouched" hyperparameters.
_DEFAULT_HYPERPARAMS: dict[str, float] = {
    "learning_rate": 0.3,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

_TAG_RE = re.compile(r"_(\d{8})\.json$")


@dataclass
class ModelArtifact:
    """Parsed metadata for a single saved XGBoost model artifact.

    Attributes:
        path: Absolute path to the booster ``.json`` file.
        kind: Classification label (``tuned_xgboost`` or ``train_xgboost``).
        tag: Date tag extracted from the filename (``YYYYMMDD``) or ``""``.
        created: File mtime as an ISO-8601 date string.
        filename_prefix: The matched prefix (e.g. ``tuned_model_``).
        hyperparameters: Best hyperparameters found in sibling results, if any.
        feature_set: List of feature column names used to train the model.
        split_protocol: Short description of the train/val/test split used.
        metrics: Dict keyed by split name with the saved metrics.
        results_path: Path to the sibling results JSON if present.
        fingerprint_source: ``"filename"`` or ``"hyperparams"``.
    """

    path: Path
    kind: str
    tag: str
    created: str
    filename_prefix: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_set: list[str] = field(default_factory=list)
    split_protocol: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    results_path: Path | None = None
    fingerprint_source: str = "filename"

    def to_row(self) -> dict[str, str]:
        """Return a flat dict suitable for a Markdown table row.

        Returns:
            Dict with string values for ``name``, ``kind``, ``created``,
            ``n_features``, ``n_trees``, ``key_hparams``, ``metrics``.
        """
        key_hp_parts = []
        for k in ("learning_rate", "max_depth", "n_estimators", "min_child_weight"):
            if k in self.hyperparameters:
                key_hp_parts.append(f"{k}={self.hyperparameters[k]}")
        key_hp = ", ".join(key_hp_parts) if key_hp_parts else "default"

        met_parts = []
        for split_name, split_metrics in (self.metrics or {}).items():
            if isinstance(split_metrics, dict):
                pr = split_metrics.get("pr_auc") or split_metrics.get("average_precision")
                roc = split_metrics.get("auc_roc") or split_metrics.get("roc_auc")
                if pr is not None or roc is not None:
                    met_parts.append(
                        f"{split_name}: PR={pr:.3f} ROC={roc:.3f}"
                        if pr is not None and roc is not None
                        else f"{split_name}: {pr or roc}"
                    )
        metrics_str = "; ".join(met_parts) if met_parts else "-"

        return {
            "name": self.path.name,
            "kind": self.kind,
            "created": self.created,
            "tag": self.tag,
            "n_features": str(len(self.feature_set)),
            "key_hparams": key_hp,
            "split_protocol": self.split_protocol or "-",
            "metrics": metrics_str,
        }


def parse_filename(path: Path) -> tuple[str | None, str, str]:
    """Extract ``(prefix, kind, tag)`` from a model filename.

    Args:
        path: Path to a candidate ``.json`` model artifact.

    Returns:
        Tuple ``(prefix, kind, tag)``. If no known prefix matches, ``prefix``
        is ``None`` and ``kind`` is ``"unknown"``.
    """
    name = path.name
    # Longest prefix first so ``tuned_model_`` wins over ``model_``.
    for prefix in sorted(_PREFIX_MAP.keys(), key=len, reverse=True):
        if name.startswith(prefix):
            kind, _ = _PREFIX_MAP[prefix]
            tag_match = _TAG_RE.search(name)
            tag = tag_match.group(1) if tag_match else ""
            return prefix, kind, tag
    return None, "unknown", ""


def find_sibling_results(model_path: Path, prefix: str, tag: str) -> Path | None:
    """Locate the sibling results JSON for a given model artifact.

    Args:
        model_path: Path to the model JSON.
        prefix: Matched filename prefix returned by :func:`parse_filename`.
        tag: Date tag returned by :func:`parse_filename`.

    Returns:
        Path to the sibling results file if it exists, else ``None``.
    """
    if prefix not in _PREFIX_MAP or not tag:
        return None
    _, results_prefix = _PREFIX_MAP[prefix]
    candidate = model_path.parent / f"{results_prefix}{tag}.json"
    if candidate.exists():
        return candidate
    return None


def _is_tuned_hyperparams(hp: dict[str, Any]) -> bool:
    """Decide whether a hyperparameter dict looks like Optuna-tuned output.

    Args:
        hp: Hyperparameter mapping extracted from a results JSON.

    Returns:
        ``True`` if at least two values deviate from XGBoost defaults.
    """
    if not hp:
        return False
    deviations = 0
    for key, default in _DEFAULT_HYPERPARAMS.items():
        if key in hp:
            try:
                if abs(float(hp[key]) - float(default)) > 1e-9:
                    deviations += 1
            except (TypeError, ValueError):
                continue
    return deviations >= 2


def _load_results(results_path: Path) -> dict[str, Any]:
    """Safely load a results JSON file.

    Args:
        results_path: Path to the results file.

    Returns:
        Parsed JSON dict, or empty dict on failure.
    """
    try:
        with results_path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def build_artifact(path: Path) -> ModelArtifact | None:
    """Build a :class:`ModelArtifact` from a model JSON path.

    Args:
        path: Path to a candidate model JSON file.

    Returns:
        A populated :class:`ModelArtifact`, or ``None`` if the filename does
        not match any known training-script prefix.
    """
    prefix, kind, tag = parse_filename(path)
    if prefix is None:
        return None

    created = _dt.datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
    results_path = find_sibling_results(path, prefix, tag)

    hyperparameters: dict[str, Any] = {}
    feature_set: list[str] = []
    split_protocol = ""
    metrics: dict[str, Any] = {}

    if results_path is not None:
        data = _load_results(results_path)
        if "best_params" in data and isinstance(data["best_params"], dict):
            hyperparameters = dict(data["best_params"])
        elif "params" in data and isinstance(data["params"], dict):
            hyperparameters = dict(data["params"])
        elif "hyperparameters" in data and isinstance(data["hyperparameters"], dict):
            hyperparameters = dict(data["hyperparameters"])

        fc = data.get("feature_cols")
        if isinstance(fc, list):
            feature_set = [str(x) for x in fc]

        split_protocol = str(
            data.get("split_protocol")
            or data.get("split")
            or data.get("dataset", "")
        )

        splits = data.get("splits")
        if isinstance(splits, list):
            for entry in splits:
                if isinstance(entry, dict):
                    name = entry.get("split") or entry.get("name") or "split"
                    metrics[str(name)] = entry
        elif isinstance(splits, dict):
            metrics = splits

    fingerprint_source = "filename"
    # Reclassify by fingerprint when filename is ambiguous (``model_`` default
    # that actually carries tuned hyperparameters).
    if prefix == "model_" and _is_tuned_hyperparams(hyperparameters):
        kind = "tuned_xgboost"
        fingerprint_source = "hyperparams"

    return ModelArtifact(
        path=path,
        kind=kind,
        tag=tag,
        created=created,
        filename_prefix=prefix,
        hyperparameters=hyperparameters,
        feature_set=feature_set,
        split_protocol=split_protocol,
        metrics=metrics,
        results_path=results_path,
        fingerprint_source=fingerprint_source,
    )


def discover_artifacts(search_dirs: Iterable[Path]) -> list[ModelArtifact]:
    """Scan directories for model artifacts and classify each.

    Args:
        search_dirs: Directories to scan (non-recursive) for ``*.json``
            files matching a known model prefix.

    Returns:
        List of :class:`ModelArtifact` sorted by creation date descending.
    """
    found: dict[Path, ModelArtifact] = {}
    for d in search_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            prefix, _, _ = parse_filename(p)
            if prefix is None:
                continue
            artifact = build_artifact(p)
            if artifact is not None:
                found[p.resolve()] = artifact
    return sorted(found.values(), key=lambda a: (a.created, a.path.name), reverse=True)


def render_markdown_table(artifacts: list[ModelArtifact]) -> str:
    """Render a Markdown table listing all discovered model artifacts.

    Args:
        artifacts: List of artifacts to include.

    Returns:
        A Markdown string with a header line and one row per artifact.
    """
    header = (
        "| name | kind | created | tag | n_features | key_hparams | split_protocol | metrics |\n"
        "|------|------|---------|-----|------------|-------------|----------------|---------|\n"
    )
    if not artifacts:
        return header + "| _(none found)_ |  |  |  |  |  |  |  |\n"

    rows = []
    for a in artifacts:
        row = a.to_row()
        rows.append(
            "| {name} | {kind} | {created} | {tag} | {n_features} | {key_hparams} | {split_protocol} | {metrics} |".format(
                **row
            )
        )
    return header + "\n".join(rows) + "\n"


def artifact_to_dict(artifact: ModelArtifact) -> dict[str, Any]:
    """Serialize an artifact to a JSON-friendly dict.

    Args:
        artifact: The artifact to serialize.

    Returns:
        Dict representation with paths coerced to strings.
    """
    d = asdict(artifact)
    d["path"] = str(artifact.path)
    d["results_path"] = str(artifact.results_path) if artifact.results_path else None
    return d
