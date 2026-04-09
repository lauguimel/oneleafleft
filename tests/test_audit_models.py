"""Tests for the model inventory audit script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_models import run_audit  # noqa: E402
from src.evaluation.model_registry import (  # noqa: E402
    build_artifact,
    parse_filename,
)


def _write_booster_stub(path: Path) -> None:
    """Write a minimal JSON stub emulating an XGBoost ``save_model`` file."""
    path.write_text(
        json.dumps(
            {
                "version": [2, 0, 0],
                "learner": {
                    "attributes": {},
                    "feature_names": ["f1", "f2"],
                    "learner_model_param": {"num_feature": "2"},
                },
            }
        )
    )


def test_audit_models_classifies_fixture_artifacts(tmp_path: Path) -> None:
    """Two synthetic artifacts are correctly classified and rendered."""
    models_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    models_dir.mkdir()
    data_dir.mkdir()

    # ── Tuned model fixture ──────────────────────────────────────────────
    tuned_model = data_dir / "tuned_model_20260101.json"
    _write_booster_stub(tuned_model)
    tuned_results = data_dir / "optuna_results_20260101.json"
    tuned_results.write_text(
        json.dumps(
            {
                "dataset": "fixtures/train_val_test.parquet",
                "n_trials": 50,
                "best_trial": 17,
                "best_params": {
                    "learning_rate": 0.042,
                    "max_depth": 9,
                    "min_child_weight": 4,
                    "subsample": 0.77,
                    "colsample_bytree": 0.63,
                    "reg_alpha": 0.5,
                    "reg_lambda": 2.3,
                    "gamma": 0.1,
                },
                "best_val_pr_auc": 0.059,
                "feature_cols": ["f1", "f2", "f3"],
                "split_protocol": "temporal_train<=2020_val2021_test2022",
                "splits": [
                    {"split": "train", "pr_auc": 0.12, "auc_roc": 0.88},
                    {"split": "val", "pr_auc": 0.059, "auc_roc": 0.81},
                    {"split": "test", "pr_auc": 0.055, "auc_roc": 0.80},
                ],
            }
        )
    )

    # ── Default training fixture ─────────────────────────────────────────
    default_model = models_dir / "model_20260202.json"
    _write_booster_stub(default_model)
    default_results = models_dir / "results_20260202.json"
    default_results.write_text(
        json.dumps(
            {
                "dataset": "fixtures/train_val_test.parquet",
                "feature_cols": ["f1", "f2"],
                "n_features": 2,
                "scale_pos_weight": 3.2,
                "splits": [
                    {"split": "train", "pr_auc": 0.09, "auc_roc": 0.85},
                    {"split": "val", "pr_auc": 0.041, "auc_roc": 0.79},
                ],
            }
        )
    )

    output = tmp_path / "results" / "models_inventory.md"
    artifacts = run_audit([models_dir, data_dir], output)

    # Both should be discovered.
    assert len(artifacts) == 2
    by_name = {a.path.name: a for a in artifacts}
    assert "tuned_model_20260101.json" in by_name
    assert "model_20260202.json" in by_name

    # Classification assertions.
    assert by_name["tuned_model_20260101.json"].kind == "tuned_xgboost"
    assert by_name["model_20260202.json"].kind == "train_xgboost"

    # Metadata extracted from sibling results.
    tuned = by_name["tuned_model_20260101.json"]
    assert tuned.tag == "20260101"
    assert tuned.hyperparameters.get("max_depth") == 9
    assert tuned.feature_set == ["f1", "f2", "f3"]
    assert "temporal" in tuned.split_protocol

    default = by_name["model_20260202.json"]
    assert default.feature_set == ["f1", "f2"]
    assert default.hyperparameters == {}  # no best_params in default results

    # Markdown output contains both rows.
    md = output.read_text()
    assert "# Model Inventory" in md
    assert "tuned_model_20260101.json" in md
    assert "model_20260202.json" in md
    assert "tuned_xgboost" in md
    assert "train_xgboost" in md


def test_parse_filename_prefix_priority(tmp_path: Path) -> None:
    """``tuned_model_`` must not be mistakenly matched as ``model_``."""
    p = tmp_path / "tuned_model_20260101.json"
    p.write_text("{}")
    prefix, kind, tag = parse_filename(p)
    assert prefix == "tuned_model_"
    assert kind == "tuned_xgboost"
    assert tag == "20260101"


def test_build_artifact_unknown_prefix_returns_none(tmp_path: Path) -> None:
    """Files that don't match any known prefix are skipped."""
    p = tmp_path / "ablation_results_20260101.json"
    p.write_text("{}")
    assert build_artifact(p) is None
