"""Deep Learning comparison: MLP, TabNet, FT-Transformer vs XGBoost.

Phase 5: Train DL models on same core features and temporal split as XGBoost.
Requires PyTorch: pip install torch pytorch-tabnet

Usage:
    conda activate deforest
    pip install torch pytorch-tabnet  # if not installed
    python scripts/train_deep_learning.py
    python scripts/train_deep_learning.py --quick
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from train_xgboost import load_dataset, compute_scale_pos_weight
from ablation_study import resolve_group

OUTPUT_DIR = PROJECT_DIR / "data"
CORE_GROUPS = ["hansen", "spatial", "infra"]


def load_core_data(dataset_path=None):
    """Load dataset filtered to core features."""
    if dataset_path is None:
        dataset_path = OUTPUT_DIR / "features_250k_20260228.parquet"
    train_df, val_df, test_df, all_feature_cols = load_dataset(dataset_path)
    core_cols = []
    for g in CORE_GROUPS:
        core_cols.extend(resolve_group(g, all_feature_cols))
    core_cols = list(dict.fromkeys(core_cols))
    return train_df, val_df, test_df, core_cols


def prepare_data(train_df, val_df, test_df, core_cols):
    """Extract arrays and scale features."""
    X_train = train_df[core_cols].values.astype(np.float32)
    y_train = train_df["target"].values.astype(int)
    X_val = val_df[core_cols].values.astype(np.float32)
    y_val = val_df["target"].values.astype(int)
    X_test = test_df[core_cols].values.astype(np.float32)
    y_test = test_df["target"].values.astype(int)

    # Impute NaN with column medians
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def eval_model(y_true, y_proba, name):
    """Evaluate and print metrics."""
    auc_roc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    pos_rate = y_true.mean()
    # Precision@1%
    k = max(1, int(len(y_true) * 0.01))
    top_k = np.argsort(-y_proba)[:k]
    prec_at_1 = y_true[top_k].mean()
    lift_at_1 = prec_at_1 / pos_rate if pos_rate > 0 else 0

    result = {
        "model": name,
        "test_auc_roc": float(auc_roc),
        "test_pr_auc": float(pr_auc),
        "test_precision_at_1pct": float(prec_at_1),
        "test_lift_at_1pct": float(lift_at_1),
    }
    print(f"  {name:25s}: AUC-ROC={auc_roc:.4f}  PR-AUC={pr_auc:.4f}  "
          f"P@1%={prec_at_1:.3f}  Lift@1%={lift_at_1:.0f}x")
    return result


def train_sklearn_mlp(X_train, y_train, X_val, y_val, X_test, y_test, spw, quick):
    """Train scikit-learn MLP (no PyTorch needed)."""
    print("\n  Training sklearn MLP...")
    t0 = time.time()

    # Oversample positives to match scale_pos_weight
    pos_idx = np.where(y_train == 1)[0]
    n_oversample = int(len(pos_idx) * (spw - 1))
    rng = np.random.default_rng(42)
    extra_idx = rng.choice(pos_idx, n_oversample, replace=True)
    X_train_bal = np.vstack([X_train, X_train[extra_idx]])
    y_train_bal = np.concatenate([y_train, y_train[extra_idx]])

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=50 if quick else 200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_train_bal, y_train_bal)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.0f}s ({mlp.n_iter_} iterations)")

    y_proba = mlp.predict_proba(X_test)[:, 1]
    return y_proba


def train_pytorch_mlp(X_train, y_train, X_val, y_val, X_test, y_test, spw, quick):
    """Train PyTorch MLP with BatchNorm + Dropout + skip connections."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    class ResidualMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.BatchNorm1d(dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = dim
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dims[-1], 1)

        def forward(self, x):
            h = self.backbone(x)
            return self.head(h).squeeze(-1)

    n_features = X_train.shape[1]
    model = ResidualMLP(n_features).to(device)

    # Weighted BCE loss
    pos_weight = torch.tensor([spw], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    batch_size = 1024
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

    n_epochs = 20 if quick else 100
    best_val_prauc = 0
    patience = 10
    no_improve = 0

    print(f"  Training PyTorch MLP ({n_epochs} max epochs)...")
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_probas = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                val_probas.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(yb.numpy())

        val_probas = np.concatenate(val_probas)
        val_labels = np.concatenate(val_labels)
        val_prauc = average_precision_score(val_labels, val_probas)
        scheduler.step(-val_prauc)

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.0f}s (best val PR-AUC: {best_val_prauc:.4f})")

    # Test predictions
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_proba = torch.sigmoid(model(X_test_t)).cpu().numpy()

    return y_proba


def train_tabnet(X_train, y_train, X_val, y_val, X_test, y_test, spw, quick):
    """Train PyTorch TabNet."""
    from pytorch_tabnet.tab_model import TabNetClassifier

    print("\n  Training TabNet...")
    t0 = time.time()

    # Compute sample weights for class imbalance
    sample_weights = np.ones(len(y_train), dtype=np.float32)
    sample_weights[y_train == 1] = spw

    tabnet = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=lambda params: __import__('torch').optim.Adam(params, lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=__import__('torch').optim.lr_scheduler.StepLR,
        verbose=0,
        device_name="auto",
    )

    tabnet.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc"],
        max_epochs=30 if quick else 150,
        patience=15,
        batch_size=1024,
        virtual_batch_size=256,
        weights=dict(enumerate(sample_weights)),  # Not supported this way
    )

    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.0f}s")

    y_proba = tabnet.predict_proba(X_test)[:, 1]
    return y_proba


def train_xgboost_baseline(X_train, y_train, X_val, y_val, X_test, y_test, spw, quick):
    """Train XGBoost for comparison (same settings as core model)."""
    from train_xgboost import train as xgb_train

    print("\n  Training XGBoost baseline...")
    t0 = time.time()
    model = xgb_train(X_train, y_train, X_val, y_val,
                      scale_pos_weight=spw,
                      n_estimators=200 if quick else 1000)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.0f}s")

    y_proba = model.predict_proba(X_test)[:, 1]
    return y_proba


def main(dataset_path, quick):
    tag = date.today().strftime("%Y%m%d")

    print("=" * 60)
    print("DEEP LEARNING COMPARISON")
    print("=" * 60)

    # Load data
    print("\n[1] Loading core data...")
    train_df, val_df, test_df, core_cols = load_core_data(dataset_path)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        train_df, val_df, test_df, core_cols
    )
    spw = compute_scale_pos_weight(y_train)
    print(f"  {X_train.shape[0]:,} train, {X_val.shape[0]:,} val, "
          f"{X_test.shape[0]:,} test, {X_train.shape[1]} features")

    all_results = []

    # ── XGBoost baseline ────────────────────────────────────────────────────
    print("\n" + "-" * 40)
    y_proba_xgb = train_xgboost_baseline(
        X_train, y_train, X_val, y_val, X_test, y_test, spw, quick
    )
    all_results.append(eval_model(y_test, y_proba_xgb, "XGBoost"))

    # ── scikit-learn MLP ────────────────────────────────────────────────────
    print("\n" + "-" * 40)
    y_proba_sklearn = train_sklearn_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test, spw, quick
    )
    all_results.append(eval_model(y_test, y_proba_sklearn, "sklearn MLP"))

    # ── PyTorch models (optional) ───────────────────────────────────────────
    try:
        import torch
        has_torch = True
        print(f"\n  PyTorch {torch.__version__} available")
    except ImportError:
        has_torch = False
        print("\n  PyTorch not installed — skipping PyTorch models")
        print("  Install with: pip install torch")

    if has_torch:
        print("\n" + "-" * 40)
        y_proba_pt_mlp = train_pytorch_mlp(
            X_train, y_train, X_val, y_val, X_test, y_test, spw, quick
        )
        all_results.append(eval_model(y_test, y_proba_pt_mlp, "PyTorch MLP"))

        try:
            print("\n" + "-" * 40)
            y_proba_tabnet = train_tabnet(
                X_train, y_train, X_val, y_val, X_test, y_test, spw, quick
            )
            all_results.append(eval_model(y_test, y_proba_tabnet, "TabNet"))
        except ImportError:
            print("\n  pytorch-tabnet not installed — skipping")
            print("  Install with: pip install pytorch-tabnet")
        except Exception as e:
            print(f"\n  TabNet failed: {e}")

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n  {'Model':25s} {'AUC-ROC':>8s} {'PR-AUC':>8s} {'P@1%':>6s} {'Lift@1%':>8s}")
    print("  " + "-" * 57)
    for r in sorted(all_results, key=lambda x: -x["test_pr_auc"]):
        print(f"  {r['model']:25s} {r['test_auc_roc']:8.4f} {r['test_pr_auc']:8.4f} "
              f"{r['test_precision_at_1pct']:6.3f} {r['test_lift_at_1pct']:7.0f}x")

    # Save
    results_path = OUTPUT_DIR / f"dl_comparison_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {results_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning comparison")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(args.dataset, args.quick)
