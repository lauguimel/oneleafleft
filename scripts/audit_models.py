"""Audit saved XGBoost model artifacts and emit a Markdown inventory.

Usage::

    conda run -n deforest python scripts/audit_models.py \
        [--models-dir models] [--extra-dir data] \
        [--output results/models_inventory.md]

Scans the given directories for booster JSON files produced by the project's
training scripts, classifies each as ``tuned_xgboost`` vs ``train_xgboost``,
and writes a human-readable Markdown inventory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.model_registry import (
    ModelArtifact,
    discover_artifacts,
    render_markdown_table,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_report(artifacts: list[ModelArtifact]) -> str:
    """Build the full Markdown report body.

    Args:
        artifacts: List of discovered artifacts.

    Returns:
        Markdown string including title, summary counts, and the table.
    """
    n_tuned = sum(1 for a in artifacts if a.kind == "tuned_xgboost")
    n_default = sum(1 for a in artifacts if a.kind == "train_xgboost")
    lines = [
        "# Model Inventory",
        "",
        f"- Total artifacts: **{len(artifacts)}**",
        f"- Tuned (Optuna): **{n_tuned}**",
        f"- Default training: **{n_default}**",
        "",
        "## Artifacts",
        "",
        render_markdown_table(artifacts),
    ]
    return "\n".join(lines)


def run_audit(
    search_dirs: list[Path],
    output_path: Path,
) -> list[ModelArtifact]:
    """Run the full audit pipeline.

    Args:
        search_dirs: Directories to scan for model artifacts.
        output_path: Path where the Markdown inventory will be written.

    Returns:
        List of discovered artifacts.
    """
    artifacts = discover_artifacts(search_dirs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(artifacts))
    return artifacts


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Primary models directory to scan.",
    )
    parser.add_argument(
        "--extra-dir",
        type=Path,
        action="append",
        default=None,
        help="Additional directory to scan (can be repeated). "
        "Defaults to [data/] if not given.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "models_inventory.md",
        help="Markdown file to write.",
    )
    args = parser.parse_args()

    extras = args.extra_dir if args.extra_dir is not None else [PROJECT_ROOT / "data"]
    dirs = [args.models_dir, *extras]

    artifacts = run_audit(dirs, args.output)
    print(f"Found {len(artifacts)} artifact(s). Wrote {args.output}")
    for a in artifacts:
        print(f"  [{a.kind:14s}] {a.path.name}")


if __name__ == "__main__":
    main()
