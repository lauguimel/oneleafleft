import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Deforestation Prediction — Congo Basin
        ## 01 — Exploratory Data Analysis

        This notebook explores the legacy dataset (250K tiles from Hansen GFC 2022)
        and prepares the sample points for the new multi-dimensional pipeline.
        """
    )
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from pathlib import Path

    # Paths to legacy data
    LEGACY_DIR = Path("/Users/guillaume/Documents/Clouds/UGA/Recherche/deforestation")
    TILES_CSV = LEGACY_DIR / "src/data/tiles_250000_10N_020E_20231023.csv"
    BALANCED_CSV = LEGACY_DIR / "src/ML/dataset_balanced.csv"

    PROJECT_DIR = Path("/Users/guillaume/Documents/Recherche/Deforestation")
    return BALANCED_CSV, LEGACY_DIR, PROJECT_DIR, Path, TILES_CSV, np, pd


@app.cell
def _(TILES_CSV, pd):
    # Load the full 250K tiles dataset
    tiles = pd.read_csv(TILES_CSV)
    print(f"Tiles dataset: {tiles.shape[0]:,} rows × {tiles.shape[1]} columns")
    print(f"\nColumns: {list(tiles.columns)}")
    tiles.head()
    return (tiles,)


@app.cell
def _(tiles):
    # Basic statistics
    print("=== Target variable: lossyear_22_mean ===")
    print(tiles["lossyear_22_mean"].describe())
    print(f"\nNon-zero (deforested): {(tiles['lossyear_22_mean'] > 0).sum():,}")
    print(f"Zero (no deforestation): {(tiles['lossyear_22_mean'] == 0).sum():,}")
    print(f"Deforestation rate: {(tiles['lossyear_22_mean'] > 0).mean():.2%}")
    return


@app.cell
def _(mo, np, tiles):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Map of sample points
    ax = axes[0]
    sample = tiles.sample(min(10000, len(tiles)), random_state=42)
    scatter = ax.scatter(
        sample["longitude"],
        sample["latitude"],
        c=np.where(sample["lossyear_22_mean"] > 0, 1, 0),
        cmap="RdYlGn_r",
        s=0.5,
        alpha=0.5,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Sample points (red = deforested 2022)")

    # Distribution of lossyear_22_mean (non-zero only)
    ax = axes[1]
    nonzero = tiles["lossyear_22_mean"][tiles["lossyear_22_mean"] > 0]
    ax.hist(nonzero, bins=50, color="firebrick", alpha=0.7)
    ax.set_xlabel("lossyear_22_mean")
    ax.set_ylabel("Count")
    ax.set_title("Distribution (non-zero)")

    # Treecover2000 distribution
    ax = axes[2]
    ax.hist(tiles["treecover2000_mean"], bins=50, color="forestgreen", alpha=0.7)
    ax.set_xlabel("treecover2000_mean")
    ax.set_ylabel("Count")
    ax.set_title("Tree cover 2000")

    plt.tight_layout()
    mo.output.replace(fig)
    return axes, fig, nonzero, sample, scatter


@app.cell
def _(tiles):
    # Extract just the coordinates and target for the new pipeline
    sample_points = tiles[["longitude", "latitude", "lossyear_22_mean"]].copy()
    sample_points.columns = ["lon", "lat", "deforestation_2022"]

    # Also extract historical deforestation columns
    loss_cols = [c for c in tiles.columns if c.startswith("lossyear_") and c != "lossyear_22_mean"]
    for col in loss_cols:
        sample_points[col] = tiles[col]

    print(f"Sample points: {sample_points.shape}")
    print(f"Columns: {list(sample_points.columns)}")
    sample_points.head()
    return loss_cols, sample_points


@app.cell
def _(PROJECT_DIR, sample_points):
    # Save coordinates for the new pipeline
    output_path = PROJECT_DIR / "data" / "sample_points.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_points.to_parquet(output_path, index=False)
    print(f"Saved {len(sample_points):,} points to {output_path}")
    return (output_path,)


if __name__ == "__main__":
    app.run()
