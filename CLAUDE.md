# CLAUDE.md — Deforestation Prediction (Congo Basin)

## Project Overview

Multi-dimensional deforestation risk prediction for the Congo Basin. Combines satellite imagery, climate, demographics, conflict, economic, political, and cultural data to predict WHERE deforestation will occur. Key contribution: triple ablation study (dimension × temporal scale × spatial resolution) with SHAP interpretability.

## Repository Structure

```
├── app/               # Streamlit web demo
├── biblio/            # Literature review, references.bib
├── configs/           # Hydra/YAML experiment configs
├── data/              # .gitignored — raw GeoTIFFs, extracted features
│   ├── train_test/    # Train/test splits
│   ├── val/           # Validation splits
│   └── app/           # Precomputed data for Streamlit app
├── models/            # Saved model artifacts (.gitignored)
├── notebooks/         # Marimo notebooks (.py)
├── results/           # Plots, metrics, reports (.gitignored)
├── scripts/           # CLI scripts (extraction, training, ablation)
├── src/
│   ├── data/          # GEE extraction, data loading, sampling
│   ├── features/      # Temporal profiles, feature engineering
│   ├── models/        # XGBoost, MLP, training logic
│   ├── evaluation/    # Metrics, SHAP, spatial analysis
│   └── inference/     # Prediction pipeline, map generation
└── tests/             # Unit and integration tests
```

## Tech Stack

- **Python 3.10+**, XGBoost, scikit-learn, SHAP
- **Marimo** notebooks (NOT Jupyter)
- **Google Earth Engine** Python API for data extraction (no local rasters)
- **GeoPandas** + rasterio for vector/raster operations
- **Streamlit** for web demo (deployed on Streamlit Cloud / HF Spaces)
- Optional: PyTorch + TerraTorch for foundation model embeddings (Prithvi-EO)

## Data Pipeline

Data is extracted via GEE at 250K sample points (from legacy project). No large rasters stored locally.

Sources (33 total): Hansen GFC, Sentinel-2, SRTM, CHIRPS daily, ERA5-Land daily, VIIRS, MODIS fire, WorldPop, ACLED, WDPA, mining/forestry concessions, V-Dem, World Bank WDI, commodity prices, GREG ethnic groups, etc.

## Key Conventions

- All temporal features encoded as profiles: value, Δ1yr, Δ3yr, Δ10yr, anomaly, volatility
- Daily data (CHIRPS, ERA5) aggregated server-side in GEE before extraction
- Temporal split: train ≤2020, val 2021, test 2022 (strict, no leakage)
- Spatial split: geographic blocks to avoid autocorrelation
- Prediction is PROBABILISTIC: P(deforestation) ∈ [0,1], not binary

## GEE

- Project ID: `ee-guillaumemaitrejean`
- Init helper: `src/data/gee_utils.py` → `init_gee()`
- Hansen GFC: use `UMD/hansen/global_forest_change_2024_v1_12` (v1.12, data through 2024)

## Environment

```bash
conda activate deforest
```

## Commands

```bash
# Install
conda env create -f environment.yml
conda activate deforest

# --- Data extraction & enrichment ---
conda run -n deforest python scripts/scale_up_extraction.py   # Full 250K extraction via GEE
conda run -n deforest python scripts/add_sources.py           # Add new GEE sources to existing data
conda run -n deforest python scripts/add_spatial_features.py  # Compute spatial features
conda run -n deforest python scripts/add_tabular_features.py  # Compute tabular features
conda run -n deforest python scripts/add_buffers.py           # Add buffer-based features
conda run -n deforest python scripts/enrich_wdpa.py           # Enrich WDPA protected areas
conda run -n deforest python scripts/patch_worldpop.py        # Patch WorldPop data

# --- Dataset building ---
conda run -n deforest python scripts/build_traintest.py       # Build train/test split
conda run -n deforest python scripts/build_val.py             # Build validation split

# --- Training ---
conda run -n deforest python scripts/train_core_model.py      # Train main XGBoost model
conda run -n deforest python scripts/tune_xgboost.py          # Hyperparameter tuning (Optuna)
conda run -n deforest python scripts/train_xgboost.py         # Train XGBoost baseline
conda run -n deforest python scripts/train_deep_learning.py   # Train DL model

# --- Evaluation & ablation ---
conda run -n deforest python scripts/shap_deep_dive.py        # SHAP interpretability analysis
conda run -n deforest python scripts/ablation_study.py        # Full ablation study
conda run -n deforest python scripts/temporal_ablation.py     # Temporal scale ablation
conda run -n deforest python scripts/spatial_ablation.py      # Spatial resolution ablation
conda run -n deforest python scripts/test_no_tautology.py     # Verify no data leakage
conda run -n deforest python scripts/improve_prauc.py         # PR-AUC optimization

# --- Export & app ---
conda run -n deforest python scripts/export_predictions.py    # Export prediction maps
conda run -n deforest streamlit run app/app.py                # Run Streamlit demo

# --- Notebooks ---
marimo edit notebooks/01_eda.py
marimo edit notebooks/02_gee_extraction.py
```

## Out of Scope

- Do not modify files in data/ directly (regenerate via scripts)
- Do not commit data/ or models/ directories
- Do not install packages outside conda environment `deforest`
- Do not use Jupyter notebooks — use Marimo (.py format)

## Language

- Communication: French
- Code, commits, technical docs: English

## Legacy Project

Original project (2023) at `/Users/guillaume/Documents/Clouds/UGA/Recherche/deforestation/`:
- `src/data/tiles_250000_10N_020E_20231023.csv` — 250K tile coordinates + labels
- `src/ML/dataset_balanced.csv` — 65K balanced samples
- `src/data/data_gen.ipynb` — tiling logic
