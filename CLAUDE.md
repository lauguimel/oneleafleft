# CLAUDE.md — Deforestation Prediction (Congo Basin)

## Project Overview

Multi-dimensional deforestation risk prediction for the Congo Basin. Combines satellite imagery, climate, demographics, conflict, economic, political, and cultural data to predict WHERE deforestation will occur. Key contribution: triple ablation study (dimension × temporal scale × spatial resolution) with SHAP interpretability.

## Repository Structure

```
├── configs/           # Hydra/YAML experiment configs
├── data/              # .gitignored — raw GeoTIFFs, extracted features
├── notebooks/         # Marimo notebooks (.py)
├── scripts/           # CLI scripts (train.py, extract.py)
├── src/
│   ├── data/          # GEE extraction, data loading, sampling
│   ├── features/      # Temporal profiles, feature engineering
│   ├── models/        # XGBoost, MLP, training logic
│   ├── evaluation/    # Metrics, SHAP, spatial analysis
│   └── inference/     # Prediction pipeline, map generation
└── tests/
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

## Commands

```bash
# Install
conda env create -f environment.yml
conda activate deforest

# Run Marimo notebook
marimo edit notebooks/01_eda.py

# Run as app
marimo run notebooks/app_demo.py
```

## Legacy Project

Original project (2023) at `/Users/guillaume/Documents/Clouds/UGA/Recherche/deforestation/`:
- `src/data/tiles_250000_10N_020E_20231023.csv` — 250K tile coordinates + labels
- `src/ML/dataset_balanced.csv` — 65K balanced samples
- `src/data/data_gen.ipynb` — tiling logic
