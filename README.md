# Deforestation Prediction — Congo Basin

Multi-dimensional deforestation risk prediction for the Congo Basin. Combines 33 data sources (satellite imagery, climate, demographics, conflict, economic, political, cultural) to predict WHERE deforestation will occur.

Key contribution: triple ablation study (dimension x temporal scale x spatial resolution) with SHAP interpretability.

## Quick Start

```bash
conda env create -f environment.yml
conda activate deforest
pip install -e .
```

## Usage

```bash
# Train the core model
conda run -n deforest python scripts/train_core_model.py

# Run ablation study
conda run -n deforest python scripts/ablation_study.py

# Launch Streamlit demo
conda run -n deforest streamlit run app/app.py

# Explore data in Marimo
marimo edit notebooks/01_eda.py
```

## Project Structure

See [CLAUDE.md](CLAUDE.md) for detailed structure and conventions.

## Tech Stack

Python 3.11, XGBoost, scikit-learn, SHAP, Google Earth Engine, GeoPandas, Marimo, Streamlit.

## Authors

Guillaume Maitre-Jean, Marc Bouvier
