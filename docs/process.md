# Process & Decisions (Baseline Entity-Embedding MLP)

## Dataset
- Kaggle Movie Industry dataset at `data/raw/movies.csv` (15 columns, ~7.6k rows).

## Targets & Scope
- Target: `gross`, modeled as `log1p(gross)` to stabilize scale.
- Goal: establish a clean baseline with reusable preprocessing and model code.

## Filters / Cleaning
- Drop rows missing any required columns: `gross`, numerics (`budget`, `votes`, `runtime`, `year`), and all categoricals.
- No vote/runtime thresholds (removed the ≥50k votes and 30–300 minute bounds).
- Rare categories handled during encoding (train-derived maps), not in the loader.

## Transforms & Encoding (`src/data/encode_features.py`)
- Numerics: `log1p` on `budget` and `votes`; `runtime` and `year` left linear. Standardize numerics after log1p using a train-fitted StandardScaler.
- Categoricals: fill missing with `"Unknown"`, fit rare maps on train with `min_freq=5`, map rares to `"Other"`, then frequency-encode (fit on train, apply to val/test).
- Target: `log1p(gross)`.

## Split
- Train/val/test: 70/15/15, seed 42, shuffle (`src/data/load_data.py`).
- Processed CSVs saved at `data/processed/all_features/{train,val,test}.csv`.

## Model (Entity-Embedding MLP in `notebooks/03_baseline_model.ipynb`)
- Embeddings: one per categorical feature; dim rule `min(50, round(sqrt(cardinality) + 1))`, vocab from train maps with `Unknown`/`Other`.
- Numerics: scaled features from `prepare_features` (first columns of X).
- MLP: hidden sizes `[256, 128, 64]`, ReLU, Dropout(0.1), BatchNorm enabled, output dim 1 (`log1p(gross)`).
- Loss: MSE; Optimizer: Adam lr=1e-3; Training demo: 10 epochs, batch 256 (CPU by default in notebook).
- Tracking: train/val loss curves; test loss reported; sample predictions table (pred vs true in log and raw space).

## How to rerun
1) Ensure `data/raw/movies.csv` is present.
2) Create/activate venv; `pip install -r requirements.txt` (or `environment.yml`).
3) Regenerate splits via `python -m src.data.load_data` (writes to `data/processed/all_features/`).
4) Run `notebooks/03_baseline_model.ipynb`:
   - Uses `enc.prepare_features` for preprocessing.
   - Trains the entity-embedding MLP, plots train/val loss, evaluates test loss, and shows sample predictions.
