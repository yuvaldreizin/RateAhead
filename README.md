# RateAhead — Movie Gross Prediction PoC

## What this repo shows
- Data quality EDA on the Kaggle Movie Industry dataset.
- A quick PyTorch PoC to confirm the data is learnable for predicting `gross` (target is `log1p(gross)`), with stable, converging loss curves.

## Workflow (current)
1) EDA (`notebooks/01_data_quality_eda.ipynb`)
   - Inspect schema, missingness, target coverage, heavy tails, and rare categories.
2) Baseline Torch PoC (`notebooks/02_baseline_torch_poc.ipynb`)
   - Filters: drop missing `gross`/`votes`/`runtime`; votes ≥ 50,000; runtime 30–300; rare categories (freq < 5) → `Other`.
   - Transforms: `log1p` on `budget`, `votes`, and target `gross`; medians for missing numerics; standardize numerics.
   - Encoding: frequency encoding on categoricals (fit on train, applied to val).
   - Split: 80/20 train/val (seed 42).
   - Model: tiny MLP (128-ReLU-Dropout0.1 → 64-ReLU → 1), MSE loss, Adam lr=1e-3, 50 epochs, batch 64.
   - Signal: train/val MSE/MAE decrease and level off; val MAE ≈ 0.72 log space (~$80M) with visible convergence in the loss plots.

## Structure
- `notebooks/`
  - `01_data_quality_eda.ipynb` — EDA and data quality decisions.
  - `02_baseline_torch_poc.ipynb` — filtering, encoding, and the overfit Torch baseline on `log1p(gross)`.
  - `03_feature_analysis.ipynb` — placeholder for future analysis.
- `src/` — stubs for data/model/utils (to be filled).
- `experiments/` — config placeholders (TBD).
- `data/raw/movies.csv` — source dataset (Kaggle Movie Industry).
- `docs/process.md` — step-by-step data filters, transforms, model, and PoC signals.

## Setup (Python venv)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```
Place the Kaggle CSV at `data/raw/movies.csv`.

## Running the PoC
1) Activate the venv (above).
2) Open `notebooks/02_baseline_torch_poc.ipynb` and run all cells (expects `data/raw/movies.csv` in place).

The PoC’s loss curves serve as the “works as expected” signal for this repo right now.

For the full step-by-step pipeline (filters, encoding, model, and observed metrics), see `docs/process.md`.

## Datasets (deterministic splits)
- Baseline filtered splits are written to `notebooks/data/processed/{train,val,test}.csv` (70/15/15, seed=42) via `python -m src.data.load_data` or `build_and_save_baseline_splits()`.
- Filters: drop missing gross/votes/runtime; votes ≥ 50k; runtime 30–300; rare cats freq<5 → `Other`.
- Transforms: log1p on budget, votes, and target gross; medians for missing numerics; standardize numerics; frequency encoding for categoricals.