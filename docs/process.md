# Process & Decisions (EDA → Torch PoC)

## Dataset
- Kaggle Movie Industry dataset at `data/raw/movies.csv` (15 columns, ~7.6k rows).

## Targets & Scope
- Target: `gross`, modeled as `log1p(gross)` to stabilize scale.
- Goal: confirm data is learnable with a small overfitting baseline; not final modeling.

## Filters (applied in PoC)
- Drop rows missing: `gross`, `votes`, `runtime`.
- Votes ≥ 50,000 (remove low-signal titles).
- Runtime between 30 and 300 minutes (remove implausible lengths).
- Rare categories (freq < 5) grouped into `Other` for: `rating`, `genre`, `director`, `writer`, `star`, `country`, `company`.

## Transforms & Encoding
- Numerics: `log1p` on `budget` and `votes`; `runtime` kept linear. Medians for missing numerics. Standardize numerics after log1p.
- Categoricals: frequency encoding (fit on train, applied to val) for the columns above.
- Target: `log1p(gross)`.

## Split
- Train/val split: 80/20, random seed 42.

## Model (tiny MLP)
- Input: scaled numerics + frequency-encoded categoricals (11 dims after pipeline).
- Hidden1: Linear 128 → ReLU → Dropout(0.1)
- Hidden2: Linear 64 → ReLU
- Output: Linear 1 (predicts `log1p(gross)`)
- Loss: MSELoss; Optimizer: Adam lr=1e-3; Epochs: 50; Batch: 64; Device: CPU/GPU auto.

## Observed PoC Signal
- Train/val MSE and MAE decrease and level off (see loss plots in `02_baseline_torch_poc.ipynb`).
- Example run: val MAE ≈ 0.72 (log space), ≈ $80M absolute.
- Convergence of the loss curves is the “works as expected” check for now.

## How to rerun
1) Ensure `data/raw/movies.csv` is present.
2) Create/activate venv; `pip install -r requirements.txt`.
3) Regenerate splits (70/15/15, seed 42) via `python -m src.data.load_data` or `build_and_save_baseline_splits()`, outputs to `notebooks/data/processed/{train,val,test}.csv`.
4) Open `notebooks/02_baseline_torch_poc.ipynb` or run models using configs:
   - `experiments/shared.yaml` (shared encoder)
   - `experiments/separate.yaml` (separate encoders)
   - `experiments/pretrain_finetune.yaml` (old→new flow; adjust paths for old/new splits)
