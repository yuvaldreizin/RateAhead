# RateAhead — Movie Gross Prediction

## Concept
Predict a movie's box-office gross from its metadata and production features
(`budget`, `votes`, `runtime`, `year`, plus categorical fields like `genre`,
`director`, `writer`, `star`, `country`, `company`). The target is modeled as
`log1p(gross)` to stabilize scale and training.

The dataset is the Kaggle Movie Industry CSV (`data/raw/movies.csv`).

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

## Notebooks (03+)
- `notebooks/03_baseline_model.ipynb` — main tabular baseline using shared
  preprocessing + model helpers in `src/`. Loads the processed splits, builds
  feature matrices, trains the entity-embedding MLP, and writes artifacts to
  `notebooks/artifacts/baseline/` (loss curves, metrics, predictions, checkpoints).
- `notebooks/04_hl_gauss_compare.ipynb` — compares standard MSE regression to
  HL-Gauss regression-as-classification. Trains HL-Gauss models over a small
  `(num_bins, sigma_ratio)` sweep, selects by val MSE, and produces side-by-side
  plots/metrics in `notebooks/artifacts/hl_gauss/`.
- `notebooks/05_seperated_vs_shared_encoding.ipynb` — evaluates shared encoding
  vs feature-group-specific encoders (financial/creative/metadata). Keeps the
  architecture fixed, compares training histories and test metrics, and stores
  results under `notebooks/artifacts/seperated_vs_shared_encoding/`.
- `notebooks/06_Mixture_Of_Experts.ipynb` — hierarchical mixture-of-experts
  model with a 3x5 gating/expert tree. Logs routing diagnostics and specialization
  stats; artifacts live under `notebooks/artifacts/hme_3x5/`.
- `notebooks/07_pretrained_finetune.ipynb` — scaffold for pretrain (<=2004)
  → zero-shot yearly eval → finetune experiments. Defines the data split logic
  and helper stubs for full finetune vs head-only/adapters.

## Structure
- `notebooks/` — experiments and analysis notebooks.
- `src/` — reusable data prep, models, training, and utils.
- `experiments/` — YAML configs for experiment variants.
- `data/raw/movies.csv` — source dataset (Kaggle Movie Industry).
- `docs/process.md` — step-by-step data filters, transforms, model, and signals.

## Environment setup
### Option A: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate rateahead
```

### Option B: Python venv
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