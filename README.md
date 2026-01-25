# RateAhead — Movie Gross Prediction

## Project summary
RateAhead explores predicting a movie's box-office gross from metadata and production
features (`budget`, `votes`, `runtime`, `year`, plus categorical fields like
`genre`, `director`, `writer`, `star`, `country`, `company`). The target is modeled
as `log1p(gross)` to stabilize scale and training. The dataset is the Kaggle Movie
Industry CSV at `data/raw/movies.csv`.

## Repo structure
- `data/raw/` — raw Kaggle CSV (`movies.csv`).
- `data/processed/` — deterministic splits and experiment-specific subsets:
  - `all_features/` — default train/val/test CSVs written by
    `python -m src.data.load_data`.
  - `up_to_2005/` — pretrain split (train/val/test up to 2005).
  - `from_2005/` — post-2005 split and per-year CSVs under `by_year/`.
- `notebooks/` — experiment notebooks:
  - `01_data_quality_eda.ipynb` — schema, missingness, target coverage.
  - `02_baseline_torch_poc.ipynb` — first Torch baseline end-to-end.
  - `03_baseline_model.ipynb` — baseline using reusable `src/` helpers.
  - `04_seperated_vs_shared_encoding.ipynb` — shared vs split encoders.
  - `05_pretrained_finetune.ipynb` — pretrain → zero-shot by year → finetune.
- `notebooks/artifacts/` — plots, metrics, checkpoints, and predictions from notebooks.
- `src/` — reusable code:
  - `src/data/` — constants, filtering/splitting, feature encoding.
  - `src/models/` — encoders, heads, training loop, and tabular regressor.
  - `src/utils/` — checkpoints, metrics, plotting.
- `artifacts/` — example baseline outputs (loss curves, metrics, checkpoints).

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

## Project workflow
The baseline pipeline is implemented in `src/data/` and `src/models/`:
- Filtering: keep rows with required numeric/target values.
- Categorical prep: fill missing values with `Unknown`, group rare categories to
  `Other`, then frequency-encode using train data only.
- Numeric prep: `log1p` selected columns, median impute missing values, and
  standardize numeric features.
- Modeling: MLP regressor with shared or separate encoders for numeric vs
  categorical features (see `src/models/model.py`).
- Training: PyTorch loop with MSE loss (or optional HL-Gauss via config), tracked
  by val metrics and best-epoch checkpointing.

## Main experiments (notebooks 4 & 5)
### `04_seperated_vs_shared_encoding.ipynb`
Compares a single shared encoder against feature-group-specific encoders for
financial, creative, and metadata features. The experiment keeps the training
loop and head fixed, then measures how representation sharing affects validation
loss and test metrics. Outputs include per-model training histories, prediction
files, and summary tables under `notebooks/artifacts/seperated_vs_shared_encoding/`.

### `05_pretrained_finetune.ipynb`
Pretrains on movies up to 2005, evaluates zero-shot generalization on yearly
buckets after 2005, then runs finetune variants (full-model, head-only, or
adapter-based) to quantify data-efficiency. Results include per-run metrics,
zero-shot error vs year plots, and aggregate summaries in
`notebooks/artifacts/pretrained_finetune/`.

## Data splits (deterministic)
Generate the default train/val/test CSVs under `data/processed/all_features/`:
```bash
python -m src.data.load_data
```
This uses a fixed seed and 70/15/15 split. Precomputed splits and year buckets
for the pretrain/finetune notebook live under `data/processed/up_to_2005/` and
`data/processed/from_2005/`.

## Running notebooks
1) Activate the environment.
2) Open a notebook under `notebooks/` and run all cells.

The baseline “works as expected” signal is the loss curve and metrics produced by
`notebooks/02_baseline_torch_poc.ipynb` and `notebooks/03_baseline_model.ipynb`.