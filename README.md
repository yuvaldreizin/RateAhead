# GrossAhead — Movie Gross Prediction

## Project summary

GrossAhead explores predicting a movie’s box-office gross from production and
metadata features, including numeric attributes (budget, votes, runtime, year)
and categorical fields (genre, director, writer, star, country, company).

The target is modeled as `log1p(gross)` to stabilize scale and improve training.

The dataset is the Kaggle Movie Industry CSV located at
`data/raw/movies.csv`.

---

## Repo structure

- `data/raw/` — raw Kaggle CSV (`movies.csv`)
- `data/processed/` — deterministic splits and experiment-specific subsets
  - `all_features/` — default train/val/test CSVs
  - `up_to_2005/` — pretraining split (movies released up to 2005)
  - `from_2005/` — post-2005 split with per-year evaluation buckets
- `notebooks/` — experiments and analysis
  - `01_data_quality_eda.ipynb` — schema inspection, missingness, target coverage
  - `02_baseline_torch_poc.ipynb` — end-to-end PyTorch baseline
  - `03_baseline_model.ipynb` — hyperparameter grid search
  - `04_separated_vs_shared_encoding.ipynb` — shared vs separated encoders
  - `05_pretrained_finetune.ipynb` — pretraining, zero-shot, finetuning
- `notebooks/artifacts/` — plots, metrics, checkpoints, and prediction files
- `src/` — core library code
  - `data/` — filtering, splitting, feature encoding
  - `models/` — encoders, heads, training loop, regressor
  - `utils/` — metrics, checkpointing, plotting

---

## Environment setup

### Option A: Conda

```
conda env create -f environment.yml
conda activate rateahead
```

### Option B: Python virtual environment

```
python -m venv .venv
```

Windows:
```
.venv\Scripts\activate
```

macOS / Linux:
```
source .venv/bin/activate
```

```
pip install -r requirements.txt
```

---

## Modeling pipeline (overview)

The project follows a standard tabular machine learning pipeline:

- Data preparation  
Rows with missing required values are filtered out. Rare categorical values are
grouped into an `Other` category. Categorical features are frequency-encoded using
statistics fit on the training split only.

- Feature processing  
Selected numeric features and the target variable are transformed using log1p,
followed by standardization.

- Modeling  
A multilayer perceptron regressor is used, with either a single shared encoder or
separate encoders for numeric and categorical feature groups.

- Training and evaluation  
Models are trained in PyTorch using MSE loss, with validation-based checkpointing
and evaluation on held-out test sets.

Implementation details live in `src/`, while experiments are conducted via `notebooks/`.

---

## Main experiments

- `04_separated_vs_shared_encoding.ipynb` — compares a single shared encoder
  against feature-group-specific encoders (financial, creative, metadata). The
  training loop and prediction head are kept fixed to isolate representation
  effects. Outputs are saved under
  `notebooks/artifacts/separated_vs_shared_encoding/`.
- `05_pretrained_finetune.ipynb` — pretrains a model on movies released up to
  2005, evaluates zero-shot generalization on yearly buckets after 2005, and
  applies finetuning variants (full-model, head-only, adapter-based) to study
  data efficiency. Results are saved under
  `notebooks/artifacts/pretrained_finetune/`.

---

## Data splits (deterministic)

Default train/validation/test splits are generated under
`data/processed/all_features/` using:

```
python -m src.data.load_data
```

All splits use a fixed random seed. Precomputed pretraining splits and post-2005
yearly evaluation buckets are provided under `data/processed/up_to_2005/` and
`data/processed/from_2005/`.

---

## Running notebooks

1. Activate the environment
2. Open a notebook under `notebooks/`
3. Run all cells sequentially

A correct setup produces stable loss curves and reasonable metrics in
`03_baseline_model.ipynb`.

---

## Notes

This repository is structured for reproducible experimentation and clear
comparison between modeling choices. All reported results are generated directly
from the notebooks and saved under `notebooks/artifacts/`.
