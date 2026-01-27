# GrossAhead — Movie Gross Prediction

## Project Overview

GrossAhead studies prediction of a movie’s box-office gross using only production and metadata features.
Inputs combine numeric attributes (budget, votes, runtime, release year) and categorical fields (genre, director, writer, star, company, country).

Beyond prediction accuracy, the project is designed as a controlled study of representation learning and adaptation for tabular data.

### Research Questions

1. Does semantic feature grouping help?
We compare a fully shared baseline against models that introduce structure by grouping features (e.g., financial, creative, temporal) and performing feature ablations, to assess whether semantic inductive bias improves generalization.

2. How do fine-tuning strategies behave under temporal shift?
Models are pretrained on pre-2005 data and fine-tuned on post-2005 movies. We compare full fine-tuning, head-only training, gradual unfreezing, and parameter-efficient methods (LoRA, DoRA) under a fixed optimization budget, focusing on cost–performance trade-offs.

### 🔗 Project Links

📄 **Full Report** | 🎥 **Presentation Video**  
[Read the Report](https://technionmail-my.sharepoint.com/:b:/g/personal/yuvaldreizin_campus_technion_ac_il/IQDypq9iCq_-QYglx4VJPdB7AbDCftK9SIllhR6OBU8V8Z4?e=KQdMN3)  
[Watch the Presentation](https://www.youtube.com/watch?v=XlzKDcFw71I)

---

## Repo structure

- `data/raw/` — raw Kaggle CSV (`movies.csv`)
- `data/processed/` — deterministic splits and experiment-specific subsets
  - `all_features/` — default train/val/test CSVs
  - `up_to_2005/` — pretraining split (movies released up to 2005)
  - `from_2005/` — post-2005 split with per-year evaluation buckets
- `notebooks/` — experiments and analysis
  - `01_data_quality_eda.ipynb` — schema inspection, missingness, target coverage
  - `02_baseline_model.ipynb` — baseline model selection (includes hyperparameter grid search)
  - `03_seperated_vs_shared_encoding.ipynb` — shared vs separated encoders
  - `04_pretrained_finetune.ipynb` — pretraining, zero-shot, finetuning
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

- `03_seperated_vs_shared_encoding.ipynb` — compares a single shared encoder
  against feature-group-specific encoders (financial, creative, metadata). The
  training loop and prediction head are kept fixed to isolate representation
  effects. Outputs are saved under
  `notebooks/artifacts/seperated_vs_shared_encoding/`.
- `04_pretrained_finetune.ipynb` — pretrains a model on movies released up to
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
`02_baseline_model.ipynb`.

---

## Notes

This repository is structured for reproducible experimentation and clear
comparison between modeling choices. All reported results are generated directly
from the notebooks and saved under `notebooks/artifacts/`.
