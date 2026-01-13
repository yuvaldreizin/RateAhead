---
name: Pretrained-to-Finetune Experiments
overview: Detailed step-by-step notebook and experiment plan covering data splits, pretraining, zero-shot eval, fine-tuning variants (full, head-only, layer-wise, LoRA), epoch/data sweeps, and reporting with plots.
todos:
  - id: scaffold-notebook
    content: Create notebook sections and configs
    status: completed
  - id: define-splits
    content: Describe train<=2000 and yearly eval buckets
    status: completed
    dependencies:
      - scaffold-notebook
  - id: pretrain-plan
    content: Outline baseline pretraining run and logging
    status: completed
    dependencies:
      - define-splits
  - id: zero-shot-plan
    content: Plan zero-shot yearly eval and storage
    status: completed
    dependencies:
      - pretrain-plan
  - id: ft-variants-plan
    content: Detail FT methods and epoch sweeps
    status: pending
    dependencies:
      - zero-shot-plan
  - id: post-eval-plan
    content: Plan post-FT eval and delta calc
    status: pending
    dependencies:
      - ft-variants-plan
  - id: aggregation-plan
    content: Plan tidy results schema and saves
    status: pending
    dependencies:
      - post-eval-plan
  - id: plots-plan
    content: Plan plots (year, delta, epochs, data frac, efficiency)
    status: pending
    dependencies:
      - aggregation-plan
  - id: reporting-plan
    content: Plan markdown takeaways and notes
    status: pending
    dependencies:
      - plots-plan
  - id: todo-1768295477883-tfgzdf0f3
    content: ""
    status: pending
---

# Plan: Pretrained → Finetune Notebook & Experiments (detailed)

## Scope

- Use baseline model (existing encoder+head) for pretraining on movies released <=2000.
- Evaluate per-year slices 2000–2020 and aggregate post-2000.
- Fine-tune variants: full FT, head-only, layer-wise unfreeze, LoRA/adapters.
- Sweeps: epoch budgets (e.g., 1/3/5/10) and optional data fractions (e.g., 10/30/100%).
- Outputs: tidy results tables (CSV/JSON) and plots for per-year, delta vs zero-shot, epoch/data-size, efficiency.

## Step-by-step tasks

- Notebook scaffolding in `notebooks/05_pretrained_finetune.ipynb`:
- Sections: Intro/goals; Imports & config; Data split logic; Training/Eval utils; Pretraining run; Zero-shot eval; Fine-tune variants; Post-FT eval; Results aggregation; Plots; Reporting/notes.
- Data split (read-only plan reference to `src/data/load_data.py`):
- Define train set: movies <= 2000.
- Eval buckets: one loader per year 2000–2020; aggregate post-2000 set.
- Ensure consistent preprocessing with baseline.
- Pretraining run (reuse `src/models/train.py`/`model.py`):
- Load data splits; run baseline training; log metrics/checkpoints.
- Save config, seed, and checkpoint path for reuse.
- Zero-shot eval:
- Load pretrain checkpoint; loop yearly eval loaders; compute task metrics (e.g., RMSE/MAE/accuracy); store DataFrame and CSV/JSON.
- Fine-tuning variants (per method):
- Full FT: unfreeze all layers; shorter schedule; lower LR; epoch sweep.
- Head-only: freeze encoder; (optional) re-init head; epoch sweep.
- Layer-wise: gradual unfreeze (last block → full); tuned LR; epoch sweep.
- LoRA/adapters: add lightweight modules to encoder; train them + head; epoch sweep.
- For each: log per-epoch metrics, wall-clock, data fraction used, seed, config.
- Post-FT eval:
- Same yearly eval loop; annotate results with method, epochs, data_frac, wall_time.
- Aggregate per-year and overall; compute deltas vs zero-shot.
- Results aggregation:
- Tidy DataFrame columns: [method, year, metric, data_frac, epochs, wall_time, seed].
- Save combined CSV/JSON; include std if multi-seed.
- Plots:
- Metric vs year overlay by method (incl. zero-shot).
- Delta vs zero-shot per year.
- Metric vs epochs per method; facet by data fraction if run.
- Metric vs data fraction per method; optional separate curves by epoch budget.
- Efficiency: metric vs total time or epochs.
- Aggregate distribution (box/violin) across years per method; table of mean±std.
- Reporting (markdown cells):
- Document configs, seeds, data splits, sweeps.
- Summarize gains/losses per method, epoch-budget effects, data-size sensitivity, efficiency trade-offs.
- Note open questions/future tweaks.

## Outputs

- Structured notebook with code/markdown for the above steps.
- Saved artifacts under `notebooks/artifacts/` (plots, CSV/JSON results).