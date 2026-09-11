# GNN-BERT Music Context Understanding

CSE425 / EEE474 / CSE715 — Neural Networks project.

A hybrid GNN + BERT system for understanding musical context across
four tasks: tag classification, genre classification, multi-modal
fusion, and cross-modal retrieval.

## Environment

- Python: 3.13.5
- PyTorch: 2.11.0+cu128 (Colab GPU, Tesla T4) / 2.13.0+cpu (local dev)
- CUDA: 12.8
- Local dev machine: Ryzen 5 5600G (CPU-only, no discrete GPU)
- Key packages: torch-geometric 2.8.0, transformers, librosa, scikit-learn

## Project Structure

- `src/` — core model and pipeline code (audio features, graph
  construction, BERT/GNN/fusion/contrastive models, training)
- `data/splits/` — train/val/test membership per dataset (tracked)
- `data/raw/`, `data/processed/` — datasets and derived features/graphs
  (gitignored — regenerate via the root-level scripts)
- `results/` — final metrics, plots, and the master results table
- `notebooks/demo_context.ipynb` — end-to-end inference demo
- `report/` — final report and supporting documentation

## Tasks

1. **BERT tag classification** (MagnaTagATune, title+artist+album → top-50 tags)
2. **GNN genre classification** (GTZAN segment graphs) + CNN baseline
3. **GNN-BERT fusion** (MagnaTagATune, 4-way ablation) + DEAM emotion regression extension
4. **Contrastive retrieval** (MusicCaps, dual-encoder InfoNCE)

Full results: see `results/master_results_table.md`.

## Setup

​```
pip install -r requirements.txt
​```

See `config.yaml` for all hyperparameters and the fixed seed (42).
Training was run on Google Colab (T4 GPU); preprocessing and evaluation
run locally on CPU.


## Reproducibility

Every task's final metrics were independently re-verified by reloading
saved checkpoints and re-running evaluation via `src/evaluate.py` —
see `results/final_summary_all_tasks.json`.