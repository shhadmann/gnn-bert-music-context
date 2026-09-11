# GNN-BERT Music Context Understanding

CSE425 / EEE474 / CSE715 - Neural Networks project.

A hybrid GNN + BERT system for understanding musical context across
four tasks: multi-label tag classification, genre classification,
GNN-BERT fusion (with an emotion-regression extension) and
cross-modal audio-caption retrieval.

**Full report:** [`report/final_report.pdf`](report/final_report.pdf)
**Full results:** [`results/master_results_table.md`](results/master_results_table.md)

## Key Results

| Task | Headline result |
|---|---|
| Task 1 - BERT tag classification | Macro-F1 = 0.311, AUC-PR = 0.295 |
| Task 2 - GNN vs. CNN (GTZAN) | GNN Macro-F1 = 0.289; CNN Macro-F1 = 0.693 |
| Task 3 - Fusion ablations (best) | Early Concat / Cross-Attention Macro-F1 ≈ 0.32–0.33 |
| Task 3 - DEAM emotion regression | Valence R² = 0.075, Arousal R² = 0.108 |
| Task 4 - Contrastive retrieval | R@10 ≈ 0.03–0.035 (2.3–2.8× random chance) |
| Task 4 - Human evaluation | Mean 3.36/5 across 5 listeners, 10 items |

See the report for full discussion and analysis of every result.

## Repository Structure

```text
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml                # all hyperparameters, fixed seed (42)
│
├── src/                       # core, reusable pipeline code
│   ├── audio_features.py      # resample, log-mel, chroma, segmentation
│   ├── graph_builder.py       # segment graph construction
│   ├── bert_encoder.py        # Task 1 BERT tag classifier
│   ├── gnn_model.py           # Task 2/3 GNN + CNN baseline + emotion regressor
│   ├── fusion_model.py        # Task 3 cross-attention / early-concat fusion
│   ├── contrastive.py         # Task 4 dual-encoder + InfoNCE
│   ├── train.py               # all training loops
│   └── evaluate.py            # consolidated re-verification of every result
│
├── scripts/                   # one-off dataset download/processing/analysis
│                              # scripts, run from repo root, e.g.:
│                              #   python scripts/build_gtzan_split.py
│
├── data/
│   ├── raw/                   # gitignored — GTZAN, MagnaTagATune, DEAM, MusicCaps
│   ├── processed/             # gitignored — extracted features (.npz), graphs (.pt)
│   └── splits/                # tracked — train/val/test membership per dataset
│
├── results/                   # final metrics, plots, master results table
├── notebooks/
│   └── demo_context.ipynb     # end-to-end inference demo
└── report/
    └── final_report.pdf       # final report (NeurIPS format)  

```
