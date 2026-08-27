# GNN-BERT Music Context Understanding

CSE425 / EEE474 / CSE715 — Neural Networks project.

## Environment

- Python: 3.13.5
- PyTorch: 2.11.0+cu128 (Colab GPU) / 2.13.0+cpu (local dev)
- CUDA: 12.8 (Colab, Tesla T4)
- Local machine: Ryzen 5 5600G (CPU-only, no discrete GPU)
- Key packages: torch-geometric 2.8.0, transformers, librosa, scikit-learn

## Datasets

- GTZAN — genre classification (Task 2)
- MagnaTagATune — tag classification (Task 1), fusion (Task 3)
- DEAM — emotion regression extension (Task 3, Stage B)
- MusicCaps — contrastive retrieval (Task 4)

## Setup