# Master Results Table

All numbers verified reproducible via `src/evaluate.py` — recomputed metrics matched originally-saved training results to within floating-point precision.

## Task 1: BERT Tag Classifier (MagnaTagATune)

| Metric | Value |
|---|---|
| Macro-F1 | 0.3115 |
| Micro-F1 | 0.3830 |
| AUC-PR | 0.2949 |
| Threshold used | 0.10 |

## Task 2: GNN vs. CNN Baseline (GTZAN)

| Model | Macro-F1 | AUC-PR |
|---|---|---|
| GNN (segment graphs) | 0.2892 | 0.3417 |
| CNN baseline (full mel-spectrogram) | 0.6926 | 0.7596 |

## Task 3, Stage A: GNN-BERT Fusion Ablations (MagnaTagATune)

| Ablation | Macro-F1 | Micro-F1 | AUC-PR | Threshold |
|---|---|---|---|---|
| BERT-only | 0.3115 | 0.3830 | 0.2949 | 0.10 |
| GNN-only | 0.1248 | 0.1790 | 0.1089 | 0.05 |
| Early Concat | 0.3318 | 0.3844 | 0.2983 | 0.15 |
| Cross-Attention | 0.3168 | 0.3975 | 0.2956 | 0.20 |

## Task 3, Stage B: DEAM Emotion Regression (extension)

| Target | MAE | R² |
|---|---|---|
| Valence | 0.8799 | 0.0750 |
| Arousal | 0.9642 | 0.1083 |

Loss weights: α=0.5, β=0.5 (verified comparable valence/arousal loss magnitudes during training)

## Task 4: Contrastive Retrieval (MusicCaps)

| Direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| Audio → Caption | 0.0050 | 0.0202 | 0.0353 |
| Caption → Audio | 0.0076 | 0.0151 | 0.0290 |

For reference, random-chance R@k on 793 test candidates: R@1≈0.0013, R@5≈0.0063, R@10≈0.0126
