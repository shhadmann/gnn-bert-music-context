"""
Build a single master results table across all four tasks, pulling
from final_summary_all_tasks.json (the verified, reproducible numbers).
Outputs both a printed table and a markdown table for direct use in
the report.
"""

import json

with open("results/final_summary_all_tasks.json") as f:
    summary = json.load(f)

lines = []
lines.append("# Master Results Table\n")
lines.append("All numbers verified reproducible via `src/evaluate.py` — recomputed metrics matched originally-saved training results to within floating-point precision.\n")

lines.append("## Task 1: BERT Tag Classifier (MagnaTagATune)\n")
t1 = summary["task1_bert"]["originally_saved"]
lines.append("| Metric | Value |")
lines.append("|---|---|")
lines.append(f"| Macro-F1 | {t1['macro_f1']:.4f} |")
lines.append(f"| Micro-F1 | {t1['micro_f1']:.4f} |")
lines.append(f"| AUC-PR | {t1['aucpr']:.4f} |")
lines.append(f"| Threshold used | {t1['threshold_used']:.2f} |\n")

lines.append("## Task 2: GNN vs. CNN Baseline (GTZAN)\n")
t2g = summary["task2_gnn"]["originally_saved"]
t2c = summary["task2_cnn"]["originally_saved"]
lines.append("| Model | Macro-F1 | AUC-PR |")
lines.append("|---|---|---|")
lines.append(f"| GNN (segment graphs) | {t2g['macro_f1']:.4f} | {t2g['aucpr']:.4f} |")
lines.append(f"| CNN baseline (full mel-spectrogram) | {t2c['macro_f1']:.4f} | {t2c['aucpr']:.4f} |\n")

lines.append("## Task 3, Stage A: GNN-BERT Fusion Ablations (MagnaTagATune)\n")
lines.append("| Ablation | Macro-F1 | Micro-F1 | AUC-PR | Threshold |")
lines.append("|---|---|---|---|---|")
for mode, label in [("task3_bert_only", "BERT-only"), ("task3_gnn_only", "GNN-only"),
                     ("task3_early_concat", "Early Concat"), ("task3_cross_attention", "Cross-Attention")]:
    m = summary[mode]["originally_saved"]
    lines.append(f"| {label} | {m['macro_f1']:.4f} | {m['micro_f1']:.4f} | {m['aucpr']:.4f} | {m['threshold_used']:.2f} |")
lines.append("")

lines.append("## Task 3, Stage B: DEAM Emotion Regression (extension)\n")
t3d = summary["task3_deam"]["originally_saved"]
lines.append("| Target | MAE | R² |")
lines.append("|---|---|---|")
lines.append(f"| Valence | {t3d['valence_mae']:.4f} | {t3d['valence_r2']:.4f} |")
lines.append(f"| Arousal | {t3d['arousal_mae']:.4f} | {t3d['arousal_r2']:.4f} |")
lines.append(f"\nLoss weights: α={t3d['alpha']}, β={t3d['beta']} (verified comparable valence/arousal loss magnitudes during training)\n")

lines.append("## Task 4: Contrastive Retrieval (MusicCaps)\n")
t4 = summary["task4_contrastive"]["originally_saved"]
lines.append("| Direction | R@1 | R@5 | R@10 |")
lines.append("|---|---|---|---|")
lines.append(f"| Audio → Caption | {t4['audio_to_caption_R@1']:.4f} | {t4['audio_to_caption_R@5']:.4f} | {t4['audio_to_caption_R@10']:.4f} |")
lines.append(f"| Caption → Audio | {t4['caption_to_audio_R@1']:.4f} | {t4['caption_to_audio_R@5']:.4f} | {t4['caption_to_audio_R@10']:.4f} |")
lines.append(f"\nFor reference, random-chance R@k on 793 test candidates: R@1≈0.0013, R@5≈0.0063, R@10≈0.0126\n")

with open("results/master_results_table.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print("\nSaved: results/master_results_table.md")