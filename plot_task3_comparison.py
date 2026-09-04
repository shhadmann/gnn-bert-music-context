import json
import matplotlib.pyplot as plt
from pathlib import Path

modes = ["bert_only", "gnn_only", "early_concat", "cross_attention"]
labels = ["BERT-only", "GNN-only", "Early Concat", "Cross-Attention"]

results = {}
for mode in modes:
    with open(f"results/task3_{mode}_test_metrics.json") as f:
        results[mode] = json.load(f)

fig, ax = plt.subplots(figsize=(9, 5.5))
x = range(len(modes))
width = 0.3
macro_f1_vals = [results[m]["macro_f1"] for m in modes]
aucpr_vals = [results[m]["aucpr"] for m in modes]

ax.bar([i - width/2 for i in x], macro_f1_vals, width, label="Macro-F1")
ax.bar([i + width/2 for i in x], aucpr_vals, width, label="AUC-PR")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("Score")
ax.set_title("Task 3: Fusion Ablation Comparison (MagnaTagATune)")
ax.legend()
ax.grid(alpha=0.3, axis="y")

for i, v in enumerate(macro_f1_vals):
    ax.text(i - width/2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
for i, v in enumerate(aucpr_vals):
    ax.text(i + width/2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)

out_path = Path("results/plots/task3_ablation_comparison.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

comparison_table = {labels[i]: results[modes[i]] for i in range(len(modes))}
with open("results/task3_ablation_comparison_table.json", "w") as f:
    json.dump(comparison_table, f, indent=2)
print("\nComparison table:")
for label, metrics in comparison_table.items():
    print(f"  {label}: {metrics}")