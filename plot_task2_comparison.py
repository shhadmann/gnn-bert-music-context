import json
import matplotlib.pyplot as plt
from pathlib import Path

with open("results/task2_gnn_history.json") as f:
    gnn_history = json.load(f)
with open("results/task2_cnn_history.json") as f:
    cnn_history = json.load(f)
with open("results/task2_gnn_test_metrics.json") as f:
    gnn_test = json.load(f)
with open("results/task2_cnn_test_metrics.json") as f:
    cnn_test = json.load(f)

# --- Plot 1: training curves side by side ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

gnn_epochs = [h["epoch"] for h in gnn_history]
gnn_f1 = [h["macro_f1"] for h in gnn_history]
gnn_aucpr = [h["aucpr"] for h in gnn_history]

cnn_epochs = [h["epoch"] for h in cnn_history]
cnn_f1 = [h["macro_f1"] for h in cnn_history]
cnn_aucpr = [h["aucpr"] for h in cnn_history]

axes[0].plot(gnn_epochs, gnn_f1, label="GNN", marker=".")
axes[0].plot(cnn_epochs, cnn_f1, label="CNN baseline", marker=".")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation Macro-F1")
axes[0].set_title("Macro-F1 vs Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(gnn_epochs, gnn_aucpr, label="GNN", marker=".")
axes[1].plot(cnn_epochs, cnn_aucpr, label="CNN baseline", marker=".")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation AUC-PR")
axes[1].set_title("AUC-PR vs Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("Task 2: GNN vs. CNN Baseline — GTZAN Genre Classification")
plt.tight_layout()

out_path = Path("results/plots/task2_gnn_vs_cnn_curves.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# --- Plot 2: final test metrics bar chart ---
fig, ax = plt.subplots(figsize=(6, 5))
models = ["GNN\n(segment graphs)", "CNN baseline\n(full mel-spectrogram)"]
macro_f1_vals = [gnn_test["macro_f1"], cnn_test["macro_f1"]]
aucpr_vals = [gnn_test["aucpr"], cnn_test["aucpr"]]

x = range(len(models))
width = 0.35
ax.bar([i - width/2 for i in x], macro_f1_vals, width, label="Macro-F1")
ax.bar([i + width/2 for i in x], aucpr_vals, width, label="AUC-PR")
ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.set_ylabel("Score")
ax.set_title("Task 2: Final Test Metrics")
ax.legend()
ax.grid(alpha=0.3, axis="y")

for i, v in enumerate(macro_f1_vals):
    ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center")
for i, v in enumerate(aucpr_vals):
    ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center")

out_path2 = Path("results/plots/task2_final_comparison.png")
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2}")

# --- Comparison table, saved as both console output and JSON ---
comparison = {
    "GNN (segment graphs)": {"epochs_trained": len(gnn_history), **gnn_test},
    "CNN baseline (full mel-spectrogram)": {"epochs_trained": len(cnn_history), **cnn_test},
}
print("\nFinal comparison:")
for model, metrics in comparison.items():
    print(f"  {model}: {metrics}")

with open("results/task2_comparison_table.json", "w") as f:
    json.dump(comparison, f, indent=2)
print("\nSaved: results/task2_comparison_table.json")