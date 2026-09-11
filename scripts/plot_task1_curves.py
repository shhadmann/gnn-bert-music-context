import json
import matplotlib.pyplot as plt
from pathlib import Path

with open("results/task1_history.json") as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
macro_f1 = [h["macro_f1"] for h in history]
micro_f1 = [h["micro_f1"] for h in history]
aucpr = [h["aucpr"] for h in history]

plt.figure(figsize=(8, 5))
plt.plot(epochs, macro_f1, marker="o", label="Val Macro-F1")
plt.plot(epochs, micro_f1, marker="o", label="Val Micro-F1")
plt.plot(epochs, aucpr, marker="o", label="Val AUC-PR")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Task 1: BERT Tag Classifier — Validation Metrics vs Epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(epochs)

out_path = Path("results/plots/task1_metrics_curve.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")