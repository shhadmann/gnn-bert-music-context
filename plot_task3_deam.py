import json
import matplotlib.pyplot as plt
from pathlib import Path

with open("results/task3_deam_history.json") as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
val_r2 = [h["valence_r2"] for h in history]
aro_r2 = [h["arousal_r2"] for h in history]
val_mae = [h["valence_mae"] for h in history]
aro_mae = [h["arousal_mae"] for h in history]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(epochs, val_r2, label="Valence R²", marker=".")
axes[0].plot(epochs, aro_r2, label="Arousal R²", marker=".")
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("R²")
axes[0].set_title("DEAM: R² vs Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, val_mae, label="Valence MAE", marker=".")
axes[1].plot(epochs, aro_mae, label="Arousal MAE", marker=".")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].set_title("DEAM: MAE vs Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("Task 3, Stage B: DEAM Emotion Regression")
plt.tight_layout()

out_path = Path("results/plots/task3_deam_curves.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")