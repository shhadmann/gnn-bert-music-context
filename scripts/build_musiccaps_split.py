"""
Build MusicCaps train/val/test split. Each row is one atomic
(audio, caption) pair, so a simple random split is safe — no risk of
the same audio appearing under a different caption across the
train/test boundary, since MusicCaps has exactly one caption per clip.
"""

import json
import random
from pathlib import Path

import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

random.seed(config["seed"])

graphs_dir = Path("data/processed/musiccaps/graphs")
ytids = sorted(p.stem for p in graphs_dir.glob("*.pt"))
random.shuffle(ytids)

n = len(ytids)
n_train = int(n * 0.70)
n_val = int(n * 0.15)

train_ids = ytids[:n_train]
val_ids = ytids[n_train:n_train + n_val]
test_ids = ytids[n_train + n_val:]

splits_dir = Path("data/splits")
splits_dir.mkdir(parents=True, exist_ok=True)

with open(splits_dir / "musiccaps_train.json", "w") as f:
    json.dump(train_ids, f)
with open(splits_dir / "musiccaps_val.json", "w") as f:
    json.dump(val_ids, f)
with open(splits_dir / "musiccaps_test.json", "w") as f:
    json.dump(test_ids, f)

print(f"Train: {len(train_ids)}")
print(f"Val:   {len(val_ids)}")
print(f"Test:  {len(test_ids)}")
print(f"Total: {len(train_ids) + len(val_ids) + len(test_ids)}")