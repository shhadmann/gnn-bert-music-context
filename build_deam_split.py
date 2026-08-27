"""
Build DEAM's train/val split (per the spec's minimum requirement),
plus an additional held-out test split as this project's own added
practice (documented explicitly, not spec-required). Split at the
song level using the seed from config.yaml.
"""

import json
import random
from pathlib import Path

import pandas as pd
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

random.seed(config["seed"])

features_dir = Path("data/processed/deam/features")
song_ids = sorted(int(p.stem) for p in features_dir.glob("*.npz"))
print(f"Total songs with features: {len(song_ids)}")

random.shuffle(song_ids)

n = len(song_ids)
n_train = int(n * 0.70)
n_val = int(n * 0.15)
# remainder -> test_extension

train_ids = song_ids[:n_train]
val_ids = song_ids[n_train:n_train + n_val]
test_ids = song_ids[n_train + n_val:]

splits_dir = Path("data/splits")
splits_dir.mkdir(parents=True, exist_ok=True)

with open(splits_dir / "deam_train.json", "w") as f:
    json.dump(train_ids, f)
with open(splits_dir / "deam_val.json", "w") as f:
    json.dump(val_ids, f)
with open(splits_dir / "deam_test_extension.json", "w") as f:
    json.dump(test_ids, f)

print(f"Train: {len(train_ids)}")
print(f"Val:   {len(val_ids)}")
print(f"Test (project extension, beyond spec minimum): {len(test_ids)}")
print(f"Total: {len(train_ids) + len(val_ids) + len(test_ids)}")