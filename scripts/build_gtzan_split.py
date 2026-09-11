"""
Build a genre-stratified train/val/test split for GTZAN.
GTZAN has no official spec-mandated split and no artist metadata,
so this is a project-defined split, seeded for reproducibility,
stratified by genre to keep proportions equal across partitions.
"""

import json
import random
from pathlib import Path

import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

seed = config["seed"]
random.seed(seed)

graphs_root = Path("data/processed/gtzan_graphs")
genres = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

train_ratio, val_ratio = 0.70, 0.15  # test = remaining 0.15

train_list, val_list, test_list = [], [], []

for genre in genres:
    genre_folder = graphs_root / genre
    tracks = sorted([p.stem for p in genre_folder.glob("*.pt")])
    random.shuffle(tracks)

    n = len(tracks)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    for t in tracks[:n_train]:
        train_list.append({"genre": genre, "track": t})
    for t in tracks[n_train:n_train + n_val]:
        val_list.append({"genre": genre, "track": t})
    for t in tracks[n_train + n_val:]:
        test_list.append({"genre": genre, "track": t})

splits_dir = Path("data/splits")
splits_dir.mkdir(parents=True, exist_ok=True)

with open(splits_dir / "gtzan_train.json", "w") as f:
    json.dump(train_list, f, indent=2)
with open(splits_dir / "gtzan_val.json", "w") as f:
    json.dump(val_list, f, indent=2)
with open(splits_dir / "gtzan_test.json", "w") as f:
    json.dump(test_list, f, indent=2)

print(f"Train: {len(train_list)}")
print(f"Val:   {len(val_list)}")
print(f"Test:  {len(test_list)}")
print(f"Total: {len(train_list) + len(val_list) + len(test_list)}")

# Verify genre balance across splits
for genre in genres:
    tr = sum(1 for x in train_list if x["genre"] == genre)
    va = sum(1 for x in val_list if x["genre"] == genre)
    te = sum(1 for x in test_list if x["genre"] == genre)
    print(f"{genre}: train={tr}, val={va}, test={te}")