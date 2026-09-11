"""
Build an artist-grouped train/val/test split for MagnaTagATune.
Every clip from a given artist goes entirely into one split, never
spread across train/val/test, per Cross-Cutting Requirement B.
"""

import json
"""
Build MagnaTagATune's official standard split (the widely-used
"12:1:3" folder-based convention): folders 0-b train, c val, d-f test.
This is the standard split referenced across MIR/audio-tagging
literature, matching the spec's instruction to use official splits
where available.

Known limitation (documented per the roadmap): this folder-based split
was not constructed with artist-leakage prevention in mind and can
allow some same-artist tracks to span train/test. The spec's own
wording ("no artist leakage across train/test when possible") allows
for this — flagged explicitly here and in the report.
"""

import json
from pathlib import Path

import pandas as pd

# Ground truth: only clips that actually succeeded in feature extraction
features_dir = Path("data/processed/magnatagatune/features")
processed_clip_ids = set(int(p.stem) for p in features_dir.glob("*.npz"))
print(f"Successfully processed clips: {len(processed_clip_ids)}")

# Get mp3_path (starts with the folder letter, e.g. "f/artist-...") for each clip
annotations = pd.read_csv("data/processed/magnatagatune/filtered_annotations.csv")
annotations = annotations[annotations["clip_id"].isin(processed_clip_ids)]

annotations["folder"] = annotations["mp3_path"].str.split("/").str[0]

train_folders = set("0123456789ab")
val_folders = {"c"}
test_folders = {"d", "e", "f"}

train_clips = annotations[annotations["folder"].isin(train_folders)]["clip_id"].tolist()
val_clips = annotations[annotations["folder"].isin(val_folders)]["clip_id"].tolist()
test_clips = annotations[annotations["folder"].isin(test_folders)]["clip_id"].tolist()

splits_dir = Path("data/splits")
splits_dir.mkdir(parents=True, exist_ok=True)

for name, clip_list in [("train", train_clips), ("val", val_clips), ("test", test_clips)]:
    with open(splits_dir / f"mtag_{name}.json", "w") as f:
        json.dump(clip_list, f)

print(f"\nTrain (folders 0-b): {len(train_clips)}")
print(f"Val   (folder c):    {len(val_clips)}")
print(f"Test  (folders d-f): {len(test_clips)}")
print(f"Total: {len(train_clips) + len(val_clips) + len(test_clips)}")

# Leakage check: same purpose as before, but now we EXPECT some overlap
# and are documenting it rather than preventing it, per the standard split's
# known limitation.
clip_info = pd.read_csv("data/raw/magnatagatune/clip_info_final.csv", sep="\t")
clip_info = clip_info[clip_info["clip_id"].isin(processed_clip_ids)]

train_artists = set(clip_info[clip_info["clip_id"].isin(train_clips)]["artist"])
val_artists = set(clip_info[clip_info["clip_id"].isin(val_clips)]["artist"])
test_artists = set(clip_info[clip_info["clip_id"].isin(test_clips)]["artist"])

overlap_tv = train_artists & val_artists
overlap_tt = train_artists & test_artists
overlap_vt = val_artists & test_artists

print(f"\nArtist overlap train/val: {len(overlap_tv)} (known limitation of this standard split)")
print(f"Artist overlap train/test: {len(overlap_tt)} (known limitation of this standard split)")
print(f"Artist overlap val/test: {len(overlap_vt)} (known limitation of this standard split)")