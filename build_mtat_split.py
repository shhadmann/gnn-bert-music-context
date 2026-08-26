"""
Build an artist-grouped train/val/test split for MagnaTagATune.
Every clip from a given artist goes entirely into one split, never
spread across train/val/test, per Cross-Cutting Requirement B.
"""

import json
import random
from pathlib import Path

import pandas as pd
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

random.seed(config["seed"])

# Ground truth: only clips that actually succeeded in feature extraction
features_dir = Path("data/processed/magnatagatune/features")
processed_clip_ids = set(int(p.stem) for p in features_dir.glob("*.npz"))
print(f"Successfully processed clips: {len(processed_clip_ids)}")

# Get artist info for each processed clip
clip_info = pd.read_csv("data/raw/magnatagatune/clip_info_final.csv", sep="\t")
clip_info = clip_info[clip_info["clip_id"].isin(processed_clip_ids)]
print(f"Clips with matching artist info: {len(clip_info)}")

# Group clip_ids by artist
artist_to_clips = clip_info.groupby("artist")["clip_id"].apply(list).to_dict()
artists = list(artist_to_clips.keys())
random.shuffle(artists)
print(f"Unique artists: {len(artists)}")

total_clips = sum(len(v) for v in artist_to_clips.values())
train_target = int(total_clips * 0.70)
val_target = int(total_clips * 0.15)

train_clips, val_clips, test_clips = [], [], []
train_count = val_count = 0

for artist in artists:
    clips = artist_to_clips[artist]
    if train_count < train_target:
        train_clips.extend(clips)
        train_count += len(clips)
    elif val_count < val_target:
        val_clips.extend(clips)
        val_count += len(clips)
    else:
        test_clips.extend(clips)

splits_dir = Path("data/splits")
splits_dir.mkdir(parents=True, exist_ok=True)

for name, clip_list in [("train", train_clips), ("val", val_clips), ("test", test_clips)]:
    with open(splits_dir / f"mtag_{name}.json", "w") as f:
        json.dump(clip_list, f)

print(f"\nTrain: {len(train_clips)}")
print(f"Val:   {len(val_clips)}")
print(f"Test:  {len(test_clips)}")
print(f"Total: {len(train_clips) + len(val_clips) + len(test_clips)}")

# Leakage check: confirm no artist appears in more than one split
train_artists = set(clip_info[clip_info["clip_id"].isin(train_clips)]["artist"])
val_artists = set(clip_info[clip_info["clip_id"].isin(val_clips)]["artist"])
test_artists = set(clip_info[clip_info["clip_id"].isin(test_clips)]["artist"])

overlap_tv = train_artists & val_artists
overlap_tt = train_artists & test_artists
overlap_vt = val_artists & test_artists

print(f"\nArtist overlap train/val: {len(overlap_tv)}")
print(f"Artist overlap train/test: {len(overlap_tt)}")
print(f"Artist overlap val/test: {len(overlap_vt)}")