"""
Filter MagnaTagATune annotations down to the top-50 most frequent tags,
per the spec's Task 1 deliverable ("MagnaTagATune tag subset (top-50 tags)").
Keeps only clips with at least one positive tag among the top-50
(standard practice for this dataset — most of the 188 tags are rare/noisy).
"""

import json
from pathlib import Path

import pandas as pd

df = pd.read_csv("data/raw/magnatagatune/annotations_final.csv", sep="\t")

tag_columns = [c for c in df.columns if c not in ("clip_id", "mp3_path")]
print(f"Total tags available: {len(tag_columns)}")

# Rank tags by frequency (number of clips with that tag = 1)
tag_counts = df[tag_columns].sum().sort_values(ascending=False)
top50_tags = tag_counts.head(50).index.tolist()

print("\nTop 10 tags by frequency:")
print(tag_counts.head(10))

# Keep only clip_id, mp3_path, and the top-50 tag columns
filtered = df[["clip_id", "mp3_path"] + top50_tags].copy()

# Keep only clips with at least one positive tag among the top-50
has_any_top50_tag = filtered[top50_tags].sum(axis=1) > 0
filtered = filtered[has_any_top50_tag].reset_index(drop=True)

out_dir = Path("data/processed/magnatagatune")
out_dir.mkdir(parents=True, exist_ok=True)

filtered.to_csv(out_dir / "filtered_annotations.csv", index=False)
with open(out_dir / "top50_tags.json", "w") as f:
    json.dump(top50_tags, f, indent=2)

print(f"\nOriginal clips: {len(df)}")
print(f"Clips with >=1 top-50 tag: {len(filtered)}")
print(f"Saved: {out_dir / 'filtered_annotations.csv'}")
print(f"Saved: {out_dir / 'top50_tags.json'}")