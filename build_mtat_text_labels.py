"""
Build the BERT input dataset for Task 1: title + artist + album as
X_text, top-50 tags as the multi-label target. Matched against the
clips that actually succeeded in feature extraction.
"""

import json
from pathlib import Path

import pandas as pd

clip_info = pd.read_csv("data/raw/magnatagatune/clip_info_final.csv", sep="\t")
annotations = pd.read_csv("data/processed/magnatagatune/filtered_annotations.csv")

with open("data/processed/magnatagatune/top50_tags.json") as f:
    top50_tags = json.load(f)

# Merge: clip_id -> title/artist/album + top-50 tag columns
merged = annotations.merge(
    clip_info[["clip_id", "title", "artist", "album"]],
    on="clip_id",
    how="inner",
)
print(f"Annotations: {len(annotations)}")
print(f"Merged with metadata: {len(merged)}")

# Build the text field
merged["text"] = (
    merged["title"].fillna("").astype(str) + " " +
    merged["artist"].fillna("").astype(str) + " " +
    merged["album"].fillna("").astype(str)
).str.strip()

# Check for empty text fields (would be a real problem for BERT input)
empty_text = (merged["text"].str.len() == 0).sum()
print(f"Clips with empty text field: {empty_text}")

out_cols = ["clip_id", "text"] + top50_tags
final = merged[out_cols]

out_path = Path("data/processed/magnatagatune/text_labels.csv")
final.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(final.head(3)[["clip_id", "text"]])