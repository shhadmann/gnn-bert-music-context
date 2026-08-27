"""
Merge DEAM's two static annotation files (songs 1-2000 and 2000-2058)
into one clean CSV with stripped column names, matched against the
actual available audio files.
"""

import pandas as pd
from pathlib import Path

ann_dir = Path("data/raw/deam/annotations/annotations averaged per song/song_level")

cols_needed = ["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"]

s1 = pd.read_csv(ann_dir / "static_annotations_averaged_songs_1_2000.csv")
s1.columns = s1.columns.str.strip()
s1 = s1[cols_needed]

s2 = pd.read_csv(ann_dir / "static_annotations_averaged_songs_2000_2058.csv")
s2.columns = s2.columns.str.strip()
s2 = s2[cols_needed]

merged = pd.concat([s1, s2], ignore_index=True)
print(f"Merged annotations: {merged.shape}")

# Match against actual audio files present
audio_dir = Path("data/raw/deam/MEMD_audio")
audio_song_ids = set(int(p.stem) for p in audio_dir.glob("*.mp3"))
print(f"Audio files available: {len(audio_song_ids)}")

merged = merged[merged["song_id"].isin(audio_song_ids)].reset_index(drop=True)
print(f"Annotations with matching audio: {len(merged)}")

out_dir = Path("data/processed/deam")
out_dir.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_dir / "annotations_merged.csv", index=False)
print(f"Saved: {out_dir / 'annotations_merged.csv'}")
print(merged.head())