"""
Batch-extract audio features for all DEAM clips.
Saves per-clip results to data/processed/deam/features/ as .npz files,
skipping any corrupted/unreadable files.
"""

import sys
sys.path.append("src")

from pathlib import Path
import pandas as pd
from audio_features import load_config, process_track, AudioLoadError
import numpy as np
from tqdm import tqdm

config = load_config("config.yaml")

annotations = pd.read_csv("data/processed/deam/annotations_merged.csv")

raw_root = Path("data/raw/deam/MEMD_audio")
out_root = Path("data/processed/deam/features")
out_root.mkdir(parents=True, exist_ok=True)

skipped = []
processed_count = 0

for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
    song_id = int(row["song_id"])
    mp3_path = raw_root / f"{song_id}.mp3"

    if not mp3_path.exists():
        skipped.append((str(mp3_path), "file not found"))
        continue

    try:
        result = process_track(str(mp3_path), config)
    except AudioLoadError as e:
        skipped.append((str(mp3_path), str(e)))
        continue

    out_path = out_root / f"{song_id}.npz"
    np.savez(
        out_path,
        full_log_mel=result["full_log_mel"],
        segment_features=np.array(result["segment_features"]),
        duration_sec=result["duration_sec"],
        n_segments=result["n_segments"],
        song_id=song_id,
    )
    processed_count += 1

print(f"\nProcessed: {processed_count} clips")
print(f"Skipped: {len(skipped)}")
for path, err in skipped:
    print(f"  - {path}: {err}")