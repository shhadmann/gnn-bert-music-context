"""
Batch-extract audio features for all available MusicCaps clips.
Saves per-clip results to data/processed/musiccaps/features/ as .npz
files, skipping any corrupted/unreadable files.
"""

import sys
sys.path.append("src")

from pathlib import Path
import pandas as pd
from audio_features import load_config, process_track, AudioLoadError
import numpy as np
from tqdm import tqdm

config = load_config("config.yaml")

df = pd.read_csv("data/processed/musiccaps/available_clips.csv")

raw_root = Path("data/raw/musiccaps/audio")
out_root = Path("data/processed/musiccaps/features")
out_root.mkdir(parents=True, exist_ok=True)

skipped = []
processed_count = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    ytid = row["ytid"]
    mp3_path = raw_root / f"{ytid}.mp3"

    if not mp3_path.exists():
        skipped.append((ytid, "file not found"))
        continue

    try:
        result = process_track(str(mp3_path), config, segment_duration_override=3)
    except AudioLoadError as e:
        skipped.append((ytid, str(e)))
        continue

    out_path = out_root / f"{ytid}.npz"
    np.savez(
        out_path,
        full_log_mel=result["full_log_mel"],
        segment_features=np.array(result["segment_features"]),
        duration_sec=result["duration_sec"],
        n_segments=result["n_segments"],
        ytid=ytid,
    )
    processed_count += 1

print(f"\nProcessed: {processed_count} clips")
print(f"Skipped: {len(skipped)}")
for ytid, err in skipped[:20]:
    print(f"  - {ytid}: {err}")
if len(skipped) > 20:
    print(f"  ... and {len(skipped) - 20} more")