"""
Batch-extract audio features for the filtered MagnaTagATune clips
(top-50 tag subset, 21,111 clips). Saves per-clip results to
data/processed/magnatagatune/features/ as .npz files, skipping any
corrupted/unreadable files, same as the GTZAN pipeline.
"""

import sys
sys.path.append("src")

from pathlib import Path
import pandas as pd
from audio_features import load_config, process_track, AudioLoadError
import numpy as np
from tqdm import tqdm

config = load_config("config.yaml")

df = pd.read_csv("data/processed/magnatagatune/filtered_annotations.csv")

raw_root = Path("data/raw/magnatagatune")
out_root = Path("data/processed/magnatagatune/features")
out_root.mkdir(parents=True, exist_ok=True)

skipped = []
processed_count = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    clip_id = row["clip_id"]
    mp3_path = raw_root / row["mp3_path"]

    if not mp3_path.exists():
        skipped.append((str(mp3_path), "file not found"))
        continue

    try:
        result = process_track(str(mp3_path), config)
    except AudioLoadError as e:
        skipped.append((str(mp3_path), str(e)))
        continue

    out_path = out_root / f"{clip_id}.npz"
    np.savez(
        out_path,
        full_log_mel=result["full_log_mel"],
        segment_features=np.array(result["segment_features"]),
        duration_sec=result["duration_sec"],
        n_segments=result["n_segments"],
        clip_id=clip_id,
    )
    processed_count += 1

print(f"\nProcessed: {processed_count} clips")
print(f"Skipped: {len(skipped)}")
for path, err in skipped[:20]:  # show first 20 max
    print(f"  - {path}: {err}")
if len(skipped) > 20:
    print(f"  ... and {len(skipped) - 20} more")

# Save skip log for reference
with open(out_root.parent / "skipped_clips.txt", "w") as f:
    for path, err in skipped:
        f.write(f"{path}\t{err}\n")