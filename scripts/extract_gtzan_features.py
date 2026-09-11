"""
Batch-extract audio features for all GTZAN tracks.
Saves per-track results to data/processed/gtzan/ as .npz files
(one per track), skipping any corrupted/unreadable files.
"""

import sys
sys.path.append("src")

from pathlib import Path
from audio_features import load_config, process_track, AudioLoadError
import numpy as np
from tqdm import tqdm

config = load_config("config.yaml")

raw_root = Path("data/raw/gtzan/genres_original")
out_root = Path("data/processed/gtzan")
out_root.mkdir(parents=True, exist_ok=True)

genres = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

skipped = []
processed_count = 0

for genre in genres:
    genre_folder = raw_root / genre
    out_genre_folder = out_root / genre
    out_genre_folder.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(genre_folder.glob("*.wav"))
    for wav_path in tqdm(wav_files, desc=genre):
        try:
            result = process_track(str(wav_path), config)
        except AudioLoadError as e:
            skipped.append((str(wav_path), str(e)))
            continue

        out_path = out_genre_folder / (wav_path.stem + ".npz")
        np.savez(
            out_path,
            full_log_mel=result["full_log_mel"],
            segment_features=np.array(result["segment_features"]),
            duration_sec=result["duration_sec"],
            n_segments=result["n_segments"],
            genre=genre,
        )
        processed_count += 1

print(f"\nProcessed: {processed_count} files")
print(f"Skipped (corrupted/unreadable): {len(skipped)}")
for path, err in skipped:
    print(f"  - {path}")