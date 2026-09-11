"""
Batch-download MusicCaps audio clips via yt-dlp, trimmed to exact
10-second windows with ffmpeg. Resumable (skips already-downloaded
clips) and logs failures without stopping the batch — YouTube-sourced
downloads are expected to have some attrition (deleted/private/
region-locked videos), this is normal, not a bug.
"""

import subprocess
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

df = pd.read_csv("data/raw/musiccaps/musiccaps-public.csv")

out_dir = Path("data/raw/musiccaps/audio")
out_dir.mkdir(parents=True, exist_ok=True)

tmp_dir = Path("data/raw/musiccaps/_tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

log_path = Path("data/raw/musiccaps/download_failures.txt")
failures = []
succeeded = 0
skipped_existing = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    ytid = row["ytid"]
    start = row["start_s"]
    end = row["end_s"]
    duration = end - start

    final_path = out_dir / f"{ytid}.mp3"
    if final_path.exists():
        skipped_existing += 1
        continue

    url = f"https://www.youtube.com/watch?v={ytid}"
    raw_path = tmp_dir / f"{ytid}_raw.mp3"

    dl_cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-o", str(raw_path),
        "--quiet", "--no-warnings",
        url,
    ]
    dl_result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=60)

    if dl_result.returncode != 0 or not raw_path.exists():
        failures.append((ytid, "download_failed", dl_result.stderr[-300:]))
        continue

    trim_cmd = [
        "ffmpeg", "-y", "-i", str(raw_path),
        "-t", str(duration),
        str(final_path),
    ]
    trim_result = subprocess.run(trim_cmd, capture_output=True, text=True, timeout=30)

    raw_path.unlink(missing_ok=True)  # clean up temp file regardless of outcome

    if trim_result.returncode != 0 or not final_path.exists():
        failures.append((ytid, "trim_failed", trim_result.stderr[-300:]))
        continue

    succeeded += 1
    time.sleep(0.3)  # small delay to reduce chance of rate-limiting

print(f"\nSucceeded: {succeeded}")
print(f"Skipped (already existed): {skipped_existing}")
print(f"Failed: {len(failures)}")

with open(log_path, "w", encoding="utf-8") as f:
    for ytid, reason, detail in failures:
        f.write(f"{ytid}\t{reason}\t{detail}\n")
print(f"Failure log saved to: {log_path}")