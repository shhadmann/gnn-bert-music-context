"""
Retry only the MusicCaps clips that failed in the first pass, now with
--js-runtimes deno --remote-components ejs:github, which fixes YouTube's
JS challenge requirement that caused the original failures.
"""

import subprocess
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

df = pd.read_csv("data/raw/musiccaps/musiccaps-public.csv")

# Extract just the failed ytids from the log (tab-separated, first column)
failure_log = Path("data/raw/musiccaps/download_failures.txt")
failed_ytids = set()
with open(failure_log, encoding="utf-8") as f:
    for line in f:
        if "\t" in line:
            failed_ytids.add(line.split("\t")[0].strip())

print(f"Retrying {len(failed_ytids)} previously-failed clips")

retry_df = df[df["ytid"].isin(failed_ytids)]

out_dir = Path("data/raw/musiccaps/audio")
out_dir.mkdir(parents=True, exist_ok=True)
tmp_dir = Path("data/raw/musiccaps/_tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

still_failing = []
succeeded = 0

for _, row in tqdm(retry_df.iterrows(), total=len(retry_df)):
    ytid = row["ytid"]
    start = row["start_s"]
    end = row["end_s"]
    duration = end - start

    final_path = out_dir / f"{ytid}.mp3"
    if final_path.exists():
        continue

    url = f"https://www.youtube.com/watch?v={ytid}"
    raw_path = tmp_dir / f"{ytid}_raw.mp3"

    dl_cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "--js-runtimes", "deno",
        "--remote-components", "ejs:github",
        "-o", str(raw_path),
        "--quiet", "--no-warnings",
        url,
    ]
    dl_result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=90)

    if dl_result.returncode != 0 or not raw_path.exists():
        still_failing.append((ytid, dl_result.stderr[-300:]))
        continue

    trim_cmd = [
        "ffmpeg", "-y", "-i", str(raw_path),
        "-t", str(duration),
        str(final_path),
    ]
    trim_result = subprocess.run(trim_cmd, capture_output=True, text=True, timeout=30)
    raw_path.unlink(missing_ok=True)

    if trim_result.returncode != 0 or not final_path.exists():
        still_failing.append((ytid, "trim_failed: " + trim_result.stderr[-200:]))
        continue

    succeeded += 1
    time.sleep(0.3)

print(f"\nRecovered: {succeeded}")
print(f"Still failing: {len(still_failing)}")

with open("data/raw/musiccaps/download_failures_retry.txt", "w", encoding="utf-8") as f:
    for ytid, detail in still_failing:
        f.write(f"{ytid}\t{detail}\n")