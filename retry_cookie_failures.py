"""
Retry only the cookie/auth-walled MusicCaps clips using browser cookies.
"""

import subprocess
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

df = pd.read_csv("data/raw/musiccaps/musiccaps-public.csv")

failure_log = Path("data/raw/musiccaps/download_failures_retry.txt")
cookie_ytids = []
with open(failure_log, encoding="utf-8") as f:
    for line in f:
        if "\t" in line and ("cookies" in line.lower() or "from-browser" in line.lower()):
            cookie_ytids.append(line.split("\t")[0].strip())

print(f"Retrying {len(cookie_ytids)} cookie-walled clips")

retry_df = df[df["ytid"].isin(cookie_ytids)]

out_dir = Path("data/raw/musiccaps/audio")
tmp_dir = Path("data/raw/musiccaps/_tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

succeeded = 0
still_failing = []

for _, row in tqdm(retry_df.iterrows(), total=len(retry_df)):
    ytid = row["ytid"]
    start, end = row["start_s"], row["end_s"]
    duration = end - start

    final_path = out_dir / f"{ytid}.mp3"
    if final_path.exists():
        continue

    url = f"https://www.youtube.com/watch?v={ytid}"
    raw_path = tmp_dir / f"{ytid}_raw.mp3"

    dl_cmd = [
        "yt-dlp", "-x", "--audio-format", "mp3",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "--js-runtimes", "deno",
        "--remote-components", "ejs:github",
        "--cookies-from-browser", "chrome",
        "-o", str(raw_path),
        "--quiet", "--no-warnings",
        url,
    ]
    dl_result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=90)

    if dl_result.returncode != 0 or not raw_path.exists():
        still_failing.append((ytid, dl_result.stderr[-200:]))
        continue

    trim_cmd = ["ffmpeg", "-y", "-i", str(raw_path), "-t", str(duration), str(final_path)]
    trim_result = subprocess.run(trim_cmd, capture_output=True, text=True, timeout=30)
    raw_path.unlink(missing_ok=True)

    if trim_result.returncode == 0 and final_path.exists():
        succeeded += 1
    else:
        still_failing.append((ytid, "trim_failed"))

    time.sleep(0.3)

print(f"\nRecovered: {succeeded}")
print(f"Still failing: {len(still_failing)}")