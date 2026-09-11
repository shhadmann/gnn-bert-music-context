import subprocess
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/raw/musiccaps/musiccaps-public.csv")
row = df.iloc[0]

ytid = row["ytid"]
start = row["start_s"]
end = row["end_s"]

print(f"Testing download: ytid={ytid}, start={start}s, end={end}s")

url = f"https://www.youtube.com/watch?v={ytid}"

# Step 1: download with yt-dlp, forcing precise keyframe cuts
raw_path = "test_clip_raw.mp3"
cmd = [
    "yt-dlp",
    "-x", "--audio-format", "mp3",
    "--download-sections", f"*{start}-{end}",
    "--force-keyframes-at-cuts",
    "-o", raw_path,
    url,
]
result = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", result.returncode)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1500:])

# Step 2: independently re-trim with ffmpeg to guarantee exact duration,
# regardless of what yt-dlp actually grabbed
final_path = "test_clip_final.mp3"
duration = end - start
trim_cmd = [
    "ffmpeg", "-y",
    "-i", raw_path,
    "-t", str(duration),
    final_path,
]
trim_result = subprocess.run(trim_cmd, capture_output=True, text=True)
print("Trim return code:", trim_result.returncode)