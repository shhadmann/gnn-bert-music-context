import pandas as pd
from pathlib import Path

s1 = pd.read_csv("data/raw/deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv")
s2 = pd.read_csv("data/raw/deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv")

print("File 1 shape:", s1.shape)
print("File 1 columns:", list(s1.columns))
print(s1.head(3))
print()
print("File 2 shape:", s2.shape)
print("File 2 columns:", list(s2.columns))
print(s2.head(3))

print()
audio_dir = Path("data/raw/deam/MEMD_audio")
audio_files = sorted(audio_dir.glob("*"))[:10]
print("First 10 audio files:", [f.name for f in audio_files])
print("Total audio files:", len(list(audio_dir.glob("*"))))