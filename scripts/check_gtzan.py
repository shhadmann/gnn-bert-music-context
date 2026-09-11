from pathlib import Path

root = Path("data/raw/gtzan/genres_original")
genres = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

total = 0
for genre in genres:
    folder = root / genre
    if not folder.exists():
        print(f"MISSING FOLDER: {genre}")
        continue
    count = len(list(folder.glob("*.wav")))
    print(f"{genre}: {count} files")
    total += count

print(f"\nTotal: {total} files (expected 1000)")