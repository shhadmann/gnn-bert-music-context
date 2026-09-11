"""
Prepare materials for the 5-listener human evaluation: pulls the top-1
retrieved audio clip for each of the 10 queries, copies the mp3 files
to a shareable folder, and generates a rating sheet.
"""

import json
import shutil
from pathlib import Path

with open("results/task4_retrieval_examples.json") as f:
    examples = json.load(f)

out_dir = Path("results/human_eval_clips")
out_dir.mkdir(parents=True, exist_ok=True)

rating_sheet_rows = []

for i, ex in enumerate(examples, 1):
    top1_ytid = ex["top3_matches"][0]["ytid"]
    src_path = Path("data/raw/musiccaps/audio") / f"{top1_ytid}.mp3"
    dst_path = out_dir / f"query_{i}.mp3"

    if src_path.exists():
        shutil.copy(src_path, dst_path)
        status = "copied"
    else:
        status = "MISSING SOURCE FILE"

    rating_sheet_rows.append({
        "query_number": i,
        "query_caption": ex["query_caption"],
        "audio_file": f"query_{i}.mp3",
        "retrieved_ytid": top1_ytid,
        "was_actually_correct_match": ex["top3_matches"][0]["is_correct_match"],
        "status": status,
    })

    print(f"Query {i}: {status} -> query_{i}.mp3 (retrieved: {top1_ytid}, "
          f"actually correct: {ex['top3_matches'][0]['is_correct_match']})")

with open(out_dir / "rating_sheet_reference.json", "w") as f:
    json.dump(rating_sheet_rows, f, indent=2)

print(f"\nAll clips + reference sheet saved to: {out_dir}")