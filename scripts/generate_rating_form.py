"""
Generate a plain-text rating form for the human evaluation listeners.
Does not reveal which clip is the actual correct match.
"""

import json

with open("results/human_eval_clips/rating_sheet_reference.json") as f:
    rows = json.load(f)

lines = [
    "MUSIC-CAPTION MATCH RATING FORM",
    "=" * 60,
    "",
    "For each item: read the caption, listen to the numbered audio",
    "file (query_1.mp3, query_2.mp3, etc.), and rate how well the",
    "audio matches the caption on a scale of 1-5:",
    "",
    "  1 = Not at all a match",
    "  2 = Poor match",
    "  3 = Somewhat a match",
    "  4 = Good match",
    "  5 = Excellent match",
    "",
    "Write your rating number next to each item.",
    "=" * 60,
    "",
]

for row in rows:
    lines.append(f"Item {row['query_number']} — file: {row['audio_file']}")
    lines.append(f'Caption: "{row["query_caption"]}"')
    lines.append("Your rating (1-5): ______")
    lines.append("")

with open("results/human_eval_clips/rating_form.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Saved: results/human_eval_clips/rating_form.txt")
print(f"Send the entire 'human_eval_clips' folder (10 mp3s + rating_form.txt) to each of your 5 listeners.")