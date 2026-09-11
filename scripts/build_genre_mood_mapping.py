"""
Frozen genre/mood tag categorization for Task 3's t-SNE, decided and
saved BEFORE any embeddings are generated, per the roadmap's requirement
that this mapping not be chosen post-hoc from observed cluster structure.
"""

import json

GENRE_TAGS = ["classical", "techno", "rock", "opera", "indian", "pop",
              "new age", "dance", "country", "metal", "classic"]

MOOD_TAGS = ["slow", "fast", "ambient", "loud", "quiet", "soft", "weird"]

with open("data/processed/magnatagatune/genre_mood_mapping.json", "w") as f:
    json.dump({"genre_tags": GENRE_TAGS, "mood_tags": MOOD_TAGS}, f, indent=2)

print("Genre tags:", GENRE_TAGS)
print("Mood tags:", MOOD_TAGS)
print("Saved: data/processed/magnatagatune/genre_mood_mapping.json")