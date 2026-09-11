"""
Aggregate the 5-listener human evaluation ratings and compare against
which items' top-1 retrieval was actually the correct match.
"""

import json
import numpy as np

ratings = {
    1: [2, 3, 2, 1, 3],
    2: [1, 1, 2, 1, 2],
    3: [5, 4, 5, 4, 5],
    4: [4, 4, 3, 5, 4],
    5: [4, 5, 4, 3, 4],
    6: [4, 3, 5, 4, 3],
    7: [3, 3, 2, 4, 3],
    8: [5, 5, 4, 5, 4],
    9: [1, 2, 1, 2, 1],
    10: [5, 4, 5, 4, 5],
}

with open("results/human_eval_clips/rating_sheet_reference.json") as f:
    reference = json.load(f)

per_item_avg = {}
all_scores = []
for item_num, scores in ratings.items():
    avg = np.mean(scores)
    per_item_avg[item_num] = avg
    all_scores.extend(scores)

overall_mean = np.mean(all_scores)
overall_std = np.std(all_scores)

print("Per-item average rating (1-5 scale):")
for r in reference:
    item_num = r["query_number"]
    was_correct = r["was_actually_correct_match"]
    print(f"  Item {item_num}: avg={per_item_avg[item_num]:.2f}  "
          f"(top-1 retrieval was actual correct match: {was_correct})")

print(f"\nOverall mean rating across all 50 ratings (10 items x 5 listeners): {overall_mean:.2f}")
print(f"Overall std: {overall_std:.2f}")

# Compare: items where top-1 WAS the correct match vs items where it wasn't
correct_items = [r["query_number"] for r in reference if r["was_actually_correct_match"]]
incorrect_items = [r["query_number"] for r in reference if not r["was_actually_correct_match"]]

if correct_items:
    correct_avg = np.mean([per_item_avg[i] for i in correct_items])
    print(f"\nAvg rating for items where top-1 WAS correct: {correct_avg:.2f} (n={len(correct_items)})")
if incorrect_items:
    incorrect_avg = np.mean([per_item_avg[i] for i in incorrect_items])
    print(f"Avg rating for items where top-1 was NOT correct: {incorrect_avg:.2f} (n={len(incorrect_items)})")

results = {
    "per_item_average": {str(k): float(v) for k, v in per_item_avg.items()},
    "overall_mean": float(overall_mean),
    "overall_std": float(overall_std),
    "raw_ratings": ratings,
    "n_listeners": 5,
    "n_items": 10,
}
with open("results/task4_human_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/task4_human_eval_results.json")