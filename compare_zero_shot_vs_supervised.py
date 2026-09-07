"""
Compare Task 4's zero-shot tag predictions (embedding similarity, no
supervision) against Task 3's supervised cross-attention model, using
the SAME MusicCaps captions as input to Task 3's BERT branch.

Important caveat, stated explicitly: Task 3's BERT was trained on
title+artist+album text (MagnaTagATune), not full natural-language
captions -- so applying it to MusicCaps captions is out-of-distribution
for that model. This comparison is informative but not a clean
apples-to-apples evaluation, and is reported with that limitation
named directly, not silently.
"""

import sys
sys.path.append("src")

import json
import torch
import numpy as np
import yaml

from bert_encoder import BertTagClassifier, load_tokenizer, tokenize_batch
from train import MusicCapsDataset

with open("config.yaml") as f:
    config = yaml.safe_load(f)
device = torch.device("cpu")

with open("data/processed/magnatagatune/top50_tags.json") as f:
    top50_tags = json.load(f)
with open("results/task3_cross_attention_test_metrics.json") as f:
    threshold = json.load(f)["threshold_used"]
with open("results/task4_zero_shot_predictions.json") as f:
    zero_shot_preds = json.load(f)

tokenizer = load_tokenizer(config["bert"]["model_name"])

# Task 3's BERT-only branch (classifier head), applied out-of-distribution to captions
bert = BertTagClassifier(config["bert"]["model_name"], num_tags=50)
bert.load_state_dict(torch.load("results/task3_cross_attention_bert.pt", map_location=device))
bert.eval()

captions = [p["caption"] for p in zero_shot_preds]
zero_shot_tags = [set(p["zero_shot_predicted_tags"]) for p in zero_shot_preds]

with torch.no_grad():
    tokenized = tokenize_batch(tokenizer, captions, max_length=config["bert"]["max_length"])
    logits, _ = bert(tokenized["input_ids"], tokenized["attention_mask"])
    probs = torch.sigmoid(logits)

# NOTE: threshold_used (tuned on MagnaTagATune's title+artist+album text)
# is not usable here -- verified directly that on caption text, minimum
# probability (0.253) already exceeds it (0.20), causing every tag to be
# predicted for every caption. This confirms Task 3's BERT is poorly
# calibrated on this out-of-distribution input, itself a real finding.
# Using top-5 by probability instead, for a comparable-shape (not
# comparable-confidence) comparison against zero-shot's top-5.
TOP_K = 5
supervised_tags = []
for i in range(len(captions)):
    top_k_idx = np.argsort(-probs[i].numpy())[:TOP_K]
    predicted = [top50_tags[j] for j in top_k_idx]
    supervised_tags.append(set(predicted))

# Compare: how often do the two methods agree on at least one tag?
overlap_counts = [len(zero_shot_tags[i] & supervised_tags[i]) for i in range(len(captions))]
any_overlap = sum(1 for c in overlap_counts if c > 0)
avg_zero_shot_tags = np.mean([len(t) for t in zero_shot_tags])
avg_supervised_tags = np.mean([len(t) for t in supervised_tags])

print(f"Total captions compared: {len(captions)}")
print(f"Average tags per caption (zero-shot, fixed top-5): {avg_zero_shot_tags:.2f}")
print(f"Average tags per caption (Task 3 supervised, out-of-distribution): {avg_supervised_tags:.2f}")
print(f"Captions with >=1 tag agreement between methods: {any_overlap}/{len(captions)} ({100*any_overlap/len(captions):.1f}%)")
print(f"Average tag overlap count per caption: {np.mean(overlap_counts):.2f}")

print("\nSample side-by-side comparisons:")
for i in range(5):
    print(f"\nCaption: \"{captions[i]}\"")
    print(f"  Zero-shot (embedding similarity): {sorted(zero_shot_tags[i])}")
    print(f"  Task 3 supervised (out-of-dist.): {sorted(supervised_tags[i])}")
    print(f"  Overlap: {sorted(zero_shot_tags[i] & supervised_tags[i])}")

comparison_results = {
    "n_compared": len(captions),
    "avg_zero_shot_tags_per_caption": float(avg_zero_shot_tags),
    "avg_supervised_tags_per_caption": float(avg_supervised_tags),
    "pct_with_any_overlap": float(100 * any_overlap / len(captions)),
    "avg_overlap_count": float(np.mean(overlap_counts)),
    "caveat": "Task 3's BERT was trained on title+artist+album text (MagnaTagATune), "
              "not natural-language captions -- applying it to MusicCaps captions here "
              "is out-of-distribution for that model, not a clean apples-to-apples test.",
}
with open("results/task4_zero_shot_vs_supervised_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=2)
print("\nSaved: results/task4_zero_shot_vs_supervised_comparison.json")