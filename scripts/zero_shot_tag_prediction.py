"""
Zero-shot tag prediction via embedding similarity: embed each of the
50 MagnaTagATune tags as text, embed each MusicCaps caption, assign
tags by cosine similarity. No hard-coded mapping, genuinely zero-shot
per the spec's wording, using the trained Task 4 text encoder.
"""

import sys
sys.path.append("src")

import json
import torch
import numpy as np
import pandas as pd
import yaml

from contrastive import TextEncoder
from bert_encoder import load_tokenizer, tokenize_batch
from train import MusicCapsDataset
from transformers import BertModel

with open("config.yaml") as f:
    config = yaml.safe_load(f)
device = torch.device("cpu")

with open("data/processed/magnatagatune/top50_tags.json") as f:
    top50_tags = json.load(f)

tokenizer = load_tokenizer(config["bert"]["model_name"])
bert_model = BertModel.from_pretrained(config["bert"]["model_name"])
text_encoder = TextEncoder(bert_model, embed_dim=config["contrastive"]["embed_dim"])
text_encoder.load_state_dict(torch.load("results/contrastive_text_encoder.pt", map_location=device))
text_encoder.eval()

# Embed the 50 tag names themselves as short text queries
with torch.no_grad():
    tag_tokenized = tokenize_batch(tokenizer, top50_tags, max_length=8)
    tag_embeds = text_encoder(tag_tokenized["input_ids"], tag_tokenized["attention_mask"])  # (50, embed_dim)

# --- Sanity check: do the two label spaces overlap semantically at all? ---
# Compare tag embeddings against each other to confirm the embedding space
# itself is sensible (e.g. "guitar" should be closer to "strings" than to "opera")
sim_matrix = (tag_embeds @ tag_embeds.T).numpy()
print("Sanity check — tag-to-tag similarity (should show related tags close):")
for pair in [("guitar", "strings"), ("guitar", "opera"), ("male", "female"), ("fast", "slow")]:
    i, j = top50_tags.index(pair[0]), top50_tags.index(pair[1])
    print(f"  sim({pair[0]}, {pair[1]}) = {sim_matrix[i, j]:.3f}")

# --- Embed all MusicCaps test captions ---
with open("data/splits/musiccaps_test.json") as f:
    test_ids = json.load(f)
test_ds = MusicCapsDataset(test_ids)

captions = [test_ds[i][1] for i in range(len(test_ds))]
ytids = [test_ds[i][2] for i in range(len(test_ds))]

with torch.no_grad():
    cap_tokenized = tokenize_batch(tokenizer, captions, max_length=config["bert"]["max_length"])
    cap_embeds = text_encoder(cap_tokenized["input_ids"], cap_tokenized["attention_mask"])  # (N, embed_dim)

# --- Zero-shot tag assignment: top-k tags per caption by similarity ---
cap_tag_sim = (cap_embeds @ tag_embeds.T).numpy()  # (N, 50)
TOP_K = 5

zero_shot_predictions = []
for i in range(len(captions)):
    top_k_idx = np.argsort(-cap_tag_sim[i])[:TOP_K]
    predicted_tags = [top50_tags[j] for j in top_k_idx]
    zero_shot_predictions.append({
        "ytid": ytids[i],
        "caption": captions[i][:100],
        "zero_shot_predicted_tags": predicted_tags,
    })

print(f"\nGenerated zero-shot predictions for {len(zero_shot_predictions)} captions")
print("\nSample predictions:")
for ex in zero_shot_predictions[:5]:
    print(f"  \"{ex['caption']}\" -> {ex['zero_shot_predicted_tags']}")

with open("results/task4_zero_shot_predictions.json", "w") as f:
    json.dump(zero_shot_predictions, f, indent=2)
print("\nSaved: results/task4_zero_shot_predictions.json")