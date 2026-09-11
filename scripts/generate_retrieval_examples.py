"""
Generate 10 qualitative retrieval examples for Task 4: given a query
caption, show the top-3 audio matches retrieved by the trained
contrastive model.
"""

import sys
sys.path.append("src")

import json
import torch
import numpy as np
import yaml
from pathlib import Path

from contrastive import GraphEncoder, TextEncoder
from bert_encoder import load_tokenizer, tokenize_batch
from train import MusicCapsDataset, musiccaps_collate_fn
from torch.utils.data import DataLoader
from transformers import BertModel

with open("config.yaml") as f:
    config = yaml.safe_load(f)
device = torch.device("cpu")

with open("data/splits/musiccaps_test.json") as f:
    test_ids = json.load(f)

tokenizer = load_tokenizer(config["bert"]["model_name"])
test_ds = MusicCapsDataset(test_ids)
collate = lambda b: musiccaps_collate_fn(b, tokenizer, config["bert"]["max_length"])
test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate)

in_channels = test_ds[0][0].x.shape[1]
embed_dim = config["contrastive"]["embed_dim"]

graph_encoder = GraphEncoder(in_channels=in_channels, hidden_channels=config["gnn"]["hidden_channels"],
                              num_layers=config["gnn"]["num_layers"], embed_dim=embed_dim,
                              dropout=config["gnn"]["dropout"])
graph_encoder.load_state_dict(torch.load("results/contrastive_graph_encoder.pt", map_location=device))
graph_encoder.eval()

bert_model = BertModel.from_pretrained(config["bert"]["model_name"])
text_encoder = TextEncoder(bert_model, embed_dim=embed_dim)
text_encoder.load_state_dict(torch.load("results/contrastive_text_encoder.pt", map_location=device))
text_encoder.eval()

# Collect all test embeddings, captions, and ytids
all_graph_embeds, all_text_embeds, all_captions, all_ytids = [], [], [], []

with torch.no_grad():
    for graph_batch, tokenized, ytids in test_loader:
        g = graph_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        t = text_encoder(tokenized["input_ids"], tokenized["attention_mask"])
        all_graph_embeds.append(g)
        all_text_embeds.append(t)
        all_ytids.extend(ytids)

all_graph_embeds = torch.cat(all_graph_embeds)
all_text_embeds = torch.cat(all_text_embeds)

# Get the actual caption text for each ytid, for display
caption_lookup = {}
for ytid in all_ytids:
    graph = torch.load(f"data/processed/musiccaps/graphs/{ytid}.pt", weights_only=False)
    caption_lookup[ytid] = graph.caption

print(f"Total test examples: {len(all_ytids)}")

# Pick 10 diverse query captions (evenly spaced through the test set)
np.random.seed(config["seed"])
query_indices = np.random.choice(len(all_ytids), size=10, replace=False)

results = []
for qi in query_indices:
    query_ytid = all_ytids[qi]
    query_caption = caption_lookup[query_ytid]
    query_embed = all_text_embeds[qi]

    # Similarity of this caption against ALL audio embeddings
    sims = (all_graph_embeds @ query_embed).numpy()
    top3_idx = np.argsort(-sims)[:3]

    correct_rank = int(np.where(np.argsort(-sims) == qi)[0][0]) + 1

    top3 = []
    for idx in top3_idx:
        top3.append({
            "ytid": all_ytids[idx],
            "similarity": float(sims[idx]),
            "is_correct_match": bool(idx == qi),
        })

    results.append({
        "query_caption": query_caption,
        "query_ytid": query_ytid,
        "top3_matches": top3,
        "correct_match_rank": correct_rank,
    })

    print(f"\nQuery: \"{query_caption[:100]}\"")
    print(f"  (true match: {query_ytid}, ranked #{correct_rank} out of {len(all_ytids)})")
    for i, m in enumerate(top3, 1):
        marker = " <-- CORRECT" if m["is_correct_match"] else ""
        print(f"  Top {i}: {m['ytid']} (sim={m['similarity']:.3f}){marker}")

with open("results/task4_retrieval_examples.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/task4_retrieval_examples.json")