"""
Generate 3 case studies for Task 3 (cross-attention model). Per the
roadmap's decision gate, no valid caption/lyric source exists between
MagnaTagATune and MusicCaps, so this uses the documented tag-based
fallback: audio -> segment graph -> graph path -> GNN representation
-> predicted/ground-truth tags.
"""

import sys
sys.path.append("src")

import json
import numpy as np
import pandas as pd
import torch
import yaml

from bert_encoder import BertTagClassifier, load_tokenizer, tokenize_batch
from gnn_model import GNNGenreClassifier
from fusion_model import CrossAttentionFusion
from train import MTATFusionDataset, fusion_collate_fn
from torch.utils.data import DataLoader

with open("config.yaml") as f:
    config = yaml.safe_load(f)
device = torch.device("cpu")

with open("data/processed/magnatagatune/top50_tags.json") as f:
    top50_tags = json.load(f)
with open("results/task3_cross_attention_test_metrics.json") as f:
    threshold = json.load(f)["threshold_used"]

text_labels = pd.read_csv("data/processed/magnatagatune/text_labels.csv")
with open("data/splits/mtag_test.json") as f:
    test_ids = json.load(f)

tokenizer = load_tokenizer(config["bert"]["model_name"])
test_ds = MTATFusionDataset(test_ids, text_labels, top50_tags)
collate = lambda b: fusion_collate_fn(b, tokenizer, config["bert"]["max_length"])
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate)  # batch=1 for per-clip inspection

graph_dim = config["gnn"]["hidden_channels"]
in_channels = test_ds[0][0].x.shape[1]

gnn = GNNGenreClassifier(in_channels=in_channels, hidden_channels=graph_dim,
                          num_layers=config["gnn"]["num_layers"], num_classes=10,
                          dropout=config["gnn"]["dropout"])
gnn.load_state_dict(torch.load("results/task3_cross_attention_gnn.pt", map_location=device))
gnn.eval()

bert = BertTagClassifier(config["bert"]["model_name"], num_tags=50)
bert.load_state_dict(torch.load("results/task3_cross_attention_bert.pt", map_location=device))
bert.eval()

fusion = CrossAttentionFusion(graph_dim=graph_dim, bert_dim=768, num_tags=50)
fusion.load_state_dict(torch.load("results/task3_cross_attention_fusion.pt", map_location=device))
fusion.eval()

results = []
clip_ids = test_ds.clip_ids

with torch.no_grad():
    for i, (graph_batch, tokenized, tags) in enumerate(test_loader):
        clip_id = clip_ids[i]
        _, graph_emb = gnn(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        bert_out = bert.bert(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"])
        token_emb = bert_out.last_hidden_state
        logits, z, attn_weights = fusion(graph_emb, token_emb, tokenized["attention_mask"])
        probs = torch.sigmoid(logits)[0]

        true_tags = [top50_tags[j] for j in range(50) if tags[0][j] == 1]
        predicted_tags = [top50_tags[j] for j in range(50) if probs[j] > threshold]
        overlap = set(true_tags) & set(predicted_tags)

        n_segments = graph_batch.x.shape[0]
        edge_count = graph_batch.edge_index.shape[1]

        results.append({
            "clip_id": int(clip_id),
            "text": text_labels[text_labels["clip_id"] == clip_id]["text"].values[0],
            "true_tags": true_tags,
            "predicted_tags": predicted_tags,
            "correct_overlap": list(overlap),
            "n_true": len(true_tags),
            "n_overlap": len(overlap),
            "n_segments": n_segments,
            "n_edges": edge_count,
        })

# Pick 3 diverse, informative examples:
# 1. Strong correct prediction (high overlap ratio, several true tags)
# 2. A partial/interesting mismatch (some overlap, some miss)
# 3. A case with a larger, more complex graph (more segments/edges)
results_with_signal = [r for r in results if r["n_true"] >= 2]
results_with_signal.sort(key=lambda r: r["n_overlap"] / max(r["n_true"], 1), reverse=True)

case1 = results_with_signal[0]  # best overlap ratio
case2 = next(r for r in results_with_signal if 0 < r["n_overlap"] < r["n_true"])  # partial match
# Case 3: a genuinely informative miss — a false positive the model was
# confident about, revealing something about what it's actually learning
# (e.g. confusing acoustically-similar tags) rather than an arbitrary tie-break
# on segment count, which doesn't vary meaningfully across clips (fixed
# 8-second segmentation means most clips have ~4 segments regardless).
results_with_misses = [r for r in results if len(set(r["predicted_tags"]) - set(r["true_tags"])) > 0]
case3 = max(results_with_misses, key=lambda r: len(set(r["predicted_tags"]) - set(r["true_tags"])))

selected = [case1, case2, case3]

# Guard against accidental duplicate clip_ids across the 3 selections
selected_ids = [c["clip_id"] for c in selected]
if len(set(selected_ids)) < 3:
    print(f"WARNING: duplicate clip(s) selected: {selected_ids} — case selection logic needs adjustment")


for i, c in enumerate(selected, 1):
    print(f"\n--- Case Study {i} (clip {c['clip_id']}) ---")
    print(f"Text: \"{c['text']}\"")
    print(f"Graph: {c['n_segments']} segments, {c['n_edges']} edges")
    print(f"True tags: {c['true_tags']}")
    print(f"Predicted tags: {c['predicted_tags']}")
    print(f"Correctly matched: {c['correct_overlap']}")

with open("results/task3_case_studies.json", "w") as f:
    json.dump(selected, f, indent=2)
print("\nSaved: results/task3_case_studies.json")