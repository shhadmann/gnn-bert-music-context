"""
Batch-build segment graphs for all successfully processed MagnaTagATune clips.
Unlike GTZAN's single-genre label, each graph gets a 50-dim multi-hot tag
vector as its label (data.y), since MagnaTagATune tagging is multi-label.
"""

import sys
sys.path.append("src")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from graph_builder import load_config, build_segment_graph
from tqdm import tqdm

config = load_config("config.yaml")
tau = config["graph"]["similarity_threshold"]

features_dir = Path("data/processed/magnatagatune/features")
graph_out_dir = Path("data/processed/magnatagatune/graphs")
graph_out_dir.mkdir(parents=True, exist_ok=True)

# Load tag annotations to build multi-hot label vectors
annotations = pd.read_csv("data/processed/magnatagatune/filtered_annotations.csv")
with open("data/processed/magnatagatune/top50_tags.json") as f:
    top50_tags = json.load(f)

annotations = annotations.set_index("clip_id")

built = 0
skipped_too_few_segments = []
skipped_no_annotation = []
isolated_node_warnings = []

npz_files = sorted(features_dir.glob("*.npz"))
for npz_path in tqdm(npz_files):
    clip_id = int(npz_path.stem)

    if clip_id not in annotations.index:
        skipped_no_annotation.append(clip_id)
        continue

    d = np.load(npz_path)
    segment_features = d["segment_features"]

    try:
        graph = build_segment_graph(segment_features, similarity_threshold=tau)
    except ValueError as e:
        skipped_too_few_segments.append((clip_id, str(e)))
        continue

    # Attach multi-hot tag label (50-dim), not a single genre index
    tag_row = annotations.loc[clip_id, top50_tags]
    graph.y = torch.tensor(tag_row.values.astype(np.float32)).unsqueeze(0)  # shape (1, 50)
    graph.clip_id = clip_id

    # Sanity check: isolated nodes
    connected_nodes = set(graph.edge_index[0].tolist()) | set(graph.edge_index[1].tolist())
    n_nodes = graph.x.shape[0]
    if len(connected_nodes) < n_nodes:
        isolated_node_warnings.append(clip_id)

    out_path = graph_out_dir / f"{clip_id}.pt"
    torch.save(graph, out_path)
    built += 1

print(f"\nGraphs built: {built}")
print(f"Skipped (too few segments): {len(skipped_too_few_segments)}")
for cid, err in skipped_too_few_segments[:10]:
    print(f"  - {cid}: {err}")
print(f"Skipped (no annotation match): {len(skipped_no_annotation)}")
print(f"Graphs with isolated nodes: {len(isolated_node_warnings)}")
for cid in isolated_node_warnings[:10]:
    print(f"  - {cid}")