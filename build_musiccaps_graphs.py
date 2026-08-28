"""
Batch-build segment graphs for all MusicCaps clips.
Each graph gets the caption text attached (not a label vector) —
Task 4's contrastive training pairs graph embeddings against caption
embeddings directly, no classification target needed here.
"""

import sys
sys.path.append("src")

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from graph_builder import load_config, build_segment_graph
from tqdm import tqdm

config = load_config("config.yaml")
tau = config["graph"]["similarity_threshold"]

features_dir = Path("data/processed/musiccaps/features")
graph_out_dir = Path("data/processed/musiccaps/graphs")
graph_out_dir.mkdir(parents=True, exist_ok=True)

captions = pd.read_csv("data/processed/musiccaps/available_clips.csv").set_index("ytid")

built = 0
skipped_too_few_segments = []
isolated_node_warnings = []

npz_files = sorted(features_dir.glob("*.npz"))
for npz_path in tqdm(npz_files):
    ytid = npz_path.stem
    d = np.load(npz_path)
    segment_features = d["segment_features"]

    try:
        graph = build_segment_graph(segment_features, similarity_threshold=tau)
    except ValueError as e:
        skipped_too_few_segments.append((ytid, str(e)))
        continue

    graph.caption = captions.loc[ytid, "caption"]
    graph.ytid = ytid

    connected_nodes = set(graph.edge_index[0].tolist()) | set(graph.edge_index[1].tolist())
    n_nodes = graph.x.shape[0]
    if len(connected_nodes) < n_nodes:
        isolated_node_warnings.append(ytid)

    out_path = graph_out_dir / f"{ytid}.pt"
    torch.save(graph, out_path)
    built += 1

print(f"\nGraphs built: {built}")
print(f"Skipped (too few segments): {len(skipped_too_few_segments)}")
for yid, err in skipped_too_few_segments:
    print(f"  - {yid}: {err}")
print(f"Graphs with isolated nodes: {len(isolated_node_warnings)}")