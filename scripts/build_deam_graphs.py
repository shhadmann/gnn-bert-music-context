"""
Batch-build segment graphs for all DEAM clips.
Each graph gets valence_mean and arousal_mean attached as regression
targets (data.y), not a classification label.
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

features_dir = Path("data/processed/deam/features")
graph_out_dir = Path("data/processed/deam/graphs")
graph_out_dir.mkdir(parents=True, exist_ok=True)

annotations = pd.read_csv("data/processed/deam/annotations_merged.csv").set_index("song_id")

built = 0
skipped_too_few_segments = []
isolated_node_warnings = []

npz_files = sorted(features_dir.glob("*.npz"))
for npz_path in tqdm(npz_files):
    song_id = int(npz_path.stem)
    d = np.load(npz_path)
    segment_features = d["segment_features"]

    try:
        graph = build_segment_graph(segment_features, similarity_threshold=tau)
    except ValueError as e:
        skipped_too_few_segments.append((song_id, str(e)))
        continue

    valence = annotations.loc[song_id, "valence_mean"]
    arousal = annotations.loc[song_id, "arousal_mean"]
    graph.y = torch.tensor([[valence, arousal]], dtype=torch.float)  # shape (1, 2)
    graph.song_id = song_id

    connected_nodes = set(graph.edge_index[0].tolist()) | set(graph.edge_index[1].tolist())
    n_nodes = graph.x.shape[0]
    if len(connected_nodes) < n_nodes:
        isolated_node_warnings.append(song_id)

    out_path = graph_out_dir / f"{song_id}.pt"
    torch.save(graph, out_path)
    built += 1

print(f"\nGraphs built: {built}")
print(f"Skipped (too few segments): {len(skipped_too_few_segments)}")
for sid, err in skipped_too_few_segments:
    print(f"  - {sid}: {err}")
print(f"Graphs with isolated nodes: {len(isolated_node_warnings)}")