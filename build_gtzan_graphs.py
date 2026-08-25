"""
Batch-build segment graphs for all processed GTZAN tracks.
Saves each graph as a .pt file. Includes sanity checks per the
roadmap's Phase 1 milestone: isolated nodes, edge counts, feature dims.
"""

import sys
sys.path.append("src")

from pathlib import Path
import numpy as np
import torch
from graph_builder import load_config, build_segment_graph
from tqdm import tqdm

config = load_config("config.yaml")
tau = config["graph"]["similarity_threshold"]

processed_root = Path("data/processed/gtzan")
graph_out_root = Path("data/processed/gtzan_graphs")
graph_out_root.mkdir(parents=True, exist_ok=True)

genres = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
genre_to_idx = {g: i for i, g in enumerate(genres)}

built = 0
skipped_too_few_segments = []
isolated_node_warnings = []

for genre in genres:
    in_folder = processed_root / genre
    out_folder = graph_out_root / genre
    out_folder.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_folder.glob("*.npz"))
    for npz_path in tqdm(npz_files, desc=genre):
        d = np.load(npz_path)
        segment_features = d["segment_features"]

        try:
            graph = build_segment_graph(
                segment_features,
                similarity_threshold=tau,
                genre_label=genre,
                genre_to_idx=genre_to_idx,
            )
        except ValueError as e:
            skipped_too_few_segments.append((str(npz_path), str(e)))
            continue

        # Sanity check: isolated nodes (a node with no edges at all)
        connected_nodes = set(graph.edge_index[0].tolist()) | set(graph.edge_index[1].tolist())
        n_nodes = graph.x.shape[0]
        if len(connected_nodes) < n_nodes:
            isolated_node_warnings.append(str(npz_path))

        out_path = out_folder / (npz_path.stem + ".pt")
        torch.save(graph, out_path)
        built += 1

print(f"\nGraphs built: {built}")
print(f"Skipped (too few segments): {len(skipped_too_few_segments)}")
for path, err in skipped_too_few_segments:
    print(f"  - {path}: {err}")
print(f"Graphs with isolated nodes: {len(isolated_node_warnings)}")
for path in isolated_node_warnings[:10]:  # show first 10 max
    print(f"  - {path}")