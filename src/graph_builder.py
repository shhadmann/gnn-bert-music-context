"""
Graph construction for the GNN-BERT music context project.
Builds segment graphs (the required core representation per the roadmap)
from precomputed audio segment features. Nodes = audio segments,
edges = temporal adjacency + cosine similarity above threshold tau.

Chord-transition graphs are an optional extension and are not
implemented here.
"""

import numpy as np
import torch
from torch_geometric.data import Data
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def cosine_similarity_matrix(features):
    """features: (n_segments, feature_dim). Returns (n_segments, n_segments) matrix."""
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1e-8  # avoid division by zero
    normalized = features / norm
    return normalized @ normalized.T


def build_segment_graph(segment_features, similarity_threshold=0.8,
                          genre_label=None, genre_to_idx=None):
    """
    Build a single segment graph from one track's segment features.

    segment_features: np.ndarray, shape (n_segments, feature_dim)
    similarity_threshold: tau — edges added when cosine similarity exceeds this
    genre_label / genre_to_idx: optional, attaches a graph-level label (for Task 2)

    Returns: torch_geometric.data.Data
    """
    n_segments = segment_features.shape[0]
    if n_segments < 2:
        raise ValueError(f"Need at least 2 segments to build a graph, got {n_segments}")

    edges = set()

    # Temporal adjacency (undirected -> both directions)
    for i in range(n_segments - 1):
        edges.add((i, i + 1))
        edges.add((i + 1, i))

    # Similarity edges (tau-thresholded cosine similarity)
    sim_matrix = cosine_similarity_matrix(segment_features)
    for i in range(n_segments):
        for j in range(n_segments):
            if i != j and sim_matrix[i, j] > similarity_threshold:
                edges.add((i, j))

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()  # (2, num_edges)
    x = torch.tensor(segment_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    if genre_label is not None and genre_to_idx is not None:
        data.y = torch.tensor([genre_to_idx[genre_label]], dtype=torch.long)
        data.genre_str = genre_label  # human-readable, useful later for case studies

    return data