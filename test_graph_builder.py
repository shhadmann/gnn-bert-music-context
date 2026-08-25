import sys
sys.path.append("src")

import numpy as np
from graph_builder import load_config, build_segment_graph

config = load_config("config.yaml")
tau = config["graph"]["similarity_threshold"]

d = np.load("data/processed/gtzan/blues/blues.00000.npz")
segment_features = d["segment_features"]

genres = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
genre_to_idx = {g: i for i, g in enumerate(genres)}

graph = build_segment_graph(segment_features, similarity_threshold=tau,
                              genre_label="blues", genre_to_idx=genre_to_idx)

print("Node feature matrix shape (x):", graph.x.shape)
print("Edge index shape:", graph.edge_index.shape)
print("Number of edges:", graph.edge_index.shape[1])
print("Graph label (y):", graph.y)
print("Genre string:", graph.genre_str)