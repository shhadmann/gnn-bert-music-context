import sys
sys.path.append("src")

import torch
from torch_geometric.loader import DataLoader
from gnn_model import GNNGenreClassifier, CNNBaseline

# --- Test GNN on a couple of real GTZAN graphs ---
g1 = torch.load("data/processed/gtzan_graphs/blues/blues.00000.pt", weights_only=False)
g2 = torch.load("data/processed/gtzan_graphs/rock/rock.00000.pt", weights_only=False)

loader = DataLoader([g1, g2], batch_size=2)
batch = next(iter(loader))

gnn = GNNGenreClassifier(in_channels=g1.x.shape[1], num_classes=10)
gnn.eval()
with torch.no_grad():
    logits, emb = gnn(batch.x, batch.edge_index, batch.batch)

print("GNN logits shape:", logits.shape)
print("GNN graph embedding shape:", emb.shape)

# --- Test CNN on real log-mel features ---
import numpy as np
d = np.load("data/processed/gtzan/blues/blues.00000.npz")
mel = torch.tensor(d["full_log_mel"], dtype=torch.float).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)

cnn = CNNBaseline(num_classes=10)
cnn.eval()
with torch.no_grad():
    cnn_logits = cnn(mel)

print("CNN input shape:", mel.shape)
print("CNN logits shape:", cnn_logits.shape)