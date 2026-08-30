"""
GraphSAGE-based genre classifier for Task 2.
Operates on segment graphs (nodes=audio segments, from graph_builder.py).
Also includes a CNN baseline on full-track mel-spectrograms for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNGenreClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=3,
                 num_classes=10, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embedding = global_mean_pool(x, batch)  # (num_graphs, hidden_channels)
        logits = self.classifier(graph_embedding)
        return logits, graph_embedding


class CNNBaseline(nn.Module):
    """CNN on full-track log-mel spectrograms — no graph, no text.
    Required baseline (B2) for comparison against the GNN."""
    def __init__(self, num_classes=10, hidden_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, hidden_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        # x: (batch, 1, n_mels, time_frames)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_pool(x).flatten(1)
        logits = self.classifier(x)
        return logits