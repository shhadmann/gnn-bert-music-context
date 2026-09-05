"""
Dual-encoder contrastive model for Task 4 (MusicCaps retrieval).
GraphEncoder and TextEncoder project into a shared embedding space,
trained with InfoNCE so matching (audio, caption) pairs land close
together and non-matching pairs land far apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=3, embed_dim=256, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.projection = nn.Linear(hidden_channels, embed_dim)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = global_mean_pool(x, batch)
        projected = self.projection(graph_embedding)
        return F.normalize(projected, dim=-1)  # unit-norm, required for cosine similarity in InfoNCE


class TextEncoder(nn.Module):
    def __init__(self, bert_model, embed_dim=256):
        super().__init__()
        self.bert = bert_model  # the underlying HuggingFace BertModel, not our classifier wrapper
        hidden_size = self.bert.config.hidden_size
        self.projection = nn.Linear(hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_embedding)
        return F.normalize(projected, dim=-1)


def info_nce_loss(graph_embeds, text_embeds, temperature=0.07):
    """
    Symmetric InfoNCE: graph->text and text->graph, averaged.
    graph_embeds, text_embeds: (batch, embed_dim), already L2-normalized.
    """
    logits = torch.matmul(graph_embeds, text_embeds.T) / temperature  # (batch, batch)
    batch_size = graph_embeds.shape[0]
    labels = torch.arange(batch_size, device=graph_embeds.device)  # diagonal = correct pairs

    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)
    return (loss_g2t + loss_t2g) / 2