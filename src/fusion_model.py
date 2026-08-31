"""
GNN-BERT fusion models for Task 3.
CrossAttentionFusion: the spec's required architecture —
  A = softmax(QK^T / sqrt(d)), Q = g W_Q, K = H_text W_K
  z = CONCAT(g, A H_text), yhat = sigma(W z)
EarlyConcatFusion: simpler ablation baseline — no attention, just
  concatenates the pooled graph embedding and BERT's CLS embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, graph_dim, bert_dim, num_tags=50, attn_dim=128):
        super().__init__()
        self.query_proj = nn.Linear(graph_dim, attn_dim)
        self.key_proj = nn.Linear(bert_dim, attn_dim)
        self.value_proj = nn.Linear(bert_dim, attn_dim)
        self.attn_dim = attn_dim
        self.classifier = nn.Linear(graph_dim + attn_dim, num_tags)

    def forward(self, graph_embedding, bert_token_embeddings, attention_mask=None):
        # graph_embedding: (batch, graph_dim)
        # bert_token_embeddings: (batch, seq_len, bert_dim)
        Q = self.query_proj(graph_embedding).unsqueeze(1)      # (batch, 1, attn_dim)
        K = self.key_proj(bert_token_embeddings)                # (batch, seq_len, attn_dim)
        V = self.value_proj(bert_token_embeddings)               # (batch, seq_len, attn_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attn_dim ** 0.5)  # (batch, 1, seq_len)

        if attention_mask is not None:
            mask = (1.0 - attention_mask.unsqueeze(1).float()) * -1e9
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)                # (batch, 1, seq_len)
        context = torch.matmul(attn_weights, V).squeeze(1)      # (batch, attn_dim)

        z = torch.cat([graph_embedding, context], dim=1)        # (batch, graph_dim+attn_dim)
        logits = self.classifier(z)
        return logits, z, attn_weights.squeeze(1)


class EarlyConcatFusion(nn.Module):
    def __init__(self, graph_dim, bert_dim, num_tags=50):
        super().__init__()
        self.classifier = nn.Linear(graph_dim + bert_dim, num_tags)

    def forward(self, graph_embedding, bert_cls_embedding):
        z = torch.cat([graph_embedding, bert_cls_embedding], dim=1)
        logits = self.classifier(z)
        return logits, z