import sys
sys.path.append("src")

import torch
import pandas as pd
from torch_geometric.loader import DataLoader

from gnn_model import GNNGenreClassifier
from bert_encoder import BertTagClassifier, load_tokenizer, tokenize_batch
from fusion_model import CrossAttentionFusion, EarlyConcatFusion

# Real MagnaTagATune graphs (clip_id 2 and 6 — the first ones we ever inspected)
g1 = torch.load("data/processed/magnatagatune/graphs/2.pt", weights_only=False)
g2 = torch.load("data/processed/magnatagatune/graphs/6.pt", weights_only=False)

loader = DataLoader([g1, g2], batch_size=2)
batch = next(iter(loader))

gnn = GNNGenreClassifier(in_channels=g1.x.shape[1], num_classes=10)  # untrained, just testing shapes
gnn.eval()
with torch.no_grad():
    _, graph_emb = gnn(batch.x, batch.edge_index, batch.batch)
print("Graph embedding shape:", graph_emb.shape)

# Matching real text for the same two clips
text_labels = pd.read_csv("data/processed/magnatagatune/text_labels.csv")
texts = text_labels[text_labels["clip_id"].isin([2, 6])]["text"].tolist()

tokenizer = load_tokenizer()
bert_model = BertTagClassifier(num_tags=50)
bert_model.eval()

batch_tok = tokenize_batch(tokenizer, texts)
with torch.no_grad():
    outputs = bert_model.bert(input_ids=batch_tok["input_ids"], attention_mask=batch_tok["attention_mask"])
    token_embeddings = outputs.last_hidden_state
    cls_embedding = token_embeddings[:, 0, :]
print("Token embeddings shape:", token_embeddings.shape)

# Cross-attention fusion
fusion = CrossAttentionFusion(graph_dim=128, bert_dim=768, num_tags=50)
fusion.eval()
with torch.no_grad():
    logits, z, attn = fusion(graph_emb, token_embeddings, batch_tok["attention_mask"])
print("Cross-attention logits shape:", logits.shape)
print("Cross-attention z shape:", z.shape)
print("Attention weights shape:", attn.shape)

# Early concat fusion
early = EarlyConcatFusion(graph_dim=128, bert_dim=768, num_tags=50)
early.eval()
with torch.no_grad():
    logits2, z2 = early(graph_emb, cls_embedding)
print("Early concat logits shape:", logits2.shape)