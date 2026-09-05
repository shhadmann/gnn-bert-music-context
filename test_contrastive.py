import sys
sys.path.append("src")

import torch
import json
from torch_geometric.loader import DataLoader
from contrastive import GraphEncoder, TextEncoder, info_nce_loss
from bert_encoder import load_tokenizer, tokenize_batch
from transformers import BertModel

# Load 4 real MusicCaps graphs (each has .caption attached, per build_musiccaps_graphs.py)
with open("data/splits/musiccaps_train.json") as f:
    train_ids = json.load(f)

graphs = [torch.load(f"data/processed/musiccaps/graphs/{yid}.pt", weights_only=False) for yid in train_ids[:4]]
captions = [g.caption for g in graphs]

loader = DataLoader(graphs, batch_size=4)
batch = next(iter(loader))

graph_encoder = GraphEncoder(in_channels=graphs[0].x.shape[1], embed_dim=256)
graph_encoder.eval()
with torch.no_grad():
    graph_embeds = graph_encoder(batch.x, batch.edge_index, batch.batch)
print("Graph embeddings shape:", graph_embeds.shape)
print("Graph embeddings norm (should be ~1.0):", graph_embeds.norm(dim=-1))

tokenizer = load_tokenizer()
bert_model = BertModel.from_pretrained("bert-base-uncased")
text_encoder = TextEncoder(bert_model, embed_dim=256)
text_encoder.eval()

tokenized = tokenize_batch(tokenizer, captions)
with torch.no_grad():
    text_embeds = text_encoder(tokenized["input_ids"], tokenized["attention_mask"])
print("Text embeddings shape:", text_embeds.shape)

loss = info_nce_loss(graph_embeds, text_embeds, temperature=0.07)
print("InfoNCE loss (untrained, should be a real positive number):", loss.item())

print("\nSample captions used:")
for c in captions:
    print(" -", c[:80])