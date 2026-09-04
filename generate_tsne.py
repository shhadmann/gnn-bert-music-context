"""
Generate t-SNE of Task 3's fused embeddings (cross-attention model),
colored by the frozen genre/mood mapping decided before any embeddings
were generated (see build_genre_mood_mapping.py).
"""

import sys
sys.path.append("src")

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

from bert_encoder import BertTagClassifier, load_tokenizer, tokenize_batch
from gnn_model import GNNGenreClassifier
from fusion_model import CrossAttentionFusion
from train import MTATFusionDataset, fusion_collate_fn
from torch.utils.data import DataLoader
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cpu")

with open("data/processed/magnatagatune/top50_tags.json") as f:
    top50_tags = json.load(f)
with open("data/processed/magnatagatune/genre_mood_mapping.json") as f:
    mapping = json.load(f)
genre_tags = mapping["genre_tags"]
mood_tags = mapping["mood_tags"]

text_labels = pd.read_csv("data/processed/magnatagatune/text_labels.csv")
with open("data/splits/mtag_test.json") as f:
    test_ids = json.load(f)

tokenizer = load_tokenizer(config["bert"]["model_name"])
test_ds = MTATFusionDataset(test_ids, text_labels, top50_tags)
collate = lambda b: fusion_collate_fn(b, tokenizer, config["bert"]["max_length"])
test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate)

graph_dim = config["gnn"]["hidden_channels"]
in_channels = test_ds[0][0].x.shape[1]

gnn = GNNGenreClassifier(in_channels=in_channels, hidden_channels=graph_dim,
                          num_layers=config["gnn"]["num_layers"], num_classes=10,
                          dropout=config["gnn"]["dropout"])
gnn.load_state_dict(torch.load("results/task3_cross_attention_gnn.pt", map_location=device))
gnn.eval()

bert = BertTagClassifier(config["bert"]["model_name"], num_tags=50)
bert.load_state_dict(torch.load("results/task3_cross_attention_bert.pt", map_location=device))
bert.eval()

fusion = CrossAttentionFusion(graph_dim=graph_dim, bert_dim=768, num_tags=50)
fusion.load_state_dict(torch.load("results/task3_cross_attention_fusion.pt", map_location=device))
fusion.eval()

all_z = []
all_clip_ids = []
all_tags = []

with torch.no_grad():
    for graph_batch, tokenized, tags in test_loader:
        _, graph_emb = gnn(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        bert_out = bert.bert(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"])
        token_emb = bert_out.last_hidden_state
        _, z, _ = fusion(graph_emb, token_emb, tokenized["attention_mask"])
        all_z.append(z.numpy())
        all_tags.append(tags.numpy())

all_z = np.concatenate(all_z)
all_tags = np.concatenate(all_tags)
print(f"Total embeddings: {all_z.shape}")

# Apply the FROZEN color-assignment rule (decided before seeing any of this)
def assign_color(tag_row, category_tags, top50_tags):
    for tag in category_tags:  # fixed order = deterministic tie-breaking
        idx = top50_tags.index(tag)
        if tag_row[idx] == 1:
            return tag
    return "unlabeled"

genre_labels = [assign_color(row, genre_tags, top50_tags) for row in all_tags]
mood_labels = [assign_color(row, mood_tags, top50_tags) for row in all_tags]

print("Genre label counts:", pd.Series(genre_labels).value_counts().to_dict())
print("Mood label counts:", pd.Series(mood_labels).value_counts().to_dict())

# Run t-SNE once, use for both plots
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=config["seed"], perplexity=30)
z_2d = tsne.fit_transform(all_z)

def plot_tsne(z_2d, color_labels, title, out_path, unique_labels):
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))

    for label in unique_labels:
        mask = np.array(color_labels) == label
        if label == "unlabeled":
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c="lightgray", s=8, alpha=0.3, label=label)
        else:
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=[color_map[label]], s=15, alpha=0.7, label=label)

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

Path("results/plots").mkdir(parents=True, exist_ok=True)

genre_unique = ["unlabeled"] + genre_tags
plot_tsne(z_2d, genre_labels, "Task 3 t-SNE: Fused Embeddings Colored by Genre",
          "results/plots/task3_tsne_genre.png", genre_unique)

mood_unique = ["unlabeled"] + mood_tags
plot_tsne(z_2d, mood_labels, "Task 3 t-SNE: Fused Embeddings Colored by Mood",
          "results/plots/task3_tsne_mood.png", mood_unique)