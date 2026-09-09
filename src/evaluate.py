"""
Consolidated evaluation script for the GNN-BERT music context project.
Reloads every trained checkpoint and re-runs evaluation on the
appropriate test set, confirming results match what's already saved
from training. Produces one master summary file for the report.

Run locally (CPU) -- this is inference only, no training.
"""

import sys
sys.path.append("src")

import json
from pathlib import Path

import torch
import yaml

from train import (
    MTATTextDataset, collate_fn, evaluate, load_tokenizer,
    load_gtzan_graphs, evaluate_gnn, GTZANMelDataset, cnn_collate_fn, evaluate_cnn,
    MTATFusionDataset, fusion_collate_fn, evaluate_fusion,
    load_deam_graphs, evaluate_deam,
    MusicCapsDataset, musiccaps_collate_fn, compute_retrieval_metrics,
)
from bert_encoder import BertTagClassifier
from gnn_model import GNNGenreClassifier, CNNBaseline, GNNEmotionRegressor
from fusion_model import CrossAttentionFusion, EarlyConcatFusion
from contrastive import GraphEncoder, TextEncoder
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from transformers import BertModel
import pandas as pd

with open("config.yaml") as f:
    config = yaml.safe_load(f)
device = torch.device("cpu")

summary = {}


def compare(task_name, recomputed, saved_path):
    """Print recomputed vs originally-saved metrics side by side."""
    with open(saved_path) as f:
        saved = json.load(f)
    print(f"\n=== {task_name} ===")
    print(f"  Recomputed: {recomputed}")
    print(f"  Originally saved: {saved}")
    summary[task_name] = {"recomputed": recomputed, "originally_saved": saved}


# ---------------- Task 1: BERT ----------------
print("Evaluating Task 1 (BERT)...")
with open("data/processed/magnatagatune/top50_tags.json") as f:
    top50_tags = json.load(f)
text_labels = pd.read_csv("data/processed/magnatagatune/text_labels.csv")
with open("data/splits/mtag_test.json") as f:
    mtag_test_ids = json.load(f)

tokenizer = load_tokenizer(config["bert"]["model_name"])
bert = BertTagClassifier(config["bert"]["model_name"], num_tags=50)
bert.load_state_dict(torch.load("results/bert_best.pt", map_location=device))
bert.eval()

with open("results/task1_test_metrics.json") as f:
    task1_threshold = json.load(f)["threshold_used"]

test_ds1 = MTATTextDataset(text_labels, mtag_test_ids, top50_tags)
collate1 = lambda b: collate_fn(b, tokenizer, config["bert"]["max_length"])
test_loader1 = DataLoader(test_ds1, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate1)
task1_metrics = evaluate(bert, test_loader1, device, threshold=task1_threshold)
compare("task1_bert", task1_metrics, "results/task1_test_metrics.json")


# ---------------- Task 2: GNN + CNN ----------------
print("\nEvaluating Task 2 (GNN)...")
test_graphs2 = load_gtzan_graphs("data/splits/gtzan_test.json")
test_loader2 = GeoDataLoader(test_graphs2, batch_size=config["training"]["batch_size"], shuffle=False)
in_channels2 = test_graphs2[0].x.shape[1]
gnn2 = GNNGenreClassifier(in_channels=in_channels2, hidden_channels=config["gnn"]["hidden_channels"],
                            num_layers=config["gnn"]["num_layers"], num_classes=10, dropout=config["gnn"]["dropout"])
gnn2.load_state_dict(torch.load("results/gnn_gtzan_best.pt", map_location=device))
gnn2.eval()
gnn2_metrics = evaluate_gnn(gnn2, test_loader2, device)
compare("task2_gnn", gnn2_metrics, "results/task2_gnn_test_metrics.json")

print("Evaluating Task 2 (CNN baseline)...")
test_ds_cnn = GTZANMelDataset("data/splits/gtzan_test.json")
test_loader_cnn = DataLoader(test_ds_cnn, batch_size=16, shuffle=False, collate_fn=cnn_collate_fn)
cnn = CNNBaseline(num_classes=10, hidden_channels=config["cnn_baseline"]["hidden_channels"])
cnn.load_state_dict(torch.load("results/cnn_baseline_best.pt", map_location=device))
cnn.eval()
cnn_metrics = evaluate_cnn(cnn, test_loader_cnn, device)
compare("task2_cnn", cnn_metrics, "results/task2_cnn_test_metrics.json")


# ---------------- Task 3, Stage A: Fusion ablations ----------------
print("\nEvaluating Task 3 ablations...")
with open("data/splits/mtag_train.json") as f:
    mtag_train_ids = json.load(f)  # not used directly here, but MTATFusionDataset needs consistent construction

test_ds3 = MTATFusionDataset(mtag_test_ids, text_labels, top50_tags)
collate3 = lambda b: fusion_collate_fn(b, tokenizer, config["bert"]["max_length"])
test_loader3 = DataLoader(test_ds3, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate3)
graph_dim = config["gnn"]["hidden_channels"]
in_channels3 = test_ds3[0][0].x.shape[1]

for mode in ["bert_only", "gnn_only", "early_concat", "cross_attention"]:
    print(f"  Evaluating {mode}...")
    gnn3 = GNNGenreClassifier(in_channels=in_channels3, hidden_channels=graph_dim,
                                num_layers=config["gnn"]["num_layers"], num_classes=10,
                                dropout=config["gnn"]["dropout"])
    bert3 = BertTagClassifier(config["bert"]["model_name"], num_tags=50)

    with open(f"results/task3_{mode}_test_metrics.json") as f:
        mode_threshold = json.load(f)["threshold_used"]

    if mode == "bert_only":
        bert3.load_state_dict(torch.load("results/bert_best.pt", map_location=device))
        fusion3 = None
    else:
        gnn3.load_state_dict(torch.load(f"results/task3_{mode}_gnn.pt", map_location=device))
        bert3.load_state_dict(torch.load(f"results/task3_{mode}_bert.pt", map_location=device))
        if mode == "gnn_only":
            fusion3 = torch.nn.Linear(graph_dim, 50)
        elif mode == "early_concat":
            fusion3 = EarlyConcatFusion(graph_dim=graph_dim, bert_dim=768, num_tags=50)
        elif mode == "cross_attention":
            fusion3 = CrossAttentionFusion(graph_dim=graph_dim, bert_dim=768, num_tags=50)
        fusion3.load_state_dict(torch.load(f"results/task3_{mode}_fusion.pt", map_location=device))
        fusion3.eval()

    gnn3.eval()
    bert3.eval()
    mode_metrics = evaluate_fusion(gnn3, bert3, fusion3, test_loader3, device, mode, threshold=mode_threshold)
    compare(f"task3_{mode}", mode_metrics, f"results/task3_{mode}_test_metrics.json")


# ---------------- Task 3, Stage B: DEAM ----------------
print("\nEvaluating Task 3 Stage B (DEAM)...")
test_graphs_deam = load_deam_graphs("data/splits/deam_test_extension.json")
test_loader_deam = GeoDataLoader(test_graphs_deam, batch_size=config["training"]["batch_size"], shuffle=False)
in_channels_deam = test_graphs_deam[0].x.shape[1]
deam_model = GNNEmotionRegressor(in_channels=in_channels_deam, hidden_channels=config["gnn"]["hidden_channels"],
                                   num_layers=config["gnn"]["num_layers"], dropout=config["gnn"]["dropout"])
deam_model.load_state_dict(torch.load("results/deam_gnn_best.pt", map_location=device))
deam_model.eval()
deam_metrics = evaluate_deam(deam_model, test_loader_deam, device)
compare("task3_deam", deam_metrics, "results/task3_deam_test_metrics.json")


# ---------------- Task 4: Contrastive retrieval ----------------
print("\nEvaluating Task 4 (contrastive retrieval)...")
with open("data/splits/musiccaps_test.json") as f:
    musiccaps_test_ids = json.load(f)
test_ds4 = MusicCapsDataset(musiccaps_test_ids)
collate4 = lambda b: musiccaps_collate_fn(b, tokenizer, config["bert"]["max_length"])
test_loader4 = DataLoader(test_ds4, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate4)

in_channels4 = test_ds4[0][0].x.shape[1]
embed_dim = config["contrastive"]["embed_dim"]
graph_encoder4 = GraphEncoder(in_channels=in_channels4, hidden_channels=config["gnn"]["hidden_channels"],
                                num_layers=config["gnn"]["num_layers"], embed_dim=embed_dim,
                                dropout=config["gnn"]["dropout"])
graph_encoder4.load_state_dict(torch.load("results/contrastive_graph_encoder.pt", map_location=device))
graph_encoder4.eval()

bert_model4 = BertModel.from_pretrained(config["bert"]["model_name"])
text_encoder4 = TextEncoder(bert_model4, embed_dim=embed_dim)
text_encoder4.load_state_dict(torch.load("results/contrastive_text_encoder.pt", map_location=device))
text_encoder4.eval()

all_g4, all_t4 = [], []
with torch.no_grad():
    for graph_batch, tokenized, _ in test_loader4:
        g = graph_encoder4(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        t = text_encoder4(tokenized["input_ids"], tokenized["attention_mask"])
        all_g4.append(g)
        all_t4.append(t)
all_g4 = torch.cat(all_g4)
all_t4 = torch.cat(all_t4)
task4_metrics = compute_retrieval_metrics(all_g4, all_t4)
compare("task4_contrastive", task4_metrics, "results/task4_test_metrics.json")


# ---------------- Save consolidated summary ----------------
Path("results").mkdir(exist_ok=True)
with open("results/final_summary_all_tasks.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n\nSaved consolidated summary: results/final_summary_all_tasks.json")
print("Review the recomputed vs originally-saved numbers above -- they should match closely")
print("(tiny floating-point differences are normal; large discrepancies would need investigation).")