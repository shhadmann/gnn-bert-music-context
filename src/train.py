"""
Training script for the GNN-BERT music context project.
Task 1: BERT multi-label tag classifier (MagnaTagATune).
Task 2: GNN + CNN baseline genre classifier (GTZAN).
Task 3, Stage A: GNN-BERT fusion + 4-way ablation suite (MagnaTagATune).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, average_precision_score as ap_score
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from bert_encoder import BertTagClassifier, load_tokenizer, tokenize_batch
from gnn_model import GNNGenreClassifier, CNNBaseline
from fusion_model import CrossAttentionFusion, EarlyConcatFusion


# ============================================================
# Task 1: BERT tag classifier on MagnaTagATune
# ============================================================

class MTATTextDataset(Dataset):
    def __init__(self, text_labels_df, clip_ids, top50_tags):
        self.data = text_labels_df[text_labels_df["clip_id"].isin(clip_ids)].reset_index(drop=True)
        self.top50_tags = top50_tags

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        tags = row[self.top50_tags].values.astype(np.float32)
        return text, tags


def collate_fn(batch, tokenizer, max_length):
    texts = [b[0] for b in batch]
    tags = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float)
    tokenized = tokenize_batch(tokenizer, texts, max_length=max_length)
    return tokenized, tags


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for tokenized, tags in loader:
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            logits, _ = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(tags.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    binary_preds = (all_preds > threshold).astype(int)
    macro_f1 = f1_score(all_labels, binary_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, binary_preds, average="micro", zero_division=0)

    aucpr_scores = []
    for i in range(all_labels.shape[1]):
        if all_labels[:, i].sum() > 0:
            aucpr_scores.append(ap_score(all_labels[:, i], all_preds[:, i]))
    mean_aucpr = float(np.mean(aucpr_scores)) if aucpr_scores else 0.0

    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "aucpr": mean_aucpr}


def find_best_threshold(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for tokenized, tags in loader:
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            logits, _ = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(tags.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    best_threshold, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.05):
        binary_preds = (all_preds > t).astype(int)
        f1 = f1_score(all_labels, binary_preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    print(f"Best threshold found on validation set: {best_threshold:.2f} (macro_f1={best_f1:.4f})")
    return best_threshold


def train_task1(config_path="config.yaml", max_samples=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("data/processed/magnatagatune/top50_tags.json") as f:
        top50_tags = json.load(f)

    text_labels = pd.read_csv("data/processed/magnatagatune/text_labels.csv")

    with open("data/splits/mtag_train.json") as f:
        train_ids = json.load(f)
    with open("data/splits/mtag_val.json") as f:
        val_ids = json.load(f)
    with open("data/splits/mtag_test.json") as f:
        test_ids = json.load(f)

    if max_samples:
        train_ids = train_ids[:max_samples]
        val_ids = val_ids[:max_samples // 4]

    tokenizer = load_tokenizer(config["bert"]["model_name"])
    max_length = config["bert"]["max_length"]

    train_ds = MTATTextDataset(text_labels, train_ids, top50_tags)
    val_ds = MTATTextDataset(text_labels, val_ids, top50_tags)
    test_ds = MTATTextDataset(text_labels, test_ids, top50_tags)

    collate = lambda b: collate_fn(b, tokenizer, max_length)
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = BertTagClassifier(config["bert"]["model_name"], num_tags=50).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["bert"]["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    epochs = 5
    best_val_f1 = -1
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for tokenized, tags in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            tags = tags.to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, tags)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_macro_f1={val_metrics['macro_f1']:.4f}, "
              f"val_micro_f1={val_metrics['micro_f1']:.4f}, val_aucpr={val_metrics['aucpr']:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, **val_metrics})

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            Path("results").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "results/bert_best.pt")
            print(f"  -> New best model saved (val_macro_f1={best_val_f1:.4f})")

    model.load_state_dict(torch.load("results/bert_best.pt"))
    best_threshold = find_best_threshold(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device, threshold=best_threshold)
    test_metrics["threshold_used"] = best_threshold
    print(f"\nFinal test metrics (threshold={best_threshold:.2f}): {test_metrics}")

    Path("results").mkdir(exist_ok=True)
    with open("results/task1_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open("results/task1_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return model, tokenizer, test_metrics


# ============================================================
# Task 2: GNN + CNN baseline on GTZAN
# ============================================================

def load_gtzan_graphs(split_ids_path, graphs_root="data/processed/gtzan_graphs"):
    with open(split_ids_path) as f:
        split_entries = json.load(f)
    graphs = []
    for entry in split_entries:
        path = Path(graphs_root) / entry["genre"] / f"{entry['track']}.pt"
        graphs.append(torch.load(path, weights_only=False))
    return graphs


def evaluate_gnn(model, loader, device, num_classes=10):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    aucpr_scores = []
    labels_onehot = np.eye(num_classes)[all_labels]
    for c in range(num_classes):
        if labels_onehot[:, c].sum() > 0:
            aucpr_scores.append(ap_score(labels_onehot[:, c], all_probs[:, c]))
    mean_aucpr = float(np.mean(aucpr_scores)) if aucpr_scores else 0.0

    return {"macro_f1": macro_f1, "aucpr": mean_aucpr}


def train_gnn_task2(config_path="config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_graphs = load_gtzan_graphs("data/splits/gtzan_train.json")
    val_graphs = load_gtzan_graphs("data/splits/gtzan_val.json")
    test_graphs = load_gtzan_graphs("data/splits/gtzan_test.json")

    batch_size = config["training"]["batch_size"]
    train_loader = GeoDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    in_channels = train_graphs[0].x.shape[1]
    model = GNNGenreClassifier(
        in_channels=in_channels,
        hidden_channels=config["gnn"]["hidden_channels"],
        num_layers=config["gnn"]["num_layers"],
        num_classes=10,
        dropout=config["gnn"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    epochs = 60
    best_val_f1 = -1
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate_gnn(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}, "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}, val_aucpr={val_metrics['aucpr']:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, **val_metrics})

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            Path("results").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "results/gnn_gtzan_best.pt")

    model.load_state_dict(torch.load("results/gnn_gtzan_best.pt"))
    test_metrics = evaluate_gnn(model, test_loader, device)
    print(f"\nGNN final test metrics: {test_metrics}")

    with open("results/task2_gnn_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open("results/task2_gnn_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return model, test_metrics


class GTZANMelDataset(Dataset):
    def __init__(self, split_ids_path, features_root="data/processed/gtzan"):
        with open(split_ids_path) as f:
            self.entries = json.load(f)
        self.features_root = features_root
        self.genres = ["blues", "classical", "country", "disco", "hiphop",
                       "jazz", "metal", "pop", "reggae", "rock"]
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        path = Path(self.features_root) / entry["genre"] / f"{entry['track']}.npz"
        d = np.load(path)
        mel = torch.tensor(d["full_log_mel"], dtype=torch.float).unsqueeze(0)
        label = self.genre_to_idx[entry["genre"]]
        return mel, label


def cnn_collate_fn(batch):
    mels, labels = zip(*batch)
    max_len = max(m.shape[-1] for m in mels)
    padded = torch.stack([
        F.pad(m, (0, max_len - m.shape[-1])) for m in mels
    ])
    return padded, torch.tensor(labels, dtype=torch.long)


def evaluate_cnn(model, loader, device, num_classes=10):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for mels, labels in loader:
            mels = mels.to(device)
            logits = model(mels)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    aucpr_scores = []
    labels_onehot = np.eye(num_classes)[all_labels]
    for c in range(num_classes):
        if labels_onehot[:, c].sum() > 0:
            aucpr_scores.append(ap_score(labels_onehot[:, c], all_probs[:, c]))
    mean_aucpr = float(np.mean(aucpr_scores)) if aucpr_scores else 0.0

    return {"macro_f1": macro_f1, "aucpr": mean_aucpr}


def train_cnn_baseline(config_path="config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = GTZANMelDataset("data/splits/gtzan_train.json")
    val_ds = GTZANMelDataset("data/splits/gtzan_val.json")
    test_ds = GTZANMelDataset("data/splits/gtzan_test.json")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=cnn_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=cnn_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=cnn_collate_fn)

    model = CNNBaseline(num_classes=10, hidden_channels=config["cnn_baseline"]["hidden_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["cnn_baseline"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    epochs = config["cnn_baseline"]["epochs"]
    best_val_f1 = -1
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for mels, labels in train_loader:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(mels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate_cnn(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}, "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}, val_aucpr={val_metrics['aucpr']:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, **val_metrics})

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            Path("results").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "results/cnn_baseline_best.pt")

    model.load_state_dict(torch.load("results/cnn_baseline_best.pt"))
    test_metrics = evaluate_cnn(model, test_loader, device)
    print(f"\nCNN baseline final test metrics: {test_metrics}")

    with open("results/task2_cnn_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open("results/task2_cnn_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return model, test_metrics


# ============================================================
# Task 3, Stage A: GNN-BERT Fusion on MagnaTagATune
# ============================================================

class MTATFusionDataset(Dataset):
    def __init__(self, clip_ids, text_labels_df, top50_tags, graphs_root="data/processed/magnatagatune/graphs"):
        self.clip_ids = [c for c in clip_ids if (Path(graphs_root) / f"{c}.pt").exists()]
        self.text_labels = text_labels_df.set_index("clip_id")
        self.top50_tags = top50_tags
        self.graphs_root = graphs_root

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        graph = torch.load(Path(self.graphs_root) / f"{clip_id}.pt", weights_only=False)
        row = self.text_labels.loc[clip_id]
        text = row["text"]
        tags = row[self.top50_tags].values.astype("float32")
        return graph, text, tags


def fusion_collate_fn(batch, tokenizer, max_length):
    graphs, texts, tags = zip(*batch)
    graph_batch = Batch.from_data_list(list(graphs))
    tag_tensor = torch.tensor(np.array(tags), dtype=torch.float)
    tokenized = tokenize_batch(tokenizer, list(texts), max_length=max_length)
    return graph_batch, tokenized, tag_tensor


def evaluate_fusion(gnn, bert, fusion, loader, device, mode, threshold=0.5):
    gnn.eval()
    bert.eval()
    if fusion is not None:
        fusion.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for graph_batch, tokenized, tags in loader:
            graph_batch = graph_batch.to(device)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            _, graph_emb = gnn(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            bert_out = bert.bert(input_ids=input_ids, attention_mask=attention_mask)
            token_emb = bert_out.last_hidden_state
            cls_emb = token_emb[:, 0, :]

            if mode == "cross_attention":
                logits, _, _ = fusion(graph_emb, token_emb, attention_mask)
            elif mode == "early_concat":
                logits, _ = fusion(graph_emb, cls_emb)
            elif mode == "bert_only":
                logits = bert.classifier(bert.dropout(cls_emb))
            elif mode == "gnn_only":
                logits = fusion(graph_emb)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(tags.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    binary_preds = (all_preds > threshold).astype(int)
    macro_f1 = f1_score(all_labels, binary_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, binary_preds, average="micro", zero_division=0)
    aucpr_scores = [ap_score(all_labels[:, i], all_preds[:, i])
                     for i in range(all_labels.shape[1]) if all_labels[:, i].sum() > 0]
    mean_aucpr = float(np.mean(aucpr_scores)) if aucpr_scores else 0.0
    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "aucpr": mean_aucpr}


def _tune_threshold_fusion(gnn, bert, fusion, val_loader, device, mode):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.05):
        m = evaluate_fusion(gnn, bert, fusion, val_loader, device, mode, threshold=float(t))
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_t = float(t)
    print(f"Best threshold on validation: {best_t:.2f} (macro_f1={best_f1:.4f})")
    return best_t


def train_fusion_ablation(mode, config_path="config.yaml", epochs=10):
    """
    mode: one of "bert_only", "gnn_only", "early_concat", "cross_attention"
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Mode: {mode}")

    with open("data/processed/magnatagatune/top50_tags.json") as f:
        top50_tags = json.load(f)
    text_labels = pd.read_csv("data/processed/magnatagatune/text_labels.csv")

    with open("data/splits/mtag_train.json") as f:
        train_ids = json.load(f)
    with open("data/splits/mtag_val.json") as f:
        val_ids = json.load(f)
    with open("data/splits/mtag_test.json") as f:
        test_ids = json.load(f)

    tokenizer = load_tokenizer(config["bert"]["model_name"])
    max_length = config["bert"]["max_length"]

    train_ds = MTATFusionDataset(train_ids, text_labels, top50_tags)
    val_ds = MTATFusionDataset(val_ids, text_labels, top50_tags)
    test_ds = MTATFusionDataset(test_ids, text_labels, top50_tags)

    collate = lambda b: fusion_collate_fn(b, tokenizer, max_length)
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    graph_dim = config["gnn"]["hidden_channels"]
    in_channels = train_ds[0][0].x.shape[1]

    gnn = GNNGenreClassifier(
        in_channels=in_channels, hidden_channels=graph_dim,
        num_layers=config["gnn"]["num_layers"], num_classes=10,
        dropout=config["gnn"]["dropout"],
    ).to(device)

    bert = BertTagClassifier(config["bert"]["model_name"], num_tags=50).to(device)

    Path("results").mkdir(exist_ok=True)

    if mode == "bert_only":
        bert.load_state_dict(torch.load("results/bert_best.pt", map_location=device))
        for p in bert.parameters():
            p.requires_grad = False
        fusion = None

        best_t = _tune_threshold_fusion(gnn, bert, fusion, val_loader, device, mode)
        test_metrics = evaluate_fusion(gnn, bert, fusion, test_loader, device, mode, threshold=best_t)
        test_metrics["threshold_used"] = best_t
        print(f"\n[{mode}] Test metrics (no training, reused checkpoint): {test_metrics}")

        with open(f"results/task3_{mode}_test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        return test_metrics

    elif mode == "gnn_only":
        fusion = nn.Linear(graph_dim, 50).to(device)
        trainable_params = list(gnn.parameters()) + list(fusion.parameters())
    elif mode == "early_concat":
        fusion = EarlyConcatFusion(graph_dim=graph_dim, bert_dim=768, num_tags=50).to(device)
        trainable_params = list(gnn.parameters()) + list(fusion.parameters()) + list(bert.parameters())
    elif mode == "cross_attention":
        fusion = CrossAttentionFusion(graph_dim=graph_dim, bert_dim=768, num_tags=50).to(device)
        trainable_params = list(gnn.parameters()) + list(fusion.parameters()) + list(bert.parameters())
    else:
        raise ValueError(f"Unknown mode: {mode}")

    optimizer = torch.optim.AdamW(trainable_params, lr=config["bert"]["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1
    history = []

    for epoch in range(epochs):
        gnn.train()
        bert.train()
        fusion.train()
        total_loss = 0

        for graph_batch, tokenized, tags in tqdm(train_loader, desc=f"[{mode}] Epoch {epoch+1}/{epochs}"):
            graph_batch = graph_batch.to(device)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            tags = tags.to(device)

            optimizer.zero_grad()
            _, graph_emb = gnn(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            bert_out = bert.bert(input_ids=input_ids, attention_mask=attention_mask)
            token_emb = bert_out.last_hidden_state
            cls_emb = token_emb[:, 0, :]

            if mode == "cross_attention":
                logits, _, _ = fusion(graph_emb, token_emb, attention_mask)
            elif mode == "early_concat":
                logits, _ = fusion(graph_emb, cls_emb)
            elif mode == "gnn_only":
                logits = fusion(graph_emb)

            loss = criterion(logits, tags)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate_fusion(gnn, bert, fusion, val_loader, device, mode)
        print(f"[{mode}] Epoch {epoch+1}: train_loss={avg_loss:.4f}, "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}, val_aucpr={val_metrics['aucpr']:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, **val_metrics})

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(gnn.state_dict(), f"results/task3_{mode}_gnn.pt")
            torch.save(fusion.state_dict(), f"results/task3_{mode}_fusion.pt")
            torch.save(bert.state_dict(), f"results/task3_{mode}_bert.pt")

    gnn.load_state_dict(torch.load(f"results/task3_{mode}_gnn.pt"))
    fusion.load_state_dict(torch.load(f"results/task3_{mode}_fusion.pt"))
    bert.load_state_dict(torch.load(f"results/task3_{mode}_bert.pt"))

    best_t = _tune_threshold_fusion(gnn, bert, fusion, val_loader, device, mode)
    test_metrics = evaluate_fusion(gnn, bert, fusion, test_loader, device, mode, threshold=best_t)
    test_metrics["threshold_used"] = best_t
    print(f"\n[{mode}] Final test metrics: {test_metrics}")

    with open(f"results/task3_{mode}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(f"results/task3_{mode}_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


# ============================================================
# Task 3, Stage B: DEAM Emotion Regression (standalone extension)
# ============================================================

from sklearn.metrics import mean_absolute_error, r2_score
from gnn_model import GNNEmotionRegressor


def load_deam_graphs(split_ids_path, graphs_root="data/processed/deam/graphs"):
    with open(split_ids_path) as f:
        song_ids = json.load(f)
    graphs = []
    for song_id in song_ids:
        path = Path(graphs_root) / f"{song_id}.pt"
        if path.exists():
            graphs.append(torch.load(path, weights_only=False))
    return graphs


def evaluate_deam(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds, _ = model(batch.x, batch.edge_index, batch.batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    valence_mae = mean_absolute_error(all_targets[:, 0], all_preds[:, 0])
    arousal_mae = mean_absolute_error(all_targets[:, 1], all_preds[:, 1])
    valence_r2 = r2_score(all_targets[:, 0], all_preds[:, 0])
    arousal_r2 = r2_score(all_targets[:, 1], all_preds[:, 1])

    return {
        "valence_mae": float(valence_mae), "arousal_mae": float(arousal_mae),
        "valence_r2": float(valence_r2), "arousal_r2": float(arousal_r2),
    }


def train_deam_emotion(config_path="config.yaml", epochs=30):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Per the spec's minimum: train/val is the primary partition.
    # test_extension is this project's own added practice (see roadmap).
    train_graphs = load_deam_graphs("data/splits/deam_train.json")
    val_graphs = load_deam_graphs("data/splits/deam_val.json")
    test_graphs = load_deam_graphs("data/splits/deam_test_extension.json")
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test (extension): {len(test_graphs)}")

    batch_size = config["training"]["batch_size"]
    train_loader = GeoDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    in_channels = train_graphs[0].x.shape[1]
    model = GNNEmotionRegressor(
        in_channels=in_channels,
        hidden_channels=config["gnn"]["hidden_channels"],
        num_layers=config["gnn"]["num_layers"],
        dropout=config["gnn"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    alpha = config["fusion"]["alpha"]
    beta = config["fusion"]["beta"]

    best_val_mae_sum = float("inf")
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss, total_valence_loss, total_arousal_loss = 0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds, _ = model(batch.x, batch.edge_index, batch.batch)

            valence_loss = F.mse_loss(preds[:, 0], batch.y[:, 0])
            arousal_loss = F.mse_loss(preds[:, 1], batch.y[:, 1])
            loss = alpha * valence_loss + beta * arousal_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_valence_loss += valence_loss.item()
            total_arousal_loss += arousal_loss.item()

        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_valence_loss = total_valence_loss / n_batches
        avg_arousal_loss = total_arousal_loss / n_batches

        val_metrics = evaluate_deam(model, val_loader, device)
        val_mae_sum = val_metrics["valence_mae"] + val_metrics["arousal_mae"]

        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
              f"(valence_loss={avg_valence_loss:.4f}, arousal_loss={avg_arousal_loss:.4f}) | "
              f"val_valence_mae={val_metrics['valence_mae']:.4f}, val_arousal_mae={val_metrics['arousal_mae']:.4f}, "
              f"val_valence_r2={val_metrics['valence_r2']:.4f}, val_arousal_r2={val_metrics['arousal_r2']:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss,
                         "train_valence_loss": avg_valence_loss, "train_arousal_loss": avg_arousal_loss,
                         **val_metrics})

        if val_mae_sum < best_val_mae_sum:
            best_val_mae_sum = val_mae_sum
            Path("results").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "results/deam_gnn_best.pt")

    model.load_state_dict(torch.load("results/deam_gnn_best.pt"))
    test_metrics = evaluate_deam(model, test_loader, device)
    test_metrics["alpha"] = alpha
    test_metrics["beta"] = beta
    print(f"\nDEAM final test metrics (extension split): {test_metrics}")

    with open("results/task3_deam_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open("results/task3_deam_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return model, test_metrics


# ============================================================
# Task 4: Contrastive Retrieval on MusicCaps
# ============================================================

from contrastive import GraphEncoder, TextEncoder, info_nce_loss
from transformers import BertModel


class MusicCapsDataset(Dataset):
    def __init__(self, ytids, graphs_root="data/processed/musiccaps/graphs"):
        self.ytids = [y for y in ytids if (Path(graphs_root) / f"{y}.pt").exists()]
        self.graphs_root = graphs_root

    def __len__(self):
        return len(self.ytids)

    def __getitem__(self, idx):
        ytid = self.ytids[idx]
        graph = torch.load(Path(self.graphs_root) / f"{ytid}.pt", weights_only=False)
        caption = graph.caption
        return graph, caption, ytid


def musiccaps_collate_fn(batch, tokenizer, max_length):
    graphs, captions, ytids = zip(*batch)
    graph_batch = Batch.from_data_list(list(graphs))
    tokenized = tokenize_batch(tokenizer, list(captions), max_length=max_length)
    return graph_batch, tokenized, list(ytids)


def compute_retrieval_metrics(graph_embeds, text_embeds, ks=(1, 5, 10)):
    """
    graph_embeds, text_embeds: (N, embed_dim), row i of each corresponds
    to the same clip (aligned pairs). Computes R@k both directions.
    """
    sim = graph_embeds @ text_embeds.T  # (N, N), sim[i, j] = audio_i vs caption_j
    n = sim.shape[0]

    results = {}
    for direction, matrix in [("audio_to_caption", sim), ("caption_to_audio", sim.T)]:
        ranks = []
        for i in range(n):
            row = matrix[i]
            correct_score = row[i]
            rank = (row > correct_score).sum().item() + 1  # 1-indexed rank of the correct match
            ranks.append(rank)
        ranks = np.array(ranks)
        for k in ks:
            results[f"{direction}_R@{k}"] = float((ranks <= k).mean())

    return results


def train_contrastive_task4(config_path="config.yaml", epochs=15):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("data/splits/musiccaps_train.json") as f:
        train_ids = json.load(f)
    with open("data/splits/musiccaps_val.json") as f:
        val_ids = json.load(f)
    with open("data/splits/musiccaps_test.json") as f:
        test_ids = json.load(f)

    tokenizer = load_tokenizer(config["bert"]["model_name"])
    max_length = config["bert"]["max_length"]

    train_ds = MusicCapsDataset(train_ids)
    val_ds = MusicCapsDataset(val_ids)
    test_ds = MusicCapsDataset(test_ids)

    collate = lambda b: musiccaps_collate_fn(b, tokenizer, max_length)
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    in_channels = train_ds[0][0].x.shape[1]
    embed_dim = config["contrastive"]["embed_dim"]
    temperature = config["contrastive"]["temperature"]

    graph_encoder = GraphEncoder(
        in_channels=in_channels, hidden_channels=config["gnn"]["hidden_channels"],
        num_layers=config["gnn"]["num_layers"], embed_dim=embed_dim,
        dropout=config["gnn"]["dropout"],
    ).to(device)

    bert_model = BertModel.from_pretrained(config["bert"]["model_name"])
    text_encoder = TextEncoder(bert_model, embed_dim=embed_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(graph_encoder.parameters()) + list(text_encoder.parameters()),
        lr=config["bert"]["learning_rate"],
    )

    def get_all_embeddings(loader):
        graph_encoder.eval()
        text_encoder.eval()
        all_g, all_t = [], []
        with torch.no_grad():
            for graph_batch, tokenized, _ in loader:
                graph_batch = graph_batch.to(device)
                input_ids = tokenized["input_ids"].to(device)
                attention_mask = tokenized["attention_mask"].to(device)
                g = graph_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
                t = text_encoder(input_ids, attention_mask)
                all_g.append(g.cpu())
                all_t.append(t.cpu())
        return torch.cat(all_g), torch.cat(all_t)

    best_val_r5 = -1
    history = []

    for epoch in range(epochs):
        graph_encoder.train()
        text_encoder.train()
        total_loss = 0

        for graph_batch, tokenized, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            graph_batch = graph_batch.to(device)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            optimizer.zero_grad()
            g = graph_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            t = text_encoder(input_ids, attention_mask)
            loss = info_nce_loss(g, t, temperature=temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_g, val_t = get_all_embeddings(val_loader)
        val_metrics = compute_retrieval_metrics(val_g, val_t)
        val_r5_avg = (val_metrics["audio_to_caption_R@5"] + val_metrics["caption_to_audio_R@5"]) / 2

        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, "
              f"val_a2c_R@5={val_metrics['audio_to_caption_R@5']:.4f}, "
              f"val_c2a_R@5={val_metrics['caption_to_audio_R@5']:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, **val_metrics})

        if val_r5_avg > best_val_r5:
            best_val_r5 = val_r5_avg
            Path("results").mkdir(exist_ok=True)
            torch.save(graph_encoder.state_dict(), "results/contrastive_graph_encoder.pt")
            torch.save(text_encoder.state_dict(), "results/contrastive_text_encoder.pt")

    graph_encoder.load_state_dict(torch.load("results/contrastive_graph_encoder.pt"))
    text_encoder.load_state_dict(torch.load("results/contrastive_text_encoder.pt"))

    test_g, test_t = get_all_embeddings(test_loader)
    test_metrics = compute_retrieval_metrics(test_g, test_t)
    print(f"\nFinal test retrieval metrics: {test_metrics}")

    with open("results/task4_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open("results/task4_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return graph_encoder, text_encoder, test_metrics

if __name__ == "__main__":
    train_task1()