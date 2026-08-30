"""
Training script for the GNN-BERT music context project.
Task 1: BERT multi-label tag classifier (MagnaTagATune).
Task 2: GNN + CNN baseline genre classifier (GTZAN).
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
from tqdm import tqdm

from bert_encoder import BertTagClassifier, load_tokenizer, tokenize_batch
from gnn_model import GNNGenreClassifier, CNNBaseline


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
    """
    MagnaTagATune's tags are heavily imbalanced (most labels are 0 per
    clip), which commonly pushes a fixed 0.5 threshold to read F1=0
    even when the model's ranking (AUC-PR) is meaningfully non-random.
    This scans a range of thresholds on validation data and picks the
    one that actually maximizes macro-F1, rather than trusting 0.5
    blindly.
    """
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

    if max_samples:  # for quick local sanity-checks only
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

    epochs = 5  # BERT fine-tuning typically needs few epochs
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
        split_entries = json.load(f)  # list of {"genre": ..., "track": ...}
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
    """For the CNN baseline — loads full-track log-mel spectrograms directly."""
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
        mel = torch.tensor(d["full_log_mel"], dtype=torch.float).unsqueeze(0)  # (1, n_mels, time)
        label = self.genre_to_idx[entry["genre"]]
        return mel, label


def cnn_collate_fn(batch):
    """Pad variable-length time dimension to the max length in the batch."""
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


if __name__ == "__main__":
    train_task1()