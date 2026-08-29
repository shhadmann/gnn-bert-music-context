"""
Training script for Task 1: BERT multi-label tag classifier.
Designed to run on Colab (GPU) but will also run on CPU (slow) for
quick local sanity-checks on a tiny subset.
"""

import sys
sys.path.append("src")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from bert_encoder import BertTagClassifier, load_tokenizer, tokenize_batch


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
            aucpr_scores.append(average_precision_score(all_labels[:, i], all_preds[:, i]))
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
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

    # Final test evaluation using the best checkpoint, with a tuned
    # threshold rather than a blind 0.5 — see find_best_threshold's
    # docstring for why this matters given MagnaTagATune's label sparsity.
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


if __name__ == "__main__":
    train_task1()