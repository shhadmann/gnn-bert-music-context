import sys
sys.path.append("src")

import json
import torch
import pandas as pd
from bert_encoder import BertTagClassifier, load_tokenizer, tokenize_batch

with open("data/processed/magnatagatune/top50_tags.json") as f:
    top50_tags = json.load(f)
with open("data/splits/mtag_test.json") as f:
    test_ids = json.load(f)
with open("results/task1_test_metrics.json") as f:
    threshold = json.load(f)["threshold_used"]

text_labels = pd.read_csv("data/processed/magnatagatune/text_labels.csv")
test_data = text_labels[text_labels["clip_id"].isin(test_ids)].reset_index(drop=True)

sample = test_data.sample(n=5, random_state=42)

tokenizer = load_tokenizer()
model = BertTagClassifier(num_tags=50)
model.load_state_dict(torch.load("results/bert_best.pt", map_location="cpu"))
model.eval()

results = []
for _, row in sample.iterrows():
    text = row["text"]
    true_tags = [t for t in top50_tags if row[t] == 1]

    batch = tokenize_batch(tokenizer, [text])
    with torch.no_grad():
        logits, _ = model(batch["input_ids"], batch["attention_mask"])
    probs = torch.sigmoid(logits)[0]
    predicted_tags = [top50_tags[i] for i in range(50) if probs[i] > threshold]

    results.append({
        "clip_id": int(row["clip_id"]),
        "text": text,
        "true_tags": true_tags,
        "predicted_tags": predicted_tags,
    })
    print(f"\nClip {row['clip_id']}: \"{text}\"")
    print(f"  True tags:      {true_tags}")
    print(f"  Predicted tags: {predicted_tags}")

with open("results/task1_example_predictions.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/task1_example_predictions.json")