import sys
sys.path.append("src")

from bert_encoder import BertTagClassifier, load_tokenizer, tokenize_batch

print("Loading tokenizer and model (this downloads bert-base-uncased on first run, ~440MB)...")

tokenizer = load_tokenizer()
model = BertTagClassifier(num_tags=50)
model.eval()

sample_texts = ["BWV54 I Aria American Bach Soloists", "Some Rock Song The Rock Band"]
batch = tokenize_batch(tokenizer, sample_texts)

print("Input IDs shape:", batch["input_ids"].shape)

import torch
with torch.no_grad():
    logits, cls_emb = model(batch["input_ids"], batch["attention_mask"])

print("Logits shape:", logits.shape)
print("CLS embedding shape:", cls_emb.shape)
print("Sample logits (first row, first 5 values):", logits[0][:5])