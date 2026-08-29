"""
BERT-based multi-label tag classifier for Task 1.
Input: concatenated title+artist+album text.
Output: 50-dim multi-hot tag prediction.
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BertTagClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_tags=50, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)  # raw logits, sigmoid applied in loss/eval
        return logits, cls_embedding


def load_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)


def tokenize_batch(tokenizer, texts, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )