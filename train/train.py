# train_sentiment_zh_hf.py
# 中文情感分析（真实数据集：ChnSentiCorp）
# HuggingFace datasets + PyTorch + ONNX
# CPU 友好 / 无 torchtext

import jieba
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
import socket

# =========================
# 1. 中文分词
# =========================
def tokenize(text: str):
    return list(jieba.cut(text))


# =========================
# 2. 构建词表
# =========================
def build_vocab(texts, min_freq=5):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


# =========================
# 3. Dataset
# =========================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        tokens = tokenize(text)
        ids = [self.vocab.get(t, 1) for t in tokens][:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.encode(self.texts[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.stack(ys)
    return xs, ys


# =========================
# 4. 模型
# =========================
class SimpleSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x)    # [B, T, D]
        x = x.mean(dim=1)        # mean pooling
        return torch.sigmoid(self.fc(x)).squeeze(1)


# =========================
# 5. 主流程
# =========================
def main():
    dataset = load_dataset("lansinuote/ChnSentiCorp")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    print("train size:", len(train_texts))
    print("test size :", len(test_texts))

    vocab = build_vocab(train_texts)
    print("vocab size:", len(vocab))

    train_ds = SentimentDataset(train_texts, train_labels, vocab)
    test_ds = SentimentDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    model = SimpleSentiment(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(5):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"epoch {epoch} train loss {total_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x) > 0.5
                correct += (pred.float() == y).sum().item()
                total += y.size(0)

        print(f"epoch {epoch} test acc {(correct/total):.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
        },
        f"models/client_{socket.gethostname()}.pt",
    )

    print(f"模型已保存至 models/client_{socket.gethostname()}.pt")


if __name__ == "__main__":
    main()
