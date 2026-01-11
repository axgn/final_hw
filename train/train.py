import jieba
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
import socket
import os
import pymysql

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
# 5. 从数据库实时获取评论
# =========================
def load_comments_from_db(limit: int = 5000):

    host = os.getenv("MYSQL_HOST", "mysql")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "123456")
    database = os.getenv("MYSQL_DB", "blog")

    try:
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
    except Exception as e:
        print(f"failed to connect mysql: {e}")
        return []

    comments = []
    try:
        with conn.cursor() as cursor:
            sql = "SELECT content FROM comments ORDER BY created_at DESC LIMIT %s"
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            for row in rows:
                text = (row.get("content") or "").strip()
                if text:
                    comments.append(text)
    except Exception as e:
        print(f"failed to load comments from db: {e}")
    finally:
        conn.close()

    print(f"loaded {len(comments)} comments from database")
    return comments


def pseudo_label_comments(texts, model, vocab, batch_size: int = 64):
    """用当前全局模型对数据库中的评论做预测，生成伪标签。

    返回与 texts 等长的标签列表（0/1）。若 texts 为空或模型为 None，则返回空列表。
    """

    if not texts or model is None:
        return []

    ds = SentimentDataset(texts, [0] * len(texts), vocab)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    model.eval()
    all_labels = []
    with torch.no_grad():
        for x, _ in loader:
            prob = model(x)
            preds = (prob > 0.5).float()
            all_labels.extend(preds.tolist())

    # 将 float 转成 int，保证与原始监督数据标签格式一致
    return [int(1 if p >= 0.5 else 0) for p in all_labels]


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

    global_ckpt_path = "models/sentiment_zh_final.pt"
    global_ckpt = None
    if os.path.exists(global_ckpt_path):
        try:
            global_ckpt = torch.load(global_ckpt_path, map_location="cpu")
            if isinstance(global_ckpt, dict) and "vocab" in global_ckpt:
                vocab = global_ckpt["vocab"]
                print("loaded global vocab, size:", len(vocab))
        except Exception as e:
            print(f"failed to load global checkpoint: {e}")

    train_ds = SentimentDataset(train_texts, train_labels, vocab)
    test_ds = SentimentDataset(test_texts, test_labels, vocab)

    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    model = SimpleSentiment(len(vocab))

    if global_ckpt is not None:
        try:
            if isinstance(global_ckpt, dict) and "model" in global_ckpt:
                state = global_ckpt["model"]
            else:
                state = global_ckpt
            model.load_state_dict(state, strict=False)
            print("loaded global model state for continued training")
        except Exception as e:
            print(f"failed to load global model state: {e}")

    extra_texts = load_comments_from_db(limit=5000)
    extra_labels = []
    if extra_texts:
        try:
            extra_labels = pseudo_label_comments(extra_texts, model, vocab)
            print(f"pseudo labeled {len(extra_labels)} comments from db")
        except Exception as e:
            print(f"failed to pseudo label comments: {e}")
            extra_texts = []
            extra_labels = []

    if extra_texts and len(extra_texts) == len(extra_labels):
        merged_texts = list(train_texts) + list(extra_texts)
        merged_labels = list(train_labels) + list(extra_labels)
        train_ds = SentimentDataset(merged_texts, merged_labels, vocab)
        print(
            f"total train samples: labeled={len(train_texts)}, from_db={len(extra_texts)}, total={len(merged_texts)}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(1):
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

    client_ckpt_name = f"client_{socket.gethostname()}.pt"
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, client_ckpt_name)

    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "num_samples": len(train_texts),
        },
        save_path,
    )

    print(f"模型已保存至 {save_path}")



if __name__ == "__main__":
    main()
