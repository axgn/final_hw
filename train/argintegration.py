"""参数聚合脚本（情感分析模型）

使用与 `train.py` 相同的中文情感分析模型和数据，
对多个客户端保存下来的模型参数进行 FedAvg 聚合，
并在 ChnSentiCorp 测试集上评估聚合后模型的准确率。
"""

import os
from collections import Counter

import jieba
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# =========================
# 1. 中文分词
# =========================
def tokenize(text: str):
    return list(jieba.cut(text))


# =========================
# 2. 构建词表（仅在客户端训练时使用，这里作为备用）
#    聚合脚本优先使用各客户端保存下来的 vocab
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
# 4. 模型（与 train.py 中一致）
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

def federated_average_aggregate(param_files, model_dir):
    """对多个客户端的模型参数进行 FedAvg 聚合。

    支持两种保存格式：
    1) 直接保存 state_dict
    2) 保存字典 {"model": state_dict, "vocab": vocab, ...}
    """

    if not param_files:
        print("错误：没有找到任何客户端模型文件。", flush=True)
        return None, None

    print(f"找到 {len(param_files)} 个参数文件，开始进行 FedAvg 聚合...", flush=True)

    aggregated_params = None
    shared_vocab = None
    total_weight = 0.0

    for file_name in param_files:
        path = os.path.join(model_dir, file_name)
        print(f"  - 正在处理文件: {path}", flush=True)

        ckpt = torch.load(path, map_location="cpu")

        # 兼容 train.py 的保存格式
        if isinstance(ckpt, dict) and "model" in ckpt:
            client_params = ckpt["model"]
            if shared_vocab is None and "vocab" in ckpt:
                shared_vocab = ckpt["vocab"]
            # 使用每个客户端的样本数作为聚合权重，未提供时默认为 1
            client_weight = float(ckpt.get("num_samples", 1.0))
        else:
            client_params = ckpt
            client_weight = 1.0

        if client_weight <= 0:
            client_weight = 1.0

        # 预处理：统⼀转成 float，跳过不该聚合的 key
        cleaned_params = {}
        for key, tensor in client_params.items():
            if not torch.is_tensor(tensor):
                continue
            if tensor.dtype in (torch.float32, torch.float64):
                cleaned_params[key] = tensor.float()

        if not cleaned_params:
            continue

        # 按样本数加权累加参数，用于 FedAvg
        if aggregated_params is None:
            aggregated_params = {
                k: v.clone() * client_weight for k, v in cleaned_params.items()
            }
        else:
            for key, tensor in cleaned_params.items():
                if key in aggregated_params:
                    aggregated_params[key] += tensor * client_weight
                else:
                    aggregated_params[key] = tensor.clone() * client_weight

        total_weight += client_weight

    if aggregated_params is None or total_weight == 0.0:
        print("错误：未能从客户端模型中提取到可聚合参数。", flush=True)
        return None, None

    for key in aggregated_params.keys():
        aggregated_params[key] = aggregated_params[key] / float(total_weight)

    print("参数聚合完成！", flush=True)
    return aggregated_params, shared_vocab


def build_test_loader(vocab):
    """构建与 train.py 相同的 ChnSentiCorp 测试集 DataLoader。"""

    dataset = load_dataset("lansinuote/ChnSentiCorp")
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    test_ds = SentimentDataset(test_texts, test_labels, vocab)

    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    return test_loader


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x) > 0.5
            correct += (pred.float() == y).sum().item()
            total += y.size(0)

    if total == 0:
        return 0.0
    return correct / total


if __name__ == "__main__":
    # 优先使用与 train.py 一致的目录
    if os.path.isdir("./models"):
        model_dir = "./models"
    else:
        model_dir = "./models"

    # 兼容 .pt / .pth 文件
    all_files = os.listdir(model_dir) if os.path.isdir(model_dir) else []
    param_file_paths = [
        f for f in all_files
        if (f.endswith(".pt") or f.endswith(".pth")) and f.startswith("client_")
    ]

    if not param_file_paths:
        print("错误：未在 models 目录下找到任何以 client_ 开头的模型文件。", flush=True)
    else:
        # 执行聚合
        final_aggregated_params, shared_vocab = federated_average_aggregate(
            param_file_paths, model_dir
        )

        if final_aggregated_params is None:
            exit(1)

        # 如果客户端没有保存 vocab，则与 train.py 一样，用完整训练集重新构建
        if shared_vocab is None:
            print("未从客户端模型中发现 vocab，使用完整训练集重新构建 vocab。", flush=True)
            full_dataset = load_dataset("lansinuote/ChnSentiCorp")
            train_texts = full_dataset["train"]["text"]
            shared_vocab = build_vocab(train_texts)

        # 根据 vocab 大小构建模型并加载聚合参数
        global_model = SimpleSentiment(len(shared_vocab))
        global_model.load_state_dict(final_aggregated_params)
        print("聚合后的参数已成功加载到情感模型。", flush=True)

        # 在测试集上评估最终模型的性能
        print("\n开始评估聚合后模型在 ChnSentiCorp 测试集上的性能...", flush=True)
        test_loader = build_test_loader(shared_vocab)
        accuracy = test_model(global_model, test_loader)
        print(f"\n聚合后模型在测试集上的准确率为: {accuracy * 100:.2f}%", flush=True)

        # 保存聚合后的最终模型（保持与 train.py 近似的保存方式）
        save_dir = model_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "sentiment_zh_final.pt")
        torch.save({"model": final_aggregated_params, "vocab": shared_vocab}, save_path)
        print(f"聚合后的最终模型已保存至: {save_path}")
