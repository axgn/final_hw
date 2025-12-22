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
import redis


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

def federated_average_aggregate(param_files, model_dir, base_params=None, base_vocab=None, base_weight: float = 0.0):

    if not param_files and (base_params is None or base_weight <= 0.0):
        print("错误：没有找到任何客户端模型文件，且不存在可用的全局模型。", flush=True)
        return None, None, 0.0

    print(f"找到 {len(param_files)} 个新增客户端参数文件，开始进行 FedAvg 聚合 (增量模式)...", flush=True)

    aggregated_params = None
    shared_vocab = base_vocab
    total_weight = float(base_weight) if base_weight > 0 else 0.0

    # 如果传入了上一轮聚合得到的全局模型及其累计权重，
    # 则先将其作为初始值纳入本轮加权平均中。
    if base_params is not None and base_weight > 0:
        cleaned_base = {}
        for key, tensor in base_params.items():
            if not torch.is_tensor(tensor):
                continue
            if tensor.dtype in (torch.float32, torch.float64):
                cleaned_base[key] = tensor.float()

        if cleaned_base:
            aggregated_params = {
                k: v.clone() * base_weight for k, v in cleaned_base.items()
            }

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
        return None, None, 0.0

    for key in aggregated_params.keys():
        aggregated_params[key] = aggregated_params[key] / float(total_weight)

    print("参数聚合完成！", flush=True)
    return aggregated_params, shared_vocab, total_weight


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
    all_files = os.listdir(model_dir) if os.path.isdir(model_dir) else []

    # 2. Redis 用于记录“已聚合过的客户端模型”，避免重复参与后续聚合
    redis_client = None
    processed_set = set()
    param_file_paths = []
    try:
        # 同样兼容 REDIS_PORT 为 tcp://host:port 的形式
        redis_host_env = os.getenv("REDIS_HOST")
        redis_port_raw = os.getenv("REDIS_PORT")

        redis_host = redis_host_env or "redis"
        redis_port = 6379

        if redis_port_raw:
            try:
                if redis_port_raw.startswith("tcp://"):
                    addr = redis_port_raw[len("tcp://") :]
                    if ":" in addr:
                        host_part, port_part = addr.split(":", 1)
                        if not redis_host_env:
                            redis_host = host_part
                        redis_port = int(port_part)
                    else:
                        redis_port = int(addr)
                else:
                    redis_port = int(redis_port_raw)
            except ValueError:
                print(
                    f"invalid REDIS_PORT value: {redis_port_raw}, fallback to {redis_port}",
                    flush=True,
                )

        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_key = os.getenv("FEDAVG_REDIS_KEY", "fedavg:processed_models")

        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )
        processed = list(redis_client.smembers(redis_key))
        processed_set = set(processed)
        print(f"从 Redis 读取到 {len(processed)} 个已聚合客户端模型(将跳过): {processed}", flush=True)
    except Exception as e:
        print(f"连接或读取 Redis 失败，将视为暂无已聚合记录: {e}", flush=True)
        redis_client = None

    # 仅保留存在于 models 目录下、尚未在 Redis 记录为“已聚合”的客户端模型
    param_file_paths = [
        f
        for f in all_files
        if (f.endswith(".pt") or f.endswith(".pth"))
        and f.startswith("client_")
        and f not in processed_set
    ]

    # 如果不存在任何新的客户端模型，但已经有聚合好的全局模型，则无需再次聚合
    if not param_file_paths and os.path.exists(os.path.join(model_dir, "sentiment_zh_final.pt")):
        print("未检测到新的客户端模型文件，且已存在聚合后的全局模型，跳过聚合。", flush=True)
        exit(0)

    # 尝试加载上一轮聚合得到的全局模型及其累计权重，实现增量式更新
    base_params = None
    base_vocab = None
    base_weight = 0.0
    global_ckpt_path = os.path.join(model_dir, "sentiment_zh_final.pt")
    if os.path.exists(global_ckpt_path):
        try:
            global_ckpt = torch.load(global_ckpt_path, map_location="cpu")
            if isinstance(global_ckpt, dict) and "model" in global_ckpt:
                base_params = global_ckpt["model"]
                base_vocab = global_ckpt.get("vocab")
                base_weight = float(global_ckpt.get("total_weight", 0.0) or 0.0)
                print(f"已加载上一轮全局模型，累计权重 total_weight={base_weight}")
        except Exception as e:
            print(f"加载上一轮全局模型失败，将仅基于本轮客户端重新聚合: {e}")

    if not param_file_paths and base_params is None:
        print("错误：未在 models 目录下找到任何以 client_ 开头的模型文件，且无已有全局模型。", flush=True)
        exit(1)

    final_aggregated_params, shared_vocab, total_weight = federated_average_aggregate(
        param_file_paths, model_dir, base_params=base_params, base_vocab=base_vocab, base_weight=base_weight
    )

    if final_aggregated_params is None:
        exit(1)

    if shared_vocab is None:
        print("未从客户端模型中发现 vocab，使用完整训练集重新构建 vocab。", flush=True)
        full_dataset = load_dataset("lansinuote/ChnSentiCorp")
        train_texts = full_dataset["train"]["text"]
        shared_vocab = build_vocab(train_texts)

    global_model = SimpleSentiment(len(shared_vocab))
    global_model.load_state_dict(final_aggregated_params)
    print("聚合后的参数已成功加载到情感模型。", flush=True)

    print("\n开始评估聚合后模型在 ChnSentiCorp 测试集上的性能...", flush=True)
    test_loader = build_test_loader(shared_vocab)
    accuracy = test_model(global_model, test_loader)
    print(f"\n聚合后模型在测试集上的准确率为: {accuracy * 100:.2f}%", flush=True)

    save_dir = model_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sentiment_zh_final.pt")
    torch.save({
        "model": final_aggregated_params,
        "vocab": shared_vocab,
        "total_weight": float(total_weight),
    }, save_path)
    print(f"聚合后的最终模型已保存至: {save_path} (total_weight={total_weight})")

    # 聚合完成后，将本轮参与聚合的客户端模型名称写入 Redis，
    # 作为“已聚合”记录，后续聚合会自动跳过这些模型文件。
    if redis_client is not None and param_file_paths:
        try:
            redis_key = os.getenv("FEDAVG_REDIS_KEY", "fedavg:processed_models")
            redis_client.sadd(redis_key, *param_file_paths)
            print(
                f"已将本轮聚合使用的客户端模型写入 Redis 集合 {redis_key}: {param_file_paths}",
                flush=True,
            )
        except Exception as e:
            print(f"将已聚合模型名称写入 Redis 失败: {e}", flush=True)

