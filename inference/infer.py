#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""简单的中文情感分析推理服务

基于 `train/train.py` 训练得到的:
- 模型权重: models/sentiment_zh.pt

提供 HTTP 接口:
- POST /predict  JSON: {"text": "这家店真的很好，强烈推荐！"}
    返回: {"label": "pos"|"neg", "prob": 0.93}
"""

import sys
from typing import List, Tuple

import jieba
import torch
import torch.nn as nn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager


# =========================
# 与训练脚本保持一致的组件
# =========================

def tokenize(text: str) -> List[str]:
    """中文分词，保持与训练时一致。"""
    return list(jieba.cut(text))


class SimpleSentiment(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        x = self.embedding(x)  # [B, T, D]
        x = x.mean(dim=1)      # mean pooling -> [B, D]
        return torch.sigmoid(self.fc(x)).squeeze(1)  # [B]


class SentimentInferencer:
    def __init__(self, ckpt_path: str = "models/sentiment_zh_final.pt", device: str = "cpu") -> None:
        self.device = torch.device(device)

        # 加载 checkpoint
        data = torch.load(ckpt_path, map_location=self.device)
        vocab = data["vocab"]
        state_dict = data["model"]

        self.vocab = vocab
        self.pad_id = 0
        self.unk_id = 1

        self.model = SimpleSentiment(len(vocab))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    # -------- 编码与推理 --------
    def _encode(self, text: str, max_len: int = 128) -> torch.Tensor:
        tokens = tokenize(text)
        ids = [self.vocab.get(t, self.unk_id) for t in tokens][:max_len]
        if not ids:  # 空文本兜底
            ids = [self.pad_id]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def predict_proba(self, text: str) -> float:
        """返回正向情感的概率 (0~1)。"""
        x = self._encode(text).unsqueeze(0)  # [1, T]
        score = self.model(x)[0].item()
        # 数值安全裁剪
        score = max(0.0, min(1.0, float(score)))
        return score

    @torch.no_grad()
    def predict_label(self, text: str, threshold: float = 0.5) -> Tuple[str, float]:
        """返回 (label, prob)，label 为 'pos' 或 'neg'。"""
        prob = self.predict_proba(text)
        label = "pos" if prob >= threshold else "neg"
        return label, prob


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    prob: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.infer = SentimentInferencer(
        ckpt_path="models/sentiment_zh_final.pt",
        device="cpu",
    )
    yield

app = FastAPI(title="Chinese Sentiment Inference API", lifespan=lifespan)

# 允许前端通过浏览器跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 如需更严格控制可改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    infer: SentimentInferencer = app.state.infer
    label, prob = infer.predict_label(req.text, threshold=0.5)
    return PredictResponse(label=label, prob=prob)


def main() -> None:
    """启动 FastAPI 服务"""
    uvicorn.run("infer:app", host="0.0.0.0", port=9000, reload=False)


if __name__ == "__main__":
    main()
