# -*- coding: utf-8 -*-
"""
功能：调用 BAAI/bge-small-en-v1.5 生成 384 维 sentence embedding，并保存为 npz

输入：
    archive/abcnews-date-text.csv
输出：
    bge_embeddings_384d.npz
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer


# === 配置区 ===
MODEL_NAME = "BAAI/bge-small-en-v1.5"                 # ★ 更适合聚类的对比学习 embedding
CSV_PATH = os.path.join("archive", "abcnews-date-text.csv")
TEXT_COL = "headline_text"
SAMPLE_SIZE = 50000
BATCH_SIZE = 64


def load_data():
    print(f"==> 读取数据集: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, usecols=[TEXT_COL])
    df = df.dropna()
    if SAMPLE_SIZE is not None:
        df = df.head(SAMPLE_SIZE)

    texts = df[TEXT_COL].astype(str).tolist()
    print(f"==> 本次共使用 {len(texts)} 条文本")
    return texts


def load_model():
    print("==> 加载 BGE-small 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU 设备:", torch.cuda.get_device_name(0))

    # SentenceTransformer 会自动处理 tokenizer + pooling
    model = SentenceTransformer(MODEL_NAME, device=device)

    print(f"==> 模型加载完成，当前使用设备: {device}")
    return model


def compute_embeddings(model, texts):
    """
    分批生成 BGE embedding（384维），自动 L2 标准化。
    """
    all_embs = []
    n = len(texts)

    print("==> 开始生成 BGE-small embedding...")
    for i in tqdm(range(0, n, BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]
        embs = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,    # ★ 强烈推荐：对比学习 embedding 必须 normalize
            show_progress_bar=False,
        )
        all_embs.append(embs)

    all_embs = np.vstack(all_embs)
    print("==> embedding 生成完成, 形状:", all_embs.shape)
    return all_embs


def main():
    # 1) 读取文本
    texts = load_data()

    # 2) 载入 BGE-small 模型
    model = load_model()

    # 3) 生成 embedding
    embs = compute_embeddings(model, texts)
    print("BGE embedding shape =", embs.shape)

    # 4) 保存 npz 文件
    out_path = "bge_embeddings_384d.npz"
    np.savez(out_path, embeddings_384d=embs, headlines=np.array(texts))

    print("\n==> 已成功保存到:", out_path)
    print(" 📌 embeddings_384d:", embs.shape)
    print(" 📌 headlines:", len(texts))


if __name__ == "__main__":
    main()
