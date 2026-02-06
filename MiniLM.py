# 1_compute_minilm_384d.py
# 功能：调用 all-MiniLM-L6-v2 生成 384 维句向量，并保存为 npz

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer


# === 配置区 ===
MODEL_NAME = "all-MiniLM-L6-v2"                       # Sentence-Transformers 模型名
CSV_PATH = os.path.join("archive", "abcnews-date-text.csv")  # 数据集路径
TEXT_COL = "headline_text"                            # 列名
SAMPLE_SIZE = 50000                                   # 先抽 5 万条做实验
BATCH_SIZE = 64                                       # batch 大小


def load_data():
    print(f"==> 读取数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, usecols=[TEXT_COL])
    df = df.dropna()
    if SAMPLE_SIZE is not None:
        df = df.head(SAMPLE_SIZE)
    texts = df[TEXT_COL].astype(str).tolist()
    print(f"==> 共 {len(texts)} 条样本用于计算 embedding")
    return texts


def load_model():
    print("==> 加载 MiniLM SentenceTransformer 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # SentenceTransformer 会自己处理 tokenizer + pooling + device
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("==> 模型加载完成，当前设备:", device)
    return model


def compute_all_embeddings(model, texts):
    """
    使用 SentenceTransformer 的 encode 分批计算 embedding
    返回: numpy 数组 [N, 384]
    """
    all_embs = []
    n = len(texts)
    print("==> 开始批量计算 MiniLM embedding...")

    for i in tqdm(range(0, n, BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]
        # normalize_embeddings=True 会直接做 L2 标准化
        embs = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embs.append(embs)

    all_embs = np.vstack(all_embs)  # [N, 384]
    print("==> 全部 embedding 计算完成, 形状:", all_embs.shape)
    return all_embs


def main():
    # 1. 读数据
    texts = load_data()

    # 2. 加载 MiniLM 模型
    model = load_model()

    # 3. 计算 384 维 embedding
    embs = compute_all_embeddings(model, texts)
    print("MiniLM embedding shape =", embs.shape)

    # 4. 保存 384 维向量
    out_path = "minilm_embeddings_384d.npz"
    np.savez(out_path, embeddings_384d=embs, headlines=np.array(texts))
    print(f"==> 已保存 MiniLM 384 维到 {out_path}")
    print("   - embeddings_384d: shape =", embs.shape)
    print("   - headlines: 文本列表, len =", len(texts))


if __name__ == "__main__":
    main()
