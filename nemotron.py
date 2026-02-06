# 1_compute_nemotron_4096d.py
# 功能：调用 nvidia/llama-embed-nemotron-8b 生成 4096 维句向量，并保存为 npz

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# === 配置区 ===
MODEL_NAME = "nvidia/llama-embed-nemotron-8b"
CSV_PATH = os.path.join("archive", "abcnews-date-text.csv")  # 数据集路径
TEXT_COL = "headline_text"                                   # 列名
SAMPLE_SIZE = 50000                                          # 先抽 5 万条做实验
BATCH_SIZE = 64                                              # batch 大小


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
    print("==> 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left",
    )

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        # 有 GPU：用 device_map="auto"，max_memory 的 key 用整数 GPU id
        max_memory = {
            0: "5GiB",       # GPU 0，对应 3060 6G，给 5GiB
            "cpu": "20GiB",  # 按你内存大小调
        }
        device_map = "auto"
        print("==> 检测到 GPU，使用 CPU+GPU 混合加载")
        print("GPU:", torch.cuda.get_device_name(0))
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            max_memory=max_memory,
            attn_implementation="eager",   # 强制用 eager 注意力
        ).eval()
    else:
        # 没有 GPU：全部丢在 CPU 上
        print("==> 未检测到 GPU，仅使用 CPU 加载（会比较慢）")
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
            attn_implementation="eager",
        ).eval()

    print("==> 模型加载完成")
    print("设备分配(device_map):", getattr(model, "hf_device_map", "N/A"))
    return tokenizer, model


def get_model_device(model):
    """
    兼容 device_map=\"auto\" 的情况，找一个实际的 device 用来放 input。
    """
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        first_key = next(iter(model.hf_device_map.keys()))
        device = model.hf_device_map[first_key]
        return torch.device(device)
    return next(model.parameters()).device


def get_batch_embeddings(tokenizer, model, texts):
    """
    texts: List[str]
    return: Tensor [batch_size, hidden_dim] on CPU
    """
    device = get_model_device(model)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        outputs = model(**encoded)
        # 先转回 fp32，数值稳定一点
        last_hidden_state = outputs.last_hidden_state.to(torch.float32)

        # === 标准 mean pooling：mask 掉 PAD 再平均 ===
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = (last_hidden_state * input_mask_expanded).sum(dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        embeddings = sum_embeddings / sum_mask.clamp(min=1e-9)

        # L2 normalize（Nemotron 官方推荐）
        embeddings = F.normalize(embeddings, dim=-1)

    return embeddings.cpu()  # 放回 CPU，方便后续拼接 & 存 npz


def compute_all_embeddings(tokenizer, model, texts):
    all_embs = []
    n = len(texts)
    print("==> 开始批量计算 embedding...")

    for i in tqdm(range(0, n, BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]
        embs = get_batch_embeddings(tokenizer, model, batch)
        all_embs.append(embs)

    all_embs = torch.cat(all_embs, dim=0).numpy()  # [N, 4096]
    print("==> 全部 embedding 计算完成, 形状:", all_embs.shape)
    return all_embs


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # 1. 读数据
    texts = load_data()

    # 2. 加载模型
    tokenizer, model = load_model()

    # 3. 计算高维 embedding (4096 维)
    embs = compute_all_embeddings(tokenizer, model, texts)
    print("高维 embedding shape =", embs.shape)

    # 4. 保存没降维的 4096 维向量
    full_out_path = "nemotron_embeddings_full_4096d.npz"
    np.savez(full_out_path, embeddings_4096d=embs, headlines=np.array(texts))
    print(f"==> 已保存原始 4096 维到 {full_out_path}")
    print("   - embeddings_4096d: shape =", embs.shape)
    print("   - headlines: 文本列表, len =", len(texts))


if __name__ == "__main__":
    main()
