# ================= Summary: PCA→128, UMAP→10, KMeans k=5 =================
# Model                Silhouette         Calinski-H     Davies-Bouldin
# BGE-384d                 0.3789            39933.7             1.1879
# MiniLM-384d              0.3800            31895.6             1.1181
# Nemotron-4096d           0.3496            34660.6             1.1777
# Qwen3-8B-4096d           0.3996            42385.5             1.0983


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the same pipeline (PCA -> UMAP -> KMeans) on 4 different embeddings:
- bge_embeddings_384d.npz
- minilm_embeddings_384d.npz
- nemotron_embeddings_full_4096d.npz
- qwen3_8b_embeddings_full_4096d.npz

Pipeline:
    embeddings -> PCA(128) -> UMAP(10) -> KMeans(k=5)
Metrics:
    Silhouette, Calinski-Harabasz, Davies-Bouldin
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import umap


# =============================
# 0. Config
# =============================

# 你当前目录下的文件名
MODEL_FILES = {
    "BGE-384d": "bge_embeddings_384d.npz",
    "MiniLM-384d": "minilm_embeddings_384d.npz",
    "Nemotron-4096d": "nemotron_embeddings_full_4096d.npz",
    "Qwen3-8B-4096d": "qwen3_8b_embeddings_full_4096d.npz",
}

# 统一的降维 + 聚类配置
PCA_DIM = 128
UMAP_DIM = 10
N_CLUSTERS = 5
RANDOM_STATE = 42


# =============================
# 工具：通用加载 .npz 里的 embeddings
# =============================
def load_embeddings(path: str):
    """
    尝试从 .npz 中拿出 embeddings 和可选的 headlines。
    兼容几种常见 key 命名：
      - 'embeddings_4096d', 'embeddings_384d', 'embeddings'
      - 否则取第一个数组
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.files)

    emb_key = None
    for k in ["embeddings_4096d", "embeddings_384d", "embeddings"]:
        if k in keys:
            emb_key = k
            break
    if emb_key is None:
        # 兜底：直接取第一个 key
        emb_key = keys[0]

    X = data[emb_key].astype(np.float32)

    # headlines 如果有就拿出来（目前不用，只是方便以后 debug）
    texts = None
    for hk in ["headlines", "texts"]:
        if hk in keys:
            texts = data[hk]
            break

    return X, texts


# =============================
# 单模型：PCA -> UMAP -> KMeans
# =============================
def run_pipeline_for_model(model_name: str, npz_path: str):
    print(f"\n================= {model_name} =================")
    print(f"Loading from: {npz_path}")

    # 1. Load embeddings
    X, texts = load_embeddings(npz_path)
    print(f"Loaded embeddings: {X.shape}")

    # 1.1 NaN / Inf 检查
    print("\n▶ Checking for NaN / Inf in embeddings ...")
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"NaN count: {nan_count}, Inf count: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print("→ Found invalid values, applying np.nan_to_num() ...")
        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        nan_after = np.isnan(X).sum()
        inf_after = np.isinf(X).sum()
        print(f"After cleaning → NaN: {nan_after}, Inf: {inf_after}")
    else:
        print("→ No NaN/Inf detected.")

    # 2. PCA: original_dim -> 128
    print("\n▶ Running PCA → 128 ...")
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    print("After PCA:", X_pca.shape)

    # 3. UMAP: 128 -> 10
    print("\n▶ Running UMAP 128 → 10 ...")
    umap_model = umap.UMAP(
        n_components=UMAP_DIM,
        n_neighbors=50,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    X_umap = umap_model.fit_transform(X_pca)
    print("After UMAP:", X_umap.shape)

    # 4. KMeans(k=5)
    print("\n▶ Running KMeans (k=5) ...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X_umap)

    # 5. Metrics
    print(f"\n===== {model_name}: PCA → UMAP → KMeans(k={N_CLUSTERS}) Results =====")
    sil = silhouette_score(X_umap, labels)
    ch = calinski_harabasz_score(X_umap, labels)
    db = davies_bouldin_score(X_umap, labels)

    print(f"Silhouette Score       : {sil:.4f}")
    print(f"Calinski-Harabasz      : {ch:.1f}")
    print(f"Davies-Bouldin Index   : {db:.4f}")

    return {
        "model": model_name,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
    }


# =============================
# 主入口：依次跑 4 个模型
# =============================
def main():
    all_results = []

    for name, path in MODEL_FILES.items():
        res = run_pipeline_for_model(name, path)
        all_results.append(res)

    # 打一个总表方便你 copy 到报告里
    print("\n================= Summary: PCA→128, UMAP→10, KMeans k=5 =================")
    print("{:<18} {:>12} {:>18} {:>18}".format(
        "Model", "Silhouette", "Calinski-H", "Davies-Bouldin"
    ))
    for r in all_results:
        print("{:<18} {:>12.4f} {:>18.1f} {:>18.4f}".format(
            r["model"], r["silhouette"], r["calinski_harabasz"], r["davies_bouldin"]
        ))

    print("\n🔥 Done — no files saved.")


if __name__ == "__main__":
    main()
