# ===== Qwen3-Embedding: L2 Norm → UMAP → KMeans(k=5) Results =====
# Silhouette Score       : 0.3976
# Calinski-Harabasz      : 40211.3
# Davies-Bouldin Index   : 1.0520



import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize  # 新增：用于L2归一化
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import umap


# =============================
# 0. Config
# =============================
NPZ_PATH = "qwen3_8b_embeddings_full_4096d.npz"


def main():
    # =============================
    # 1. Load Qwen3 Embeddings
    # =============================
    data = np.load(NPZ_PATH, allow_pickle=True)
    X_4096 = data["embeddings_4096d"].astype(np.float32)
    texts = data["headlines"]

    print(f"Loaded: {X_4096.shape} embeddings | {len(texts)} headlines")

    # =============================
    # 1.1 Check & Fix NaN / Inf
    # =============================
    print("\n▶ Checking for NaN / Inf in embeddings ...")
    nan_count = np.isnan(X_4096).sum()
    inf_count = np.isinf(X_4096).sum()
    
    if nan_count > 0 or inf_count > 0:
        print(f"→ Found invalid values (NaN: {nan_count}, Inf: {inf_count}), applying np.nan_to_num() ...")
        X_4096 = np.nan_to_num(
            X_4096,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    else:
        print("→ No NaN/Inf detected.")

    # =============================
    # 2. L2 Normalization (New Step)
    # =============================
    # Qwen/大模型嵌入通常基于余弦相似度，L2归一化后欧氏距离等价于余弦距离
    print("\n▶ Applying L2 Normalization ...")
    X_norm = normalize(X_4096, norm='l2')

    # =============================
    # 3. UMAP → 10D (Modified)
    # =============================
    # 直接在 4096 维数据上运行 UMAP，保留 PCA 可能丢失的非线性语义
    print("\n▶ Running UMAP 4096 → 10 (metric='cosine') ...")
    umap_model = umap.UMAP(
        n_components=10,
        n_neighbors=50, 
        min_dist=0.1,
        metric="cosine", # 保持使用 cosine
        random_state=42,
    )
    # 输入变为归一化后的数据
    X_10 = umap_model.fit_transform(X_norm)
    print("After UMAP:", X_10.shape)

    # =============================
    # 4. KMeans Clustering (k=5)
    # =============================
    print("\n▶ Running KMeans (k=5) ...")
    km = KMeans(n_clusters=5, random_state=42, n_init="auto")
    labels = km.fit_predict(X_10)

    # =============================
    # 5. Clustering Metrics
    # =============================
    print("\n===== Qwen3-Embedding: L2 Norm → UMAP → KMeans(k=5) Results =====")
    sil = silhouette_score(X_10, labels)
    ch = calinski_harabasz_score(X_10, labels)
    db = davies_bouldin_score(X_10, labels)

    print(f"Silhouette Score       : {sil:.4f}")
    print(f"Calinski-Harabasz      : {ch:.1f}")
    print(f"Davies-Bouldin Index   : {db:.4f}")

    print("\n🔥 Done — No file saved.")


if __name__ == "__main__":
    main()