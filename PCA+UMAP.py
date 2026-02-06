# ===== Qwen3-Embedding: PCA → UMAP → KMeans(best params) Results =====
# Silhouette Score       : 0.4721
# Calinski-Harabasz      : 58344.8
# Davies-Bouldin Index   : 0.8954

# 📈 Cluster Distribution:
#   Cluster 0: 9835 samples (19.7%)
#   Cluster 1: 8113 samples (16.2%)
#   Cluster 2: 11559 samples (23.1%)
#   Cluster 3: 20493 samples (41.0%)

# 🔥 Done — Best-params check finished.
# (qwen) xu@LAPTOP-E2VGTV54:~/dsai4203$ 

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
NPZ_PATH = "qwen3_8b_embeddings_full_4096d.npz"

# Using the data after parameter tuning
BEST_PCA_COMPONENTS = 128
BEST_UMAP_COMPONENTS = 10
BEST_UMAP_N_NEIGHBORS = 20
BEST_UMAP_MIN_DIST = 0.0019353832093454704  # 你也可以写成 0.002
BEST_K = 4


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
    print(f"NaN count: {nan_count}, Inf count: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print("→ Found invalid values, applying np.nan_to_num() ...")
        X_4096 = np.nan_to_num(
            X_4096,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    else:
        print("→ No NaN/Inf detected.")

    # 再检查一遍确保干净
    nan_count_after = np.isnan(X_4096).sum()
    inf_count_after = np.isinf(X_4096).sum()
    print(f"After cleaning → NaN: {nan_count_after}, Inf: {inf_count_after}")

    # =============================
    # 2. PCA → 128 (最优参数)
    # =============================
    print(f"\n▶ Running PCA 4096 → {BEST_PCA_COMPONENTS} ...")
    pca = PCA(n_components=BEST_PCA_COMPONENTS, random_state=42)
    X_128 = pca.fit_transform(X_4096)
    print("After PCA:", X_128.shape)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA cumulative explained variance: {explained_var:.3f}")

    # =============================
    # 3. UMAP → 10D（最优参数）
    # =============================
    print(
        f"\n▶ Running UMAP {BEST_PCA_COMPONENTS} → {BEST_UMAP_COMPONENTS} "
        f"(n_neighbors={BEST_UMAP_N_NEIGHBORS}, min_dist={BEST_UMAP_MIN_DIST:.4f}) ..."
    )
    umap_model = umap.UMAP(
        n_components=BEST_UMAP_COMPONENTS,
        n_neighbors=BEST_UMAP_N_NEIGHBORS,
        min_dist=BEST_UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    X_10 = umap_model.fit_transform(X_128)
    print("After UMAP:", X_10.shape)

    # =============================
    # 4. KMeans Clustering (k=4 最优参数)
    # =============================
    print(f"\n▶ Running KMeans (k={BEST_K}) ...")
    km = KMeans(n_clusters=BEST_K, random_state=42, n_init="auto")
    labels = km.fit_predict(X_10)

    # =============================
    # 5. Clustering Metrics
    # =============================
    print("\n===== Qwen3-Embedding: PCA → UMAP → KMeans(best params) Results =====")
    sil = silhouette_score(X_10, labels)
    ch = calinski_harabasz_score(X_10, labels)
    db = davies_bouldin_score(X_10, labels)

    print(f"Silhouette Score       : {sil:.4f}")
    print(f"Calinski-Harabasz      : {ch:.1f}")
    print(f"Davies-Bouldin Index   : {db:.4f}")

    # =============================
    # 6. Cluster Distribution
    # =============================
    print("\n📈 Cluster Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    for cid, cnt in zip(unique, counts):
        ratio = cnt / total * 100
        print(f"  Cluster {cid}: {cnt} samples ({ratio:.1f}%)")

    print("\n🔥 Done — Best-params check finished.")


if __name__ == "__main__":
    main()
