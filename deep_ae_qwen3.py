# ============================================================
# OPTIMIZATION RESULTS:
# ============================================================

# Best silhouette score (tuning subset): 0.5112
# Best parameters found:
#   hidden_dim1              : 1024
#   hidden_dim2              : 1024
#   hidden_dim3              : 128
#   latent_dim               : 15
#   batch_size               : 2048
#   epochs                   : 45
#   learning_rate            : 9.082037664301222e-05
#   n_clusters               : 4

# Best trial details:
#   Calinski-Harabasz: 17841.6
#   Davies-Bouldin   : 0.5780
#   Latent dimension : 15

# ============================================================
# Running final pipeline with best parameters (FULL DATA):
# ============================================================

# ▶ Model Architecture: [1024, 1024, 128, 15]
# ▶ Training: batch_size=2048, epochs=45, lr=0.00009
# ▶ Clustering: k=4
# /home/xu/dsai4203/deepae.py:138: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
#   scaler = torch.cuda.amp.GradScaler()  # 混合精度，显存更省，速度更快
# /home/xu/dsai4203/deepae.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   with torch.cuda.amp.autocast():
#     [Epoch 1/45] MSE Loss: 0.022429
#     [Epoch 5/45] MSE Loss: 0.018574
#     [Epoch 10/45] MSE Loss: 0.011909
#     [Epoch 15/45] MSE Loss: 0.006230
#     [Epoch 20/45] MSE Loss: 0.002960
#     [Epoch 25/45] MSE Loss: 0.001356
#     [Epoch 30/45] MSE Loss: 0.000643
#     [Epoch 35/45] MSE Loss: 0.000345
#     [Epoch 40/45] MSE Loss: 0.000224
#     [Epoch 45/45] MSE Loss: 0.000176
#   Latent space shape: (50000, 15)

# ============================================================
# FINAL RESULTS (FULL DATA):
# ============================================================

# 📊 Performance Metrics:
#    Silhouette Score      : 0.3570
#    Calinski-Harabasz     : 54522.7
#    Davies-Bouldin Index  : 0.8395

# 📈 Cluster Distribution:
#    Cluster 0: 16486 samples (33.0%)
#    Cluster 1: 5821 samples (11.6%)
#    Cluster 2: 18478 samples (37.0%)
#    Cluster 3: 9215 samples (18.4%)
# (qwen) xu@LAPTOP-E2VGTV54:~/dsai4203$ 



# deep_ae_qwen3_heavy.py
# Qwen3 4096d embeddings → Deep Autoencoder → KMeans (heavy Optuna tuning, GPU only)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


# =============================
# 0. Config
# =============================
NPZ_PATH = "qwen3_8b_embeddings_full_4096d.npz"  # 改成你的 npz 路径
RANDOM_STATE = 42

# ★ 调得更“狠”一点
N_TRIALS = 100          # Optuna 试验次数更高
SUBSAMPLE = 10000      # 调参用 1w 条，可以再往上加（看显存）

# 这几个只是“默认值”做对比用，真正训练用的是 Optuna 找到的 best_params
DEFAULT_BATCH_SIZE = 1024
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_N_CLUSTERS = 10

# =============================
# 1. 强制使用 GPU
# =============================
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA 不可用，但你要求必须用 GPU。请在有 GPU 的环境里运行这个脚本。")

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True  # 让 cudnn 自动找更快的 kernel


# =============================
# 2. Deep Autoencoder 模型
# =============================
class DeepAutoencoder(nn.Module):
    def __init__(self, hidden_dims):
        """
        hidden_dims: list[int], 如 [2048, 1024, 256, 32]
        代表：4096 → 2048 → 1024 → 256 → 32 这样的编码结构
        """
        super().__init__()

        # Encoder
        encoder_layers = []
        input_dim = 4096
        for h in hidden_dims[:-1]:
            encoder_layers.append(nn.Linear(input_dim, h))
            encoder_layers.append(nn.SiLU())
            input_dim = h
        # latent 层（最后一层不激活）
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[-1]))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder：直接一层拉回 4096 维
        self.decoder = nn.Linear(hidden_dims[-1], 4096)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


# =============================
# 3. 数据加载 & 预处理
# =============================
def load_and_prepare_data(npz_path, subsample=None):
    print(f"▶ Loading data from {npz_path} ...", flush=True)
    try:
        data = np.load(npz_path, allow_pickle=True)
        X_raw = data["embeddings_4096d"].astype(np.float32)
        print(f"  Loaded shape: {X_raw.shape}", flush=True)
    except FileNotFoundError:
        print(f"❌ Error: file {npz_path} not found.", flush=True)
        return None

    # 调参用子样本
    if subsample is not None and subsample < len(X_raw):
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_raw), size=subsample, replace=False)
        X_raw = X_raw[idx]
        print(f"  Subsampled to: {X_raw.shape}", flush=True)

    # 清理 NaN / Inf
    print("▶ Checking for NaN / Inf ...", flush=True)
    nan_count = np.isnan(X_raw).sum()
    inf_count = np.isinf(X_raw).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"⚠️ Found invalid values (NaN: {nan_count}, Inf: {inf_count}). Fixing...", flush=True)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        print("→ Data is clean.", flush=True)

    # L2 归一化
    print("▶ Applying L2 normalization ...", flush=True)
    X_norm = normalize(X_raw, norm="l2")
    return X_norm


# =============================
# 4. 训练 AE（用 GPU），返回 latent
# =============================
def train_autoencoder(X_norm, hidden_dims, batch_size, epochs, learning_rate):
    """
    这里默认用 GPU + 多进程 DataLoader + pin_memory，把 GPU 利用率尽量拉高。
    """
    tensor_x = torch.from_numpy(X_norm)
    dataset = TensorDataset(tensor_x)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,      # 可按你机器调大/调小
        pin_memory=True,
    )

    model = DeepAutoencoder(hidden_dims).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler()  # 混合精度，显存更省，速度更快

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in dataloader:
            batch = batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                z, x_rec = model(batch)
                loss = criterion(x_rec, batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"    [Epoch {epoch+1}/{epochs}] MSE Loss: {avg:.6f}", flush=True)

    # 提取 latent
    model.eval()
    full_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    latents = []
    with torch.no_grad():
        for (batch,) in full_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            z, _ = model(batch)
            latents.append(z.cpu().numpy())
    X_latent = np.concatenate(latents, axis=0)
    return X_latent, model


# =============================
# 5. Optuna 目标函数（参数更细 & 更大）
# =============================
def create_objective_function(X_full):
    def objective(trial):
        # 1) 模型结构：允许更深 & 维度粒度更细
        # encoder: 4096 → h1 → h2 → h3 → latent
        hidden_dim1 = trial.suggest_int("hidden_dim1", 1024, 3072, step=256)
        hidden_dim2 = trial.suggest_int("hidden_dim2", 512, 2048, step=128)
        hidden_dim3 = trial.suggest_int("hidden_dim3", 128, 1024, step=64)
        latent_dim = trial.suggest_int("latent_dim", 10, 80, step=5)

        hidden_dims = [hidden_dim1, hidden_dim2, hidden_dim3, latent_dim]

        # 2) 训练超参：batch size 更大、epoch 更多
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 1536, 2048])
        epochs = trial.suggest_int("epochs", 25, 60)  # 更长训练
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)

        # 3) 聚类超参
        n_clusters = trial.suggest_int("n_clusters", 4, 15)

        print(
            f"\n▶ Trial {trial.number}: hidden_dims={hidden_dims}, "
            f"batch_size={batch_size}, epochs={epochs}, lr={learning_rate:.5f}, "
            f"k={n_clusters}",
            flush=True,
        )

        try:
            # 训练 AE
            X_latent, _ = train_autoencoder(
                X_full, hidden_dims, batch_size, epochs, learning_rate
            )

            # KMeans
            kmeans = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                random_state=RANDOM_STATE,
                n_init="auto",
            )
            labels = kmeans.fit_predict(X_latent)

            if len(np.unique(labels)) < 2:
                trial.set_user_attr("error", "Only 1 cluster")
                return 0.0

            # 主指标：轮廓系数
            sil = silhouette_score(X_latent, labels)

            # 辅助指标记录
            try:
                ch = calinski_harabasz_score(X_latent, labels)
                db = davies_bouldin_score(X_latent, labels)
                trial.set_user_attr("calinski_harabasz", float(ch))
                trial.set_user_attr("davies_bouldin", float(db))
                trial.set_user_attr("n_clusters_actual", int(len(np.unique(labels))))
                trial.set_user_attr("latent_dim", latent_dim)
            except Exception:
                pass

            return float(sil)

        except Exception as e:
            trial.set_user_attr("error", f"Training/clustering failed: {e}")
            return 0.0

    return objective


# =============================
# 6. 用最佳参数跑完整流程（全数据）
# =============================
def run_final_pipeline(X_full, best_params):
    print("\n" + "=" * 60)
    print("Running final pipeline with best parameters (FULL DATA):")
    print("=" * 60)

    hidden_dims = [
        best_params["hidden_dim1"],
        best_params["hidden_dim2"],
        best_params["hidden_dim3"],
        best_params["latent_dim"],
    ]
    batch_size = best_params["batch_size"]
    epochs = best_params["epochs"]
    learning_rate = best_params["learning_rate"]
    n_clusters = best_params["n_clusters"]

    print(f"\n▶ Model Architecture: {hidden_dims}")
    print(f"▶ Training: batch_size={batch_size}, epochs={epochs}, lr={learning_rate:.5f}")
    print(f"▶ Clustering: k={n_clusters}")

    X_latent, model = train_autoencoder(
        X_full, hidden_dims, batch_size, epochs, learning_rate
    )
    print(f"  Latent space shape: {X_latent.shape}")

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=RANDOM_STATE,
        n_init="auto",
    )
    labels = kmeans.fit_predict(X_latent)

    sil = silhouette_score(X_latent, labels)
    ch = calinski_harabasz_score(X_latent, labels)
    db = davies_bouldin_score(X_latent, labels)

    unique, counts = np.unique(labels, return_counts=True)
    cluster_dist = dict(zip(unique, counts))

    print("\n" + "=" * 60)
    print("FINAL RESULTS (FULL DATA):")
    print("=" * 60)
    print("\n📊 Performance Metrics:")
    print(f"   Silhouette Score      : {sil:.4f}")
    print(f"   Calinski-Harabasz     : {ch:.1f}")
    print(f"   Davies-Bouldin Index  : {db:.4f}")

    print("\n📈 Cluster Distribution:")
    for cid, cnt in sorted(cluster_dist.items()):
        pct = cnt / len(labels) * 100
        print(f"   Cluster {cid}: {cnt} samples ({pct:.1f}%)")

    return {
        "X_latent": X_latent,
        "labels": labels,
        "model": model,
        "kmeans": kmeans,
        "metrics": {
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
        },
        "params": best_params,
    }


# =============================
# 7. main
# =============================
def main():
    print("🚀 Deep Autoencoder clustering (heavy GPU) started", flush=True)

    # 全数据（用于最后 full-data 训练）
    X_full = load_and_prepare_data(NPZ_PATH, subsample=None)
    if X_full is None:
        return

    # 调参数据（子样本）
    if SUBSAMPLE is not None and SUBSAMPLE < len(X_full):
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_full), size=SUBSAMPLE, replace=False)
        X_tuning = X_full[idx]
        print(f"\n▶ Using subsample: {SUBSAMPLE} rows for Optuna tuning", flush=True)
    else:
        X_tuning = X_full

    print(f"\n▶ Starting Optuna optimization ({N_TRIALS} trials)...", flush=True)
    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="autoencoder_clustering_tuning_heavy",
    )

    objective_func = create_objective_function(X_tuning)
    study.optimize(objective_func, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS:")
    print("=" * 60)
    print(f"\nBest silhouette score (tuning subset): {study.best_value:.4f}")
    print("Best parameters found:")
    for k, v in study.best_params.items():
        print(f"  {k:25}: {v}")

    best_trial = study.best_trial
    if best_trial.user_attrs:
        print("\nBest trial details:")
        if "calinski_harabasz" in best_trial.user_attrs:
            print(f"  Calinski-Harabasz: {best_trial.user_attrs['calinski_harabasz']:.1f}")
        if "davies_bouldin" in best_trial.user_attrs:
            print(f"  Davies-Bouldin   : {best_trial.user_attrs['davies_bouldin']:.4f}")
        if "latent_dim" in best_trial.user_attrs:
            print(f"  Latent dimension : {best_trial.user_attrs['latent_dim']}")

    # 用最佳参数在 full data 上重训一遍
    result = run_final_pipeline(X_full, study.best_params)
    return result


if __name__ == "__main__":
    _ = main()
