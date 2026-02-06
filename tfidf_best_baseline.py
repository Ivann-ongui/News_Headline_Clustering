# Loading and preprocessing data...
# Total cleaned headlines: 50000

# Using Best Parameters:
#   max_features: 20000
#   ngram_range: (1, 1)
#   min_df: 7
#   max_df: 0.9442152411128398
#   lsa_components: 100
#   n_clusters: 100

# ▶ Running TF-IDF vectorization ...
# TF-IDF shape: (50000, 5434)

# ▶ Running LSA (Truncated SVD) ...
# LSA output shape: (50000, 100)

# ▶ Running KMeans clustering ...

# ===== TF-IDF + LSA Baseline Results =====
# Silhouette Score       : 0.2941
# Calinski-Harabasz      : 709 .46
# Davies-Bouldin Index   : 1.5087


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans  # ✅ 改为标准 KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import warnings
warnings.filterwarnings("ignore")

# =============================
# 1. Load and preprocess data
# =============================
print("Loading and preprocessing data...")
df = pd.read_csv("./archive/abcnews-date-text.csv", nrows=50000)
headlines = df["headline_text"].astype(str).tolist()

# 只需要停用词，不再用 punkt/word_tokenize
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    # 只保留小写字母和空格
    text = re.sub(r"[^a-z\s]", "", text.lower())
    # 简单按空格切词，避免用 word_tokenize 触发 punkt 依赖
    tokens = text.split()
    # 去停用词 & 去掉太短的词
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

cleaned = []
for t in headlines:
    c = clean_text(t)
    if c.strip():
        cleaned.append(c)

print(f"Total cleaned headlines: {len(cleaned)}")

# =============================
# 2. Apply BEST PARAMETERS (found by Optuna)
# =============================
BEST_PARAMS = {
    "max_features": 20000,
    "ngram_range": (1, 1),  # unigram → (1, 1)
    "min_df": 7,
    "max_df": 0.9442152411128398,
    "lsa_components": 100,  # from n_components
    "n_clusters": 100,
}

print("\nUsing Best Parameters:")
for k, v in BEST_PARAMS.items():
    print(f"  {k}: {v}")

# =============================
# 3. TF-IDF vectorization
# =============================
print("\n▶ Running TF-IDF vectorization ...")
tfidf = TfidfVectorizer(
    max_features=BEST_PARAMS["max_features"],
    ngram_range=BEST_PARAMS["ngram_range"],
    min_df=BEST_PARAMS["min_df"],
    max_df=BEST_PARAMS["max_df"],
)

X_tfidf = tfidf.fit_transform(cleaned)
print("TF-IDF shape:", X_tfidf.shape)

# =============================
# 4. LSA (Truncated SVD)
# =============================
print("\n▶ Running LSA (Truncated SVD) ...")
lsa = TruncatedSVD(
    n_components=BEST_PARAMS["lsa_components"],
    random_state=42
)
X_lsa = lsa.fit_transform(X_tfidf)
print("LSA output shape:", X_lsa.shape)

# =============================
# 5. KMeans clustering
# =============================
print("\n▶ Running KMeans clustering ...")
kmeans = KMeans(  
    n_clusters=BEST_PARAMS["n_clusters"],
    random_state=42,
    n_init=10,      # 显式设置（默认值，可选）
    max_iter=300    # 默认值，可选
)
labels = kmeans.fit_predict(X_lsa)

# =============================
# 6. Compute evaluation metrics
# =============================
print("\n===== TF-IDF + LSA Baseline Results =====")

sil = silhouette_score(X_lsa, labels)
ch = calinski_harabasz_score(X_lsa, labels)
db = davies_bouldin_score(X_lsa, labels)

print(f"Silhouette Score       : {sil:.4f}")
print(f"Calinski-Harabasz      : {ch:.2f}")
print(f"Davies-Bouldin Index   : {db:.4f}")

# Final summary in SAME format as embedding models:
print("\n================= Summary (TF-IDF Baseline) =================")
print("{:<18} {:>12} {:>18} {:>18}".format(
    "Model", "Silhouette", "Calinski-H", "Davies-Bouldin"
))
print("{:<18} {:>12.4f} {:>18.2f} {:>18.4f}".format(
    "TF-IDF-LSA",
    sil,
    ch,
    db
))
