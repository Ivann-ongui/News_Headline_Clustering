# News Headline Clustering Project


## File Descriptions

### Baseline (Lexical Representation)

- `lsa_kmeans_clustering.py`  
  Supporting script for LSA + KMeans clustering (used during baseline testing).

---

### LLM Embedding Generation (4 Models)

- `MiniLM.py`  
  Generate embeddings using MiniLM.

- `bge_small_384d.py`  
  Generate embeddings using BGE-Small (384-dim).

- `nemotron.py`  
  Generate embeddings using NVIDIA Nemotron.

- `qwen.py`  
  Generate embeddings using Qwen3-Embedding-8B (4096-dim).


---

### Unified Comparison(table 1)

- `tfidf_best_baseline.py`  
  Final script used to report the TF-IDF + LSA + KMeans baseline scores.

- `compare_all_embeddings.py`  
  Compare all LLM embedding models using the same clustering
  metrics (Silhouette, Calinski–Harabasz, Davies–Bouldin).


---

### Qwen3-Based Clustering Pipelines (table 2)

- `UMAP.py`  
  Qwen3 → L2 Normalization → UMAP → KMeans.

- `PCA+UMAP.py`  
  Qwen3 → PCA → UMAP → KMeans (**Final optimized system**).

- `deep_ae_qwen3.py`  
  Qwen3 → Deep Autoencoder → KMeans.



---


