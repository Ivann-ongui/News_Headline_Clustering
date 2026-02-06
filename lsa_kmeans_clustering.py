import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans  # ← Changed: use KMeans instead of MiniBatchKMeans
from sklearn.metrics import silhouette_score
import optuna
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# 1. Load and preprocess data (once)
# ----------------------------
print("Loading and preprocessing data...")
df = pd.read_csv('abcnews-date-text.csv', nrows=50000)
headlines = df['headline_text'].astype(str).tolist()

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-z\s]', '', text.lower())
    tokens = [w for w in word_tokenize(text) if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

cleaned = []
for h in headlines:
    c = clean_text(h)
    if c.strip():
        cleaned.append(c)

# Keep only first 50,000 for faster tuning (adjust if needed)
MAX_SAMPLES_FOR_TUNING = 50000
if len(cleaned) > MAX_SAMPLES_FOR_TUNING:
    cleaned = cleaned[:MAX_SAMPLES_FOR_TUNING]

print(f"Using {len(cleaned)} headlines for tuning.")

# ----------------------------
# 2. Objective function for Optuna
# ----------------------------
def objective(trial):
    # --- TF-IDF parameters ---
    max_features = trial.suggest_int("max_features", 5000, 20000, step=2500)
    ngram_choice = trial.suggest_categorical("ngram", ["unigram", "bigram"])
    ngram_range = (1, 1) if ngram_choice == "unigram" else (1, 2)
    min_df = trial.suggest_int("min_df", 2, 10)
    max_df = trial.suggest_float("max_df", 0.8, 0.95)

    # Vectorize
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )
    X_tfidf = tfidf.fit_transform(cleaned)

    # --- LSA parameters ---
    n_components = trial.suggest_int("n_components", 100, 500, step=50)
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    X_lsa = lsa.fit_transform(X_tfidf)

    # --- Clustering parameters ---
    n_clusters = trial.suggest_int("n_clusters", 20, 100, step=10)

    # Cluster using standard KMeans (← Key change)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_lsa)

    # Evaluate: Silhouette Score (on a sample to speed up)
    sample_size = min(3000, X_lsa.shape[0])
    indices = np.random.choice(X_lsa.shape[0], size=sample_size, replace=False)
    score = silhouette_score(X_lsa[indices], labels[indices])

    return score

# ----------------------------
# 3. Run Optuna Study
# ----------------------------
print("Starting Optuna hyperparameter tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # Try 30 combinations

# ----------------------------
# 4. Output best parameters
# ----------------------------
print("\n✅ Best parameters found:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"\nBest Silhouette Score: {study.best_value:.4f}")

# Save best params
import json
with open("best_params_tfidf_lsa.json", "w") as f:
    json.dump(study.best_params, f)

print("\nBest parameters saved to 'best_params_tfidf_lsa.json'")

