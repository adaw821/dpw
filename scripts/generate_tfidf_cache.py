from build_ml_recommender import load_data, build_tfidf_vectors
import pickle
from pathlib import Path

OUTPUT_DIR = Path("recommendation_outputs/ml_recommender")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df, ratings = load_data()

print("Building TF-IDF vectors...")
tfidf_vectors, norms = build_tfidf_vectors(df)

with open(OUTPUT_DIR / "tfidf_vectors.pkl", "wb") as f:
    pickle.dump(tfidf_vectors, f)

with open(OUTPUT_DIR / "tfidf_norms.pkl", "wb") as f:
    pickle.dump(norms, f)

print("TF-IDF cache saved.")