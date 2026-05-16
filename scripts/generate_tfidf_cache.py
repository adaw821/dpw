"""
Pre-build TF-IDF cache for the recommender so the deployed app
doesn't have to recompute it on every cold start.

Run from project root:
    python -m scripts.generate_tfidf_cache
or:
    python scripts/generate_tfidf_cache.py
"""
import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from build_ml_recommender import load_data, build_tfidf_vectors

OUTPUT_DIR = PROJECT_ROOT / "recommendation_outputs" / "ml_recommender"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df, ratings = load_data()

print("Building TF-IDF vectors...")
tfidf_vectors, norms = build_tfidf_vectors(df)

with open(OUTPUT_DIR / "tfidf_vectors.pkl", "wb") as f:
    pickle.dump(tfidf_vectors, f)

with open(OUTPUT_DIR / "tfidf_norms.pkl", "wb") as f:
    pickle.dump(norms, f)

print(f"TF-IDF cache saved to: {OUTPUT_DIR}")
