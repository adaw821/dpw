from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import math
import random

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYSIS_FILE = PROJECT_ROOT / "recommendation_outputs" / "recommendation_analysis_table.parquet"
ANALYSIS_FILE_FALLBACK = PROJECT_ROOT / "data" / "processed" / "recommendation_analysis_table.parquet"
RATINGS_FILE = PROJECT_ROOT / "data" / "processed" / "final_ratings_raw.parquet"
OUTPUT_DIR = PROJECT_ROOT / "recommendation_outputs" / "ml_recommender"
TABLE_DIR = OUTPUT_DIR / "tables"

RANDOM_SEED = 42
LIKE_THRESHOLD = 4.0
MIN_CO_LIKES = 2
MAX_POSITIVE_PAIRS = 4000
MAX_NEGATIVE_PAIRS = 4000
TIME_SPLIT_QUANTILE = 0.8


@dataclass
class MLRecommendation:
    query_title_year: str
    candidate_title_year: str
    movie_id: int
    final_score: float
    content_score: float
    quality_score: float
    genre_jaccard: float
    keyword_jaccard: float
    cast_jaccard: float
    director_jaccard: float
    semantic_similarity: float
    shared_genres: str
    shared_keywords: str
    shared_cast: str
    shared_directors: str
    avg_user_rating: float
    unique_users: float


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_list(values: object) -> list[str]:
    if values is None or (isinstance(values, float) and pd.isna(values)):
        return []
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    if not isinstance(values, (list, tuple)):
        try:
            values = list(values)
        except TypeError:
            return []
    out: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def compute_quality_score(df: pd.DataFrame) -> pd.Series:
    rating_component = (df["avg_user_rating"].fillna(0) / 5.0) * 0.6
    user_component = df["unique_users"].fillna(0).apply(lambda x: min(math.log1p(x) / math.log1p(341), 1.0)) * 0.4
    return (rating_component + user_component).round(4)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_path = ANALYSIS_FILE if ANALYSIS_FILE.exists() else ANALYSIS_FILE_FALLBACK
    movies = pd.read_parquet(analysis_path)
    ratings = pd.read_parquet(RATINGS_FILE)
    movies = movies[movies["eligible_for_content_rec"]].copy()
    if "rating_date" in ratings.columns:
        ratings["rating_date"] = pd.to_datetime(ratings["rating_date"], errors="coerce")

    for col in ["genres_list", "keywords_list", "top_cast_list", "director_list"]:
        movies[col] = movies[col].apply(normalize_list)

    movies["quality_score"] = compute_quality_score(movies)
    return movies, ratings


def tokenize(text: str) -> list[str]:
    return [token for token in str(text).lower().split() if token]


def build_tfidf_vectors(df: pd.DataFrame) -> tuple[dict[int, dict[str, float]], dict[int, float]]:
    doc_freq: Counter[str] = Counter()
    doc_tokens: dict[int, Counter[str]] = {}

    for row in df.itertuples(index=False):
        movie_id = int(row.movie_id)
        tokens = tokenize(row.recommendation_text)
        counter = Counter(tokens)
        doc_tokens[movie_id] = counter
        doc_freq.update(counter.keys())

    total_docs = len(doc_tokens)
    idf = {token: math.log((1 + total_docs) / (1 + freq)) + 1.0 for token, freq in doc_freq.items()}

    tfidf_vectors: dict[int, dict[str, float]] = {}
    norms: dict[int, float] = {}
    for movie_id, counter in doc_tokens.items():
        total_terms = sum(counter.values()) or 1
        vector = {token: (count / total_terms) * idf[token] for token, count in counter.items()}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        tfidf_vectors[movie_id] = vector
        norms[movie_id] = norm

    return tfidf_vectors, norms


def cosine_similarity_sparse(left: dict[str, float], right: dict[str, float], left_norm: float, right_norm: float) -> float:
    if left_norm == 0 or right_norm == 0:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
        left_norm, right_norm = right_norm, left_norm
    dot = 0.0
    for token, value in left.items():
        if token in right:
            dot += value * right[token]
    return dot / (left_norm * right_norm) if dot else 0.0


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_movie_lookup(df: pd.DataFrame) -> dict[int, pd.Series]:
    return {int(row.movie_id): row for _, row in df.iterrows()}


def build_title_lookup(df: pd.DataFrame) -> dict[str, int]:
    return {row["title_year"]: int(row["movie_id"]) for _, row in df.iterrows()}


def build_liked_users(ratings: pd.DataFrame, valid_movie_ids: set[int]) -> dict[int, set[int]]:
    liked = ratings[(ratings["rating"] >= LIKE_THRESHOLD) & ratings["tmdbId"].isin(valid_movie_ids)].copy()
    grouped = liked.groupby("tmdbId")["userId"].agg(lambda s: set(map(int, s))).to_dict()
    return {int(movie_id): users for movie_id, users in grouped.items()}


def split_ratings_by_time(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    usable = ratings.dropna(subset=["rating_date"]).copy()
    cutoff = usable["rating_date"].quantile(TIME_SPLIT_QUANTILE)
    train_ratings = usable[usable["rating_date"] <= cutoff].copy()
    test_ratings = usable[usable["rating_date"] > cutoff].copy()
    return train_ratings, test_ratings, cutoff


def build_positive_pairs(ratings: pd.DataFrame, valid_movie_ids: set[int]) -> list[tuple[int, int]]:
    liked = ratings[(ratings["rating"] >= LIKE_THRESHOLD) & ratings["tmdbId"].isin(valid_movie_ids)].copy()
    pair_counts: Counter[tuple[int, int]] = Counter()

    for _, user_df in liked.groupby("userId"):
        movie_ids = sorted({int(x) for x in user_df["tmdbId"].tolist()})
        if len(movie_ids) < 2:
            continue
        for left, right in combinations(movie_ids, 2):
            pair_counts[(left, right)] += 1

    positive_pairs = [pair for pair, count in pair_counts.items() if count >= MIN_CO_LIKES]
    positive_pairs.sort(key=lambda pair: pair_counts[pair], reverse=True)
    return positive_pairs[:MAX_POSITIVE_PAIRS]


def build_negative_pairs(
    movie_ids: list[int],
    positive_set: set[tuple[int, int]],
    pair_count: int,
) -> list[tuple[int, int]]:
    rng = random.Random(RANDOM_SEED)
    negatives: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = pair_count * 30

    while len(negatives) < pair_count and attempts < max_attempts:
        left, right = sorted(rng.sample(movie_ids, 2))
        pair = (left, right)
        if pair not in positive_set:
            negatives.add(pair)
        attempts += 1
    return sorted(negatives)


def collaborative_target(left_users: set[int], right_users: set[int]) -> float:
    if not left_users or not right_users:
        return 0.0
    return len(left_users & right_users) / len(left_users | right_users)


def build_pair_features(
    pair: tuple[int, int],
    movie_lookup: dict[int, pd.Series],
    tfidf_vectors: dict[int, dict[str, float]],
    norms: dict[int, float],
    liked_users: dict[int, set[int]],
) -> tuple[list[float], float]:
    left_id, right_id = pair
    left = movie_lookup[left_id]
    right = movie_lookup[right_id]

    left_genres = set(left["genres_list"])
    right_genres = set(right["genres_list"])
    left_keywords = set(left["keywords_list"])
    right_keywords = set(right["keywords_list"])
    left_cast = set(left["top_cast_list"])
    right_cast = set(right["top_cast_list"])
    left_directors = set(left["director_list"])
    right_directors = set(right["director_list"])

    semantic_sim = cosine_similarity_sparse(
        tfidf_vectors[left_id],
        tfidf_vectors[right_id],
        norms[left_id],
        norms[right_id],
    )

    features = [
        jaccard(left_genres, right_genres),
        jaccard(left_keywords, right_keywords),
        jaccard(left_cast, right_cast),
        jaccard(left_directors, right_directors),
        semantic_sim,
    ]
    target = collaborative_target(liked_users.get(left_id, set()), liked_users.get(right_id, set()))
    return features, target


def fit_linear_regression(feature_rows: list[list[float]], targets: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(feature_rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    x_with_bias = np.c_[np.ones(len(x)), x]
    beta, _, _, _ = np.linalg.lstsq(x_with_bias, y, rcond=None)

    intercept = beta[0]
    raw_weights = beta[1:]
    ranking_weights = np.clip(raw_weights, 0, None)
    if ranking_weights.sum() == 0:
        ranking_weights = np.array([0.35, 0.30, 0.20, 0.15, 0.25], dtype=float)
    ranking_weights = ranking_weights / ranking_weights.sum()
    return np.array([intercept], dtype=float), raw_weights, ranking_weights


def predict_targets(feature_rows: list[list[float]], intercept: np.ndarray, raw_weights: np.ndarray) -> np.ndarray:
    x = np.asarray(feature_rows, dtype=float)
    preds = float(intercept[0]) + x @ raw_weights
    return np.clip(preds, 0.0, 1.0)


def evaluate_predictions(targets: list[float], preds: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(targets, dtype=float)
    rmse = float(np.sqrt(np.mean((preds - y_true) ** 2))) if len(y_true) else float("nan")
    mae = float(np.mean(np.abs(preds - y_true))) if len(y_true) else float("nan")
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(preds) > 0:
        corr = float(np.corrcoef(y_true, preds)[0, 1])
    else:
        corr = float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_corr": corr,
        "target_mean": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "pred_mean": float(np.mean(preds)) if len(y_true) else float("nan"),
    }


def build_pair_dataset(
    source_ratings: pd.DataFrame,
    valid_movie_ids: set[int],
    movie_lookup: dict[int, pd.Series],
    tfidf_vectors: dict[int, dict[str, float]],
    norms: dict[int, float],
    limit_positive_pairs: int,
    limit_negative_pairs: int,
) -> tuple[pd.DataFrame, list[list[float]], list[float]]:
    liked_users = build_liked_users(source_ratings, valid_movie_ids)
    positive_pairs = build_positive_pairs(source_ratings, valid_movie_ids)[:limit_positive_pairs]
    positive_set = set(positive_pairs)
    negative_pairs = build_negative_pairs(sorted(valid_movie_ids), positive_set, min(len(positive_pairs), limit_negative_pairs))
    all_pairs = positive_pairs + negative_pairs

    feature_rows: list[list[float]] = []
    targets: list[float] = []
    pair_records: list[dict[str, float | int | str]] = []

    for pair in all_pairs:
        if pair[0] not in movie_lookup or pair[1] not in movie_lookup:
            continue
        features, target = build_pair_features(pair, movie_lookup, tfidf_vectors, norms, liked_users)
        feature_rows.append(features)
        targets.append(target)
        pair_records.append(
            {
                "left_movie_id": pair[0],
                "right_movie_id": pair[1],
                "genre_jaccard": features[0],
                "keyword_jaccard": features[1],
                "cast_jaccard": features[2],
                "director_jaccard": features[3],
                "semantic_similarity": features[4],
                "collaborative_target": target,
                "pair_type": "positive" if pair in positive_set else "negative",
            }
        )
    return pd.DataFrame(pair_records), feature_rows, targets


def train_weight_model(
    df: pd.DataFrame,
    ratings: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, str], dict[int, dict[str, float]], dict[int, float]]:
    cf_df = df[df["eligible_for_cf"]].copy()
    valid_movie_ids = set(int(x) for x in cf_df["movie_id"].tolist())
    movie_lookup = build_movie_lookup(cf_df)
    tfidf_vectors, norms = build_tfidf_vectors(df)
    train_ratings, test_ratings, cutoff = split_ratings_by_time(ratings)

    training_pairs, train_features, train_targets = build_pair_dataset(
        train_ratings,
        valid_movie_ids,
        movie_lookup,
        tfidf_vectors,
        norms,
        MAX_POSITIVE_PAIRS,
        MAX_NEGATIVE_PAIRS,
    )
    test_pairs, test_features, test_targets = build_pair_dataset(
        test_ratings,
        valid_movie_ids,
        movie_lookup,
        tfidf_vectors,
        norms,
        max(MAX_POSITIVE_PAIRS // 2, 1000),
        max(MAX_NEGATIVE_PAIRS // 2, 1000),
    )

    intercept, raw_weights, ranking_weights = fit_linear_regression(train_features, train_targets)
    test_preds = predict_targets(test_features, intercept, raw_weights)
    metrics = evaluate_predictions(test_targets, test_preds)
    split_info = {
        "cutoff_date": str(cutoff),
        "train_min_date": str(train_ratings["rating_date"].min()),
        "train_max_date": str(train_ratings["rating_date"].max()),
        "test_min_date": str(test_ratings["rating_date"].min()),
        "test_max_date": str(test_ratings["rating_date"].max()),
    }
    return intercept, raw_weights, ranking_weights, training_pairs, test_pairs, metrics, split_info, tfidf_vectors, norms


def compute_content_score(
    query: pd.Series,
    candidate: pd.Series,
    tfidf_vectors: dict[int, dict[str, float]],
    norms: dict[int, float],
    weights: np.ndarray,
) -> tuple[float, list[float]]:
    q_genres = set(query["genres_list"])
    c_genres = set(candidate["genres_list"])
    q_keywords = set(query["keywords_list"])
    c_keywords = set(candidate["keywords_list"])
    q_cast = set(query["top_cast_list"])
    c_cast = set(candidate["top_cast_list"])
    q_directors = set(query["director_list"])
    c_directors = set(candidate["director_list"])

    semantic = cosine_similarity_sparse(
        tfidf_vectors[int(query["movie_id"])],
        tfidf_vectors[int(candidate["movie_id"])],
        norms[int(query["movie_id"])],
        norms[int(candidate["movie_id"])],
    )
    parts = [
        jaccard(q_genres, c_genres),
        jaccard(q_keywords, c_keywords),
        jaccard(q_cast, c_cast),
        jaccard(q_directors, c_directors),
        semantic,
    ]
    score = float(np.dot(weights, np.asarray(parts, dtype=float)))
    return score, parts


def join_values(values: Iterable[str], limit: int = 4) -> str:
    ordered = sorted(values)
    return " | ".join(ordered[:limit])


def recommend(
    df: pd.DataFrame,
    title_lookup: dict[str, int],
    tfidf_vectors: dict[int, dict[str, float]],
    norms: dict[int, float],
    weights: np.ndarray,
    query_title_year: str,
    top_n: int = 10,
    min_unique_users: int = 5,
) -> list[MLRecommendation]:
    if query_title_year not in title_lookup:
        raise KeyError(f"title_year not found: {query_title_year}")

    query_id = title_lookup[query_title_year]
    query = df[df["movie_id"] == query_id].iloc[0]

    # Speed optimization: do not score every movie in a large dataset.
    # First keep movies with enough rating coverage, then prefer movies that share
    # at least one genre with the query, and finally cap the candidate pool by popularity.
    query_genres = set(query["genres_list"])
    candidate_df = df.copy()

    candidate_df = candidate_df[
        candidate_df["unique_users"].fillna(0) >= min_unique_users
    ]

    if query_genres:
        candidate_df = candidate_df[
            candidate_df["genres_list"].apply(lambda values: len(query_genres & set(values)) > 0)
        ]

    candidate_limit = 5000
    if "popularity" in candidate_df.columns:
        candidate_df = candidate_df.sort_values("popularity", ascending=False).head(candidate_limit)
    elif "vote_count" in candidate_df.columns:
        candidate_df = candidate_df.sort_values("vote_count", ascending=False).head(candidate_limit)
    else:
        candidate_df = candidate_df.head(candidate_limit)

    recommendations: list[MLRecommendation] = []
    for _, row in candidate_df.iterrows():
        if int(row["movie_id"]) == query_id:
            continue
        if float(row["unique_users"] if pd.notna(row["unique_users"]) else 0) < min_unique_users:
            continue

        content_score, parts = compute_content_score(query, row, tfidf_vectors, norms, weights)
        if content_score <= 0:
            continue

        final_score = content_score * 0.85 + float(row["quality_score"]) * 0.15

        recommendations.append(
            MLRecommendation(
                query_title_year=query_title_year,
                candidate_title_year=row["title_year"],
                movie_id=int(row["movie_id"]),
                final_score=round(final_score, 4),
                content_score=round(content_score, 4),
                quality_score=round(float(row["quality_score"]), 4),
                genre_jaccard=round(parts[0], 4),
                keyword_jaccard=round(parts[1], 4),
                cast_jaccard=round(parts[2], 4),
                director_jaccard=round(parts[3], 4),
                semantic_similarity=round(parts[4], 4),
                shared_genres=join_values(set(query["genres_list"]) & set(row["genres_list"])),
                shared_keywords=join_values(set(query["keywords_list"]) & set(row["keywords_list"]), limit=6),
                shared_cast=join_values(set(query["top_cast_list"]) & set(row["top_cast_list"])),
                shared_directors=join_values(set(query["director_list"]) & set(row["director_list"])),
                avg_user_rating=round(float(row["avg_user_rating"]), 2) if pd.notna(row["avg_user_rating"]) else float("nan"),
                unique_users=round(float(row["unique_users"]), 0) if pd.notna(row["unique_users"]) else float("nan"),
            )
        )

    recommendations.sort(key=lambda r: (r.final_score, r.content_score, r.quality_score), reverse=True)
    return recommendations[:top_n]


def save_training_outputs(
    intercept: np.ndarray,
    raw_weights: np.ndarray,
    ranking_weights: np.ndarray,
    training_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    metrics: dict[str, float],
    split_info: dict[str, str],
) -> None:
    weight_df = pd.DataFrame(
        [
            {"feature": "intercept", "weight": round(float(intercept[0]), 6)},
            {"feature": "genre_jaccard_raw", "weight": round(float(raw_weights[0]), 6)},
            {"feature": "keyword_jaccard_raw", "weight": round(float(raw_weights[1]), 6)},
            {"feature": "cast_jaccard_raw", "weight": round(float(raw_weights[2]), 6)},
            {"feature": "director_jaccard_raw", "weight": round(float(raw_weights[3]), 6)},
            {"feature": "semantic_similarity_raw", "weight": round(float(raw_weights[4]), 6)},
            {"feature": "genre_jaccard_rank_weight", "weight": round(float(ranking_weights[0]), 6)},
            {"feature": "keyword_jaccard_rank_weight", "weight": round(float(ranking_weights[1]), 6)},
            {"feature": "cast_jaccard_rank_weight", "weight": round(float(ranking_weights[2]), 6)},
            {"feature": "director_jaccard_rank_weight", "weight": round(float(ranking_weights[3]), 6)},
            {"feature": "semantic_similarity_rank_weight", "weight": round(float(ranking_weights[4]), 6)},
        ]
    )
    weight_df.to_csv(TABLE_DIR / "learned_weights.csv", index=False, encoding="utf-8-sig")
    training_pairs.to_csv(TABLE_DIR / "training_pairs_sample.csv", index=False, encoding="utf-8-sig")
    test_pairs.to_csv(TABLE_DIR / "test_pairs_sample.csv", index=False, encoding="utf-8-sig")

    evaluation_df = pd.DataFrame(
        [
            {"metric": "rmse", "value": round(metrics["rmse"], 6)},
            {"metric": "mae", "value": round(metrics["mae"], 6)},
            {"metric": "pearson_corr", "value": round(metrics["pearson_corr"], 6) if not np.isnan(metrics["pearson_corr"]) else np.nan},
            {"metric": "target_mean", "value": round(metrics["target_mean"], 6)},
            {"metric": "pred_mean", "value": round(metrics["pred_mean"], 6)},
            {"metric": "train_pair_count", "value": len(training_pairs)},
            {"metric": "test_pair_count", "value": len(test_pairs)},
        ]
    )
    evaluation_df.to_csv(TABLE_DIR / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")

    split_df = pd.DataFrame(
        [{"name": key, "value": value} for key, value in split_info.items()]
    )
    split_df.to_csv(TABLE_DIR / "time_split_info.csv", index=False, encoding="utf-8-sig")


def save_demo_outputs(
    df: pd.DataFrame,
    title_lookup: dict[str, int],
    tfidf_vectors: dict[int, dict[str, float]],
    norms: dict[int, float],
    weights: np.ndarray,
) -> None:
    demo_titles = [
        "Toy Story (1995)",
        "The Matrix (1999)",
        "Pulp Fiction (1994)",
        "Forrest Gump (1994)",
        "The Shawshank Redemption (1994)",
    ]
    rows = []
    for title in demo_titles:
        if title not in title_lookup:
            continue
        recs = recommend(df, title_lookup, tfidf_vectors, norms, weights, title, top_n=10, min_unique_users=5)
        rows.extend([rec.__dict__ for rec in recs])
    pd.DataFrame(rows).to_csv(TABLE_DIR / "demo_ml_recommendations.csv", index=False, encoding="utf-8-sig")


def save_notes(
    intercept: np.ndarray,
    raw_weights: np.ndarray,
    ranking_weights: np.ndarray,
    training_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    metrics: dict[str, float],
    split_info: dict[str, str],
) -> None:
    positive_count = int((training_pairs["pair_type"] == "positive").sum())
    negative_count = int((training_pairs["pair_type"] == "negative").sum())
    mean_target = float(training_pairs["collaborative_target"].mean()) if not training_pairs.empty else 0.0

    lines = [
        "ML Hybrid Recommender Summary",
        "",
        "Training target:",
        "- collaborative_target = Jaccard similarity of users who rated both movies >= 4.0",
        "",
        "Time split:",
        f"- cutoff_date = {split_info['cutoff_date']}",
        f"- train range = {split_info['train_min_date']} to {split_info['train_max_date']}",
        f"- test range = {split_info['test_min_date']} to {split_info['test_max_date']}",
        "",
        "Feature set:",
        f"- genre_jaccard rank weight = {ranking_weights[0]:.4f}",
        f"- keyword_jaccard rank weight = {ranking_weights[1]:.4f}",
        f"- cast_jaccard rank weight = {ranking_weights[2]:.4f}",
        f"- director_jaccard rank weight = {ranking_weights[3]:.4f}",
        f"- semantic_similarity rank weight = {ranking_weights[4]:.4f}",
        "",
        "Model:",
        "- linear regression by numpy least squares",
        f"- intercept = {float(intercept[0]):.6f}",
        f"- raw genre coefficient = {raw_weights[0]:.6f}",
        f"- raw keyword coefficient = {raw_weights[1]:.6f}",
        f"- raw cast coefficient = {raw_weights[2]:.6f}",
        f"- raw director coefficient = {raw_weights[3]:.6f}",
        f"- raw semantic coefficient = {raw_weights[4]:.6f}",
        "",
        "Training sample:",
        f"- positive pairs = {positive_count}",
        f"- negative pairs = {negative_count}",
        f"- mean collaborative target = {mean_target:.4f}",
        "",
        "Test evaluation:",
        f"- test pair count = {len(test_pairs)}",
        f"- RMSE = {metrics['rmse']:.4f}",
        f"- MAE = {metrics['mae']:.4f}",
        f"- Pearson correlation = {metrics['pearson_corr']:.4f}" if not np.isnan(metrics["pearson_corr"]) else "- Pearson correlation = NaN",
        "",
        "Final recommendation score:",
        "- final_score = learned_content_score * 0.85 + quality_score * 0.15",
        "",
        "Interpretation:",
        "- Jaccard handles exact metadata overlap",
        "- semantic similarity uses TF-IDF cosine on recommendation_text",
        "- collaborative filtering signal is used only to learn weights, not as the main online model",
    ]
    (OUTPUT_DIR / "ml_recommender_notes.txt").write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    ensure_dirs()
    df, ratings = load_data()
    (
        intercept,
        raw_weights,
        ranking_weights,
        training_pairs,
        test_pairs,
        metrics,
        split_info,
        tfidf_vectors,
        norms,
    ) = train_weight_model(df, ratings)
    title_lookup = build_title_lookup(df)

    save_training_outputs(intercept, raw_weights, ranking_weights, training_pairs, test_pairs, metrics, split_info)
    save_demo_outputs(df, title_lookup, tfidf_vectors, norms, ranking_weights)
    save_notes(intercept, raw_weights, ranking_weights, training_pairs, test_pairs, metrics, split_info)
    print(f"Saved ML hybrid recommender outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
