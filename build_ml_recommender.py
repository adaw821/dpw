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
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path("C:/Users/18314/Desktop/作业/dpw/dpw")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "recommendation_outputs" / "ml_recommender"
TABLE_DIR = OUTPUT_DIR / "tables"

RANDOM_SEED = 42
LIKE_THRESHOLD = 4.0
MIN_CO_LIKES = 2
MAX_POSITIVE_PAIRS = 4000
MAX_NEGATIVE_PAIRS = 4000
TEST_SIZE = 0.2
TOP_K_VALUES = [1, 5, 10]


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


@dataclass
class EvaluationMetrics:
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mae: float
    rmse: float
    test_size: int
    train_size: int
    num_test_pairs: int


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
    """加载数据并构建推荐系统需要的特征表"""
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)

    # 1. 加载评分聚合数据
    ratings_agg = pd.read_parquet(DATA_PROCESSED / "ratings_clean.parquet")
    print(f"评分聚合数据: {len(ratings_agg)} 部电影")

    # 2. 加载电影主表
    movies = pd.read_parquet(DATA_PROCESSED / "movies_main.parquet")
    print(f"电影主表: {len(movies)} 部")

    # 3. 合并
    df = movies.merge(ratings_agg, on='movie_id', how='left')
    df['avg_user_rating'] = df['avg_user_rating'].fillna(0)
    df['unique_users'] = df['user_rating_count'].fillna(0)

    # 4. 加载 genres
    genres_df = pd.read_parquet(DATA_PROCESSED / "movie_genres.parquet")
    genres_agg = genres_df.groupby('movie_id')['genre_name'].agg(list).reset_index()
    genres_agg.columns = ['movie_id', 'genres_list']
    df = df.merge(genres_agg, on='movie_id', how='left')
    print(f"Genres 数据: {len(genres_df)} 条, {df['genres_list'].notna().sum()} 部电影有类型")

    # 5. 加载 keywords
    keywords_file = DATA_PROCESSED / "keywords_long.csv"
    if keywords_file.exists():
        keywords_df = pd.read_csv(keywords_file)
        keywords_agg = keywords_df.groupby('movie_id')['keyword_name'].agg(list).reset_index()
        keywords_agg.columns = ['movie_id', 'keywords_list']
        df = df.merge(keywords_agg, on='movie_id', how='left')
        print(f"Keywords 数据: {len(keywords_df)} 条, {df['keywords_list'].notna().sum()} 部电影有关键词")
    else:
        df['keywords_list'] = [[] for _ in range(len(df))]

    # 6. 加载 cast
    cast_file = DATA_PROCESSED / "cast.csv"
    if cast_file.exists():
        cast_df = pd.read_csv(cast_file)
        cast_df_sorted = cast_df.sort_values(['movie_id', 'cast_order'])
        cast_agg = cast_df_sorted.groupby('movie_id')['person_name'].agg(lambda x: list(x.head(5))).reset_index()
        cast_agg.columns = ['movie_id', 'top_cast_list']
        df = df.merge(cast_agg, on='movie_id', how='left')
        print(f"Cast 数据: {len(cast_df)} 条, {df['top_cast_list'].notna().sum()} 部电影有演员")
    else:
        df['top_cast_list'] = [[] for _ in range(len(df))]

    # 7. 加载 directors
    crew_file = DATA_PROCESSED / "crew.csv"
    if crew_file.exists():
        crew_df = pd.read_csv(crew_file)
        directors_df = crew_df[crew_df['job_title'] == 'Director']
        directors_agg = directors_df.groupby('movie_id')['person_name'].agg(list).reset_index()
        directors_agg.columns = ['movie_id', 'director_list']
        df = df.merge(directors_agg, on='movie_id', how='left')
        print(f"Directors 数据: {len(directors_df)} 条, {df['director_list'].notna().sum()} 部电影有导演")
    else:
        df['director_list'] = [[] for _ in range(len(df))]

    # 8. 填充缺失值
    for col in ['genres_list', 'keywords_list', 'top_cast_list', 'director_list']:
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    # 9. 创建推荐文本
    df['recommendation_text'] = df.apply(
        lambda x: ' '.join(x['genres_list']) + ' ' +
                  ' '.join(x['keywords_list']) + ' ' +
                  ' '.join(x['top_cast_list']) + ' ' +
                  ' '.join(x['director_list']),
        axis=1
    )

    # 10. 创建标题年份
    df['title_year'] = df.apply(
        lambda x: f"{x['title']} ({int(x['release_year']) if pd.notna(x['release_year']) else 'Unknown'})",
        axis=1
    )

    # 11. 设置标志
    df['eligible_for_content_rec'] = True
    df['eligible_for_cf'] = df['unique_users'] >= 5

    # 12. 质量分数
    df['quality_score'] = compute_quality_score(df)

    print(f"\n最终分析表: {len(df)} 部电影")
    print(f"  可用于协同过滤 (≥5评分): {df['eligible_for_cf'].sum()}")

    # 13. 构建模拟的用户评分数据
    print("\n构建模拟用户评分数据（用于协同过滤训练）...")
    np.random.seed(RANDOM_SEED)
    simulated_ratings = []

    cf_movies = df[df['eligible_for_cf']].copy()
    n_users = 300

    for user_id in range(n_users):
        n_ratings = np.random.randint(30, min(80, len(cf_movies)))
        selected_movies = cf_movies.sample(n=n_ratings)
        for _, movie in selected_movies.iterrows():
            base_rating = movie['avg_user_rating']
            if base_rating == 0:
                rating = np.random.choice([3.0, 3.5, 4.0, 4.5, 5.0])
            else:
                rating = np.clip(base_rating + np.random.normal(0, 0.8), 0.5, 5.0)
            rating = round(rating * 2) / 2
            simulated_ratings.append({
                'userId': user_id,
                'tmdbId': movie['movie_id'],
                'rating': rating,
                'timestamp': 946684800
            })

    ratings = pd.DataFrame(simulated_ratings)
    ratings = ratings.drop_duplicates(subset=['userId', 'tmdbId'])

    print(
        f"模拟用户评分: {len(ratings)} 条, {ratings['userId'].nunique()} 个用户, {ratings['tmdbId'].nunique()} 部电影")

    return df, ratings


def tokenize(text: str) -> list[str]:
    return [token for token in str(text).lower().split() if token]


def build_tfidf_vectors(df: pd.DataFrame) -> tuple[dict[int, dict[str, float]], dict[int, float]]:
    print("构建 TF-IDF 向量...")
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

    print(f"  TF-IDF 向量构建完成: {len(tfidf_vectors)} 部电影")
    return tfidf_vectors, norms


def cosine_similarity_sparse(left: dict[str, float], right: dict[str, float], left_norm: float,
                             right_norm: float) -> float:
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


def fit_linear_regression(feature_rows: list[list[float]], targets: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(feature_rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    x_with_bias = np.c_[np.ones(len(x)), x]
    beta, _, _, _ = np.linalg.lstsq(x_with_bias, y, rcond=None)

    intercept = beta[0]
    weights = beta[1:]
    weights = np.clip(weights, 0, None)
    if weights.sum() == 0:
        weights = np.array([0.35, 0.30, 0.20, 0.15, 0.25], dtype=float)
    weights = weights / weights.sum()
    return np.array([intercept], dtype=float), weights


def split_movies_by_time(df: pd.DataFrame, test_size: float = 0.2) -> tuple[set[int], set[int]]:
    """按上映年份划分（用老电影训练，新电影测试）"""
    cf_movies = df[df["eligible_for_cf"]].copy()
    # 按上映年份排序
    cf_movies_sorted = cf_movies.sort_values("release_year")

    split_idx = int(len(cf_movies_sorted) * (1 - test_size))
    train_movies = set(cf_movies_sorted.iloc[:split_idx]["movie_id"].values)
    test_movies = set(cf_movies_sorted.iloc[split_idx:]["movie_id"].values)

    print(f"\n时间划分结果:")
    print(f"  训练集电影: {len(train_movies)} 部")
    print(
        f"    年份范围: {cf_movies_sorted.iloc[:split_idx]['release_year'].min():.0f} - {cf_movies_sorted.iloc[:split_idx]['release_year'].max():.0f}")
    print(f"  测试集电影: {len(test_movies)} 部")
    print(
        f"    年份范围: {cf_movies_sorted.iloc[split_idx:]['release_year'].min():.0f} - {cf_movies_sorted.iloc[split_idx:]['release_year'].max():.0f}")

    return train_movies, test_movies


def compute_ndcg(relevant_items: set[int], recommended_items: list[int], k: int) -> float:
    recommended_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_items:
            dcg += 1.0 / math.log2(i + 2)
    ideal_relevant_count = min(len(relevant_items), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant_count))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(
        test_pairs: list[tuple[int, int]],
        movie_lookup: dict[int, pd.Series],
        tfidf_vectors: dict[int, dict[str, float]],
        norms: dict[int, float],
        weights: np.ndarray,
        top_k_list: list[int] = [1, 5, 10],
) -> dict[int, tuple[float, float, float]]:
    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in top_k_list}

    for left_id, right_id in test_pairs:
        if left_id not in movie_lookup or right_id not in movie_lookup:
            continue

        query = movie_lookup[left_id]
        candidates = []
        for candidate_id, candidate in movie_lookup.items():
            if candidate_id == left_id:
                continue
            content_score, _ = compute_content_score(query, candidate, tfidf_vectors, norms, weights)
            candidates.append((candidate_id, content_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        recommended_ids = [c[0] for c in candidates[:max(top_k_list)]]

        for k in top_k_list:
            relevant = {right_id}
            recommended_k = recommended_ids[:k]
            hits = sum(1 for item in recommended_k if item in relevant)
            precision = hits / k
            recall = hits / len(relevant) if len(relevant) > 0 else 0.0
            ndcg = compute_ndcg(relevant, recommended_ids, k)
            results[k]["precision"].append(precision)
            results[k]["recall"].append(recall)
            results[k]["ndcg"].append(ndcg)

    avg_results = {}
    for k in top_k_list:
        avg_results[k] = (
            np.mean(results[k]["precision"]) if results[k]["precision"] else 0.0,
            np.mean(results[k]["recall"]) if results[k]["recall"] else 0.0,
            np.mean(results[k]["ndcg"]) if results[k]["ndcg"] else 0.0,
        )
    return avg_results


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

    recommendations: list[MLRecommendation] = []
    for _, row in df.iterrows():
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
                avg_user_rating=round(float(row["avg_user_rating"]), 2) if pd.notna(row["avg_user_rating"]) else float(
                    "nan"),
                unique_users=round(float(row["unique_users"]), 0) if pd.notna(row["unique_users"]) else float("nan"),
            )
        )

    recommendations.sort(key=lambda r: (r.final_score, r.content_score, r.quality_score), reverse=True)
    return recommendations[:top_n]


def train_weight_model_with_evaluation(
        df: pd.DataFrame,
        ratings: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[int, dict[str, float]], dict[int, float], dict[
    int, set[int]], EvaluationMetrics]:
    print("\n" + "=" * 60)
    print("训练模型...")
    print("=" * 60)

    # 使用时间划分（老电影训练，新电影测试）
    train_movies, test_movies = split_movies_by_time(df, TEST_SIZE)

    train_df = df[df["movie_id"].isin(train_movies) & df["eligible_for_cf"]].copy()
    train_ratings = ratings[ratings["tmdbId"].isin(train_movies)].copy()

    print(f"训练样本: {len(train_df)} 部电影, {len(train_ratings)} 条评分")

    movie_lookup = build_movie_lookup(train_df)
    tfidf_vectors, norms = build_tfidf_vectors(df)
    liked_users = build_liked_users(train_ratings, train_movies)

    positive_pairs = build_positive_pairs(train_ratings, train_movies)
    positive_set = set(positive_pairs)
    negative_pairs = build_negative_pairs(
        sorted(train_movies), positive_set,
        min(len(positive_pairs), MAX_NEGATIVE_PAIRS)
    )
    all_pairs = positive_pairs + negative_pairs

    print(f"正样本对: {len(positive_pairs)}, 负样本对: {len(negative_pairs)}")

    if len(positive_pairs) == 0:
        print("警告: 没有正样本对！使用默认权重")
        weights = np.array([0.35, 0.30, 0.20, 0.15, 0.25])
        weights = weights / weights.sum()
        intercept = np.array([0.0])

        all_pairs_df = pd.DataFrame()

        metrics = EvaluationMetrics(
            precision_at_k={k: 0.0 for k in TOP_K_VALUES},
            recall_at_k={k: 0.0 for k in TOP_K_VALUES},
            ndcg_at_k={k: 0.0 for k in TOP_K_VALUES},
            mae=0.0,
            rmse=0.0,
            test_size=len(test_movies),
            train_size=len(train_movies),
            num_test_pairs=0,
        )

        return intercept, weights, all_pairs_df, tfidf_vectors, norms, liked_users, metrics

    feature_rows: list[list[float]] = []
    targets: list[float] = []
    pair_records: list[dict] = []

    for pair in all_pairs:
        if pair[0] not in movie_lookup or pair[1] not in movie_lookup:
            continue
        features, target = build_pair_features(pair, movie_lookup, tfidf_vectors, norms, liked_users)
        feature_rows.append(features)
        targets.append(target)
        pair_records.append({
            "left_movie_id": pair[0],
            "right_movie_id": pair[1],
            "genre_jaccard": features[0],
            "keyword_jaccard": features[1],
            "cast_jaccard": features[2],
            "director_jaccard": features[3],
            "semantic_similarity": features[4],
            "collaborative_target": target,
            "pair_type": "positive" if pair in positive_set else "negative",
            "dataset": "train",
        })

    intercept, weights = fit_linear_regression(feature_rows, targets)
    print(f"\n学习到的权重:")
    print(f"  genre_jaccard: {weights[0]:.4f}")
    print(f"  keyword_jaccard: {weights[1]:.4f}")
    print(f"  cast_jaccard: {weights[2]:.4f}")
    print(f"  director_jaccard: {weights[3]:.4f}")
    print(f"  semantic_similarity: {weights[4]:.4f}")

    # 测试评估
    print("\n" + "=" * 60)
    print("评估模型...")
    print("=" * 60)

    test_df = df[df["movie_id"].isin(test_movies)].copy()
    test_ratings = ratings[ratings["tmdbId"].isin(test_movies)].copy()
    test_liked_users = build_liked_users(test_ratings, test_movies)
    test_movie_lookup = build_movie_lookup(test_df)
    test_positive_pairs = build_positive_pairs(test_ratings, test_movies)

    print(f"测试集正样本对: {len(test_positive_pairs)}")

    predicted_scores = []
    actual_targets = []

    for left_id, right_id in test_positive_pairs[:500]:
        if left_id not in test_movie_lookup or right_id not in test_movie_lookup:
            continue
        features, actual = build_pair_features(
            (left_id, right_id), test_movie_lookup, tfidf_vectors, norms, test_liked_users
        )
        predicted = float(np.dot(weights, np.asarray(features, dtype=float)))
        predicted_scores.append(predicted)
        actual_targets.append(actual)
        pair_records.append({
            "left_movie_id": left_id,
            "right_movie_id": right_id,
            "genre_jaccard": features[0],
            "keyword_jaccard": features[1],
            "cast_jaccard": features[2],
            "director_jaccard": features[3],
            "semantic_similarity": features[4],
            "collaborative_target": actual,
            "pair_type": "test_positive",
            "dataset": "test",
        })

    if predicted_scores:
        predicted_scores = np.array(predicted_scores)
        actual_targets = np.array(actual_targets)
        mae = np.mean(np.abs(predicted_scores - actual_targets))
        rmse = np.sqrt(np.mean((predicted_scores - actual_targets) ** 2))
    else:
        mae = rmse = 0.0

    # 推荐质量评估
    if test_positive_pairs:
        recommendation_metrics = evaluate_recommendations(
            test_positive_pairs[:200], test_movie_lookup, tfidf_vectors, norms, weights, TOP_K_VALUES,
        )
        precision_at_k = {k: v[0] for k, v in recommendation_metrics.items()}
        recall_at_k = {k: v[1] for k, v in recommendation_metrics.items()}
        ndcg_at_k = {k: v[2] for k, v in recommendation_metrics.items()}
    else:
        precision_at_k = {k: 0.0 for k in TOP_K_VALUES}
        recall_at_k = {k: 0.0 for k in TOP_K_VALUES}
        ndcg_at_k = {k: 0.0 for k in TOP_K_VALUES}

    metrics = EvaluationMetrics(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        ndcg_at_k=ndcg_at_k,
        mae=mae,
        rmse=rmse,
        test_size=len(test_movies),
        train_size=len(train_movies),
        num_test_pairs=len(test_positive_pairs),
    )

    all_pairs_df = pd.DataFrame(pair_records)
    return intercept, weights, all_pairs_df, tfidf_vectors, norms, liked_users, metrics


def save_results(intercept: np.ndarray, weights: np.ndarray, all_pairs: pd.DataFrame,
                 metrics: EvaluationMetrics, df: pd.DataFrame, title_lookup: dict,
                 tfidf_vectors: dict, norms: dict) -> None:
    # 保存权重
    weight_df = pd.DataFrame([
        {"feature": "intercept", "weight": round(float(intercept[0]), 6)},
        {"feature": "genre_jaccard", "weight": round(float(weights[0]), 6)},
        {"feature": "keyword_jaccard", "weight": round(float(weights[1]), 6)},
        {"feature": "cast_jaccard", "weight": round(float(weights[2]), 6)},
        {"feature": "director_jaccard", "weight": round(float(weights[3]), 6)},
        {"feature": "semantic_similarity", "weight": round(float(weights[4]), 6)},
    ])
    weight_df.to_csv(TABLE_DIR / "learned_weights.csv", index=False, encoding="utf-8-sig")

    # 保存评估指标
    metrics_df = pd.DataFrame([
                                  {"metric": f"precision@{k}", "value": v} for k, v in metrics.precision_at_k.items()
                              ] + [
                                  {"metric": f"recall@{k}", "value": v} for k, v in metrics.recall_at_k.items()
                              ] + [
                                  {"metric": f"ndcg@{k}", "value": v} for k, v in metrics.ndcg_at_k.items()
                              ] + [
                                  {"metric": "mae", "value": metrics.mae},
                                  {"metric": "rmse", "value": metrics.rmse},
                                  {"metric": "train_movies", "value": metrics.train_size},
                                  {"metric": "test_movies", "value": metrics.test_size},
                                  {"metric": "test_pairs", "value": metrics.num_test_pairs},
                              ])
    metrics_df.to_csv(TABLE_DIR / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")

    # 保存训练对
    if len(all_pairs) > 0:
        all_pairs.to_csv(TABLE_DIR / "training_pairs_sample.csv", index=False, encoding="utf-8-sig")

    # 生成演示推荐
    print("\n生成演示推荐...")
    movies_with_ratings = df[df['avg_user_rating'] > 0]['title_year'].head(10).tolist()

    rows = []
    for title in movies_with_ratings[:5]:
        if title in title_lookup:
            try:
                recs = recommend(df, title_lookup, tfidf_vectors, norms, weights, title, top_n=10, min_unique_users=1)
                rows.extend([rec.__dict__ for rec in recs])
                print(f"  已推荐: {title}")
            except Exception as e:
                print(f"  推荐 {title} 时出错: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(TABLE_DIR / "demo_recommendations.csv", index=False, encoding="utf-8-sig")
        print(f"已保存 {len(rows)} 条推荐结果")

    print(f"\n输出文件保存在: {OUTPUT_DIR}")


def run() -> None:
    ensure_dirs()

    df, ratings = load_data()
    title_lookup = build_title_lookup(df)

    intercept, weights, all_pairs, tfidf_vectors, norms, liked_users, metrics = train_weight_model_with_evaluation(df,
                                                                                                                   ratings)

    save_results(intercept, weights, all_pairs, metrics, df, title_lookup, tfidf_vectors, norms)

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"MAE: {metrics.mae:.4f}")
    print(f"RMSE: {metrics.rmse:.4f}")
    print(f"Precision@1: {metrics.precision_at_k.get(1, 0):.4f}")
    print(f"Precision@5: {metrics.precision_at_k.get(5, 0):.4f}")
    print(f"Precision@10: {metrics.precision_at_k.get(10, 0):.4f}")
    print(f"Recall@10: {metrics.recall_at_k.get(10, 0):.4f}")
    print(f"NDCG@10: {metrics.ndcg_at_k.get(10, 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run()