from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "recommendation_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "movies": pd.read_parquet(PROCESSED_DIR / "movies_main.parquet"),
        "genres": pd.read_parquet(PROCESSED_DIR / "movie_genres.parquet"),
        "countries": pd.read_parquet(PROCESSED_DIR / "movie_countries.parquet"),
        "companies": pd.read_parquet(PROCESSED_DIR / "movie_companies.parquet"),
        "ratings_agg": pd.read_parquet(PROCESSED_DIR / "ratings_clean.parquet"),
        "ratings_raw": pd.read_parquet(PROCESSED_DIR / "final_ratings_raw.parquet"),
        "cast": pd.read_csv(PROCESSED_DIR / "cast.csv"),
        "crew": pd.read_csv(PROCESSED_DIR / "crew.csv"),
        "keywords": pd.read_csv(PROCESSED_DIR / "keywords_long.csv"),
    }


def aggregate_primary(df: pd.DataFrame, rank_col: str, value_col: str, out_col: str) -> pd.DataFrame:
    ranked = df.sort_values(rank_col)
    primary = ranked.groupby("movie_id").first().reset_index()
    primary = primary[["movie_id", value_col]].copy()
    primary.columns = ["movie_id", out_col]
    return primary


def aggregate_list(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    grouped = (
        df.dropna(subset=[value_col])
        .groupby("movie_id")[value_col]
        .agg(list)
        .reset_index()
    )
    grouped.columns = ["movie_id", out_col]
    return grouped


def build_recommendation_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    movies = inputs["movies"].copy()
    genres = inputs["genres"].copy()
    countries = inputs["countries"].copy()
    companies = inputs["companies"].copy()
    ratings_agg = inputs["ratings_agg"].copy()
    ratings_raw = inputs["ratings_raw"].copy()
    cast = inputs["cast"].copy()
    crew = inputs["crew"].copy()
    keywords = inputs["keywords"].copy()

    genre_primary = aggregate_primary(genres, "genre_rank", "genre_name", "primary_genre")
    genre_list = aggregate_list(genres, "genre_name", "genres_list")
    country_primary = aggregate_primary(countries, "country_rank", "country_name", "primary_country")
    company_primary = aggregate_primary(companies, "company_rank", "company_name", "primary_company")

    cast_top5 = cast.sort_values(["movie_id", "cast_order"]).groupby("movie_id").head(5)
    cast_list = aggregate_list(cast_top5, "person_name", "top_cast_list")

    directors = crew[crew["job_title"].str.lower() == "director"].copy()
    director_list = aggregate_list(directors, "person_name", "director_list")

    writer_jobs = {"screenplay", "writer", "story", "novel", "author", "book"}
    writers = crew[crew["job_title"].str.lower().isin(writer_jobs)].copy()
    writer_list = aggregate_list(writers, "person_name", "writer_list")

    keyword_list = aggregate_list(keywords, "keyword_name", "keywords_list")

    interactions = (
        ratings_raw.groupby("tmdbId")
        .agg(
            unique_users=("userId", "nunique"),
            raw_rating_count=("rating", "count"),
        )
        .reset_index()
    )
    interactions.columns = ["movie_id", "unique_users", "raw_rating_count"]

    rec_df = movies.merge(genre_primary, on="movie_id", how="left")
    rec_df = rec_df.merge(genre_list, on="movie_id", how="left")
    rec_df = rec_df.merge(country_primary, on="movie_id", how="left")
    rec_df = rec_df.merge(company_primary, on="movie_id", how="left")
    rec_df = rec_df.merge(cast_list, on="movie_id", how="left")
    rec_df = rec_df.merge(director_list, on="movie_id", how="left")
    rec_df = rec_df.merge(writer_list, on="movie_id", how="left")
    rec_df = rec_df.merge(keyword_list, on="movie_id", how="left")
    rec_df = rec_df.merge(ratings_agg, on="movie_id", how="left")
    rec_df = rec_df.merge(interactions, on="movie_id", how="left")

    for col in ["genres_list", "top_cast_list", "director_list", "writer_list", "keywords_list"]:
        rec_df[col] = rec_df[col].apply(lambda x: x if isinstance(x, list) else [])

    for col in ["primary_genre", "primary_country", "primary_company"]:
        rec_df[col] = rec_df[col].fillna("Unknown")

    rec_df["genre_count"] = rec_df["genres_list"].apply(len)
    rec_df["keyword_count"] = rec_df["keywords_list"].apply(len)
    rec_df["cast_count"] = rec_df["top_cast_list"].apply(len)
    rec_df["director_count"] = rec_df["director_list"].apply(len)
    rec_df["writer_count"] = rec_df["writer_list"].apply(len)

    rec_df["metadata_nonempty_count"] = (
        (rec_df["genre_count"] > 0).astype(int)
        + (rec_df["keyword_count"] > 0).astype(int)
        + (rec_df["cast_count"] > 0).astype(int)
        + (rec_df["director_count"] > 0).astype(int)
    )

    rec_df["has_enough_metadata"] = rec_df["metadata_nonempty_count"] >= 3
    rec_df["has_enough_ratings"] = rec_df["unique_users"].fillna(0) >= 5
    rec_df["eligible_for_content_rec"] = rec_df["has_enough_metadata"]
    rec_df["eligible_for_cf"] = rec_df["has_enough_metadata"] & rec_df["has_enough_ratings"]

    rec_df["recommendation_text"] = rec_df.apply(build_recommendation_text, axis=1)
    rec_df["title_year"] = rec_df["title"].fillna("") + " (" + rec_df["release_year"].fillna(0).astype(int).astype(str) + ")"

    return rec_df


def build_recommendation_text(row: pd.Series) -> str:
    parts = [
        row.get("title", "") or "",
        row.get("primary_genre", "") or "",
        " ".join(row.get("genres_list", [])),
        " ".join(row.get("keywords_list", [])),
        " ".join(row.get("top_cast_list", [])),
        " ".join(row.get("director_list", [])),
        " ".join(row.get("writer_list", [])),
    ]
    return " ".join(part.strip().replace(" ", "_").lower() for part in parts if str(part).strip())


def flatten_counter(list_series: pd.Series) -> Counter:
    counter: Counter = Counter()
    for values in list_series:
        if isinstance(values, list):
            counter.update(values)
    return counter


def save_top_frequency_tables(rec_df: pd.DataFrame) -> None:
    tables = {
        "top_genres.csv": Counter(rec_df["primary_genre"].dropna()),
        "top_keywords.csv": flatten_counter(rec_df["keywords_list"]),
        "top_cast.csv": flatten_counter(rec_df["top_cast_list"]),
        "top_directors.csv": flatten_counter(rec_df["director_list"]),
    }
    for file_name, counter in tables.items():
        pd.DataFrame(counter.most_common(30), columns=["value", "count"]).to_csv(
            TABLE_DIR / file_name, index=False, encoding="utf-8-sig"
        )


def save_duplicate_title_table(rec_df: pd.DataFrame) -> None:
    duplicates = (
        rec_df.groupby("title")
        .agg(movie_count=("movie_id", "nunique"), years=("release_year", lambda s: sorted(s.dropna().unique().tolist())[:10]))
        .reset_index()
    )
    duplicates = duplicates[duplicates["movie_count"] > 1].sort_values("movie_count", ascending=False)
    duplicates.to_csv(TABLE_DIR / "duplicate_titles.csv", index=False, encoding="utf-8-sig")


def save_summary_tables(rec_df: pd.DataFrame) -> None:
    coverage = pd.DataFrame(
        [
            {"feature": "genres_list", "nonempty_rate": (rec_df["genre_count"] > 0).mean()},
            {"feature": "keywords_list", "nonempty_rate": (rec_df["keyword_count"] > 0).mean()},
            {"feature": "top_cast_list", "nonempty_rate": (rec_df["cast_count"] > 0).mean()},
            {"feature": "director_list", "nonempty_rate": (rec_df["director_count"] > 0).mean()},
            {"feature": "avg_user_rating", "nonempty_rate": rec_df["avg_user_rating"].notna().mean()},
            {"feature": "unique_users>=5", "nonempty_rate": (rec_df["unique_users"].fillna(0) >= 5).mean()},
        ]
    )
    coverage.to_csv(TABLE_DIR / "coverage_summary.csv", index=False, encoding="utf-8-sig")

    counts = rec_df[[
        "genre_count",
        "keyword_count",
        "cast_count",
        "director_count",
        "writer_count",
        "unique_users",
        "raw_rating_count",
    ]].describe().transpose()
    counts.to_csv(TABLE_DIR / "feature_count_summary.csv", encoding="utf-8-sig")

    eligibility = pd.DataFrame(
        [
            {"segment": "all_movies", "count": len(rec_df)},
            {"segment": "content_rec_eligible", "count": int(rec_df["eligible_for_content_rec"].sum())},
            {"segment": "cf_eligible", "count": int(rec_df["eligible_for_cf"].sum())},
        ]
    )
    eligibility.to_csv(TABLE_DIR / "eligibility_summary.csv", index=False, encoding="utf-8-sig")

    genre_summary = (
        rec_df.groupby("primary_genre")
        .agg(
            movie_count=("movie_id", "size"),
            rated_movie_count=("avg_user_rating", lambda s: s.notna().sum()),
            mean_user_rating=("avg_user_rating", "mean"),
            median_user_rating=("avg_user_rating", "median"),
            mean_unique_users=("unique_users", "mean"),
            median_unique_users=("unique_users", "median"),
            mean_popularity=("popularity", "mean"),
            mean_vote_average=("vote_average", "mean"),
        )
        .sort_values("movie_count", ascending=False)
        .reset_index()
    )
    genre_summary.to_csv(TABLE_DIR / "genre_performance_summary.csv", index=False, encoding="utf-8-sig")

    year_summary = (
        rec_df.dropna(subset=["release_year"])
        .groupby("release_year")
        .agg(
            movie_count=("movie_id", "size"),
            mean_user_rating=("avg_user_rating", "mean"),
            mean_unique_users=("unique_users", "mean"),
            mean_popularity=("popularity", "mean"),
            mean_vote_average=("vote_average", "mean"),
        )
        .reset_index()
        .sort_values("release_year")
    )
    year_summary.to_csv(TABLE_DIR / "year_trend_summary.csv", index=False, encoding="utf-8-sig")

    corr_cols = [
        "release_year",
        "runtime",
        "popularity",
        "vote_average",
        "vote_count",
        "avg_user_rating",
        "unique_users",
        "genre_count",
        "keyword_count",
        "cast_count",
        "director_count",
        "metadata_nonempty_count",
    ]
    corr = rec_df[corr_cols].corr(numeric_only=True)
    corr.to_csv(TABLE_DIR / "numeric_correlation_matrix.csv", encoding="utf-8-sig")


def make_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 1.1
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["grid.color"] = "#d9d9d9"
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["grid.alpha"] = 0.7


def save_barplot(df: pd.DataFrame, x: str, y: str, title: str, out_name: str) -> None:
    plt.figure()
    sns.barplot(data=df, x=x, y=y, color="#4C78A8")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / out_name, dpi=160)
    plt.close()


def save_histplot(series: pd.Series, title: str, xlabel: str, out_name: str) -> None:
    plt.figure()
    sns.histplot(series.dropna(), bins=30, color="#59A14F")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / out_name, dpi=160)
    plt.close()


def save_scatterplot(rec_df: pd.DataFrame) -> None:
    plot_df = rec_df[(rec_df["keyword_count"] > 0) & rec_df["unique_users"].notna()].copy()
    plt.figure()
    sns.scatterplot(
        data=plot_df,
        x="keyword_count",
        y="unique_users",
        hue="primary_genre",
        alpha=0.6,
        legend=False,
    )
    plt.title("Keyword Count vs Unique Users")
    plt.xlabel("Keyword count")
    plt.ylabel("Unique users")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "keyword_count_vs_unique_users.png", dpi=160)
    plt.close()


def save_correlation_heatmap(rec_df: pd.DataFrame) -> None:
    cols = [
        "runtime",
        "popularity",
        "vote_average",
        "vote_count",
        "avg_user_rating",
        "unique_users",
        "genre_count",
        "keyword_count",
        "cast_count",
        "director_count",
        "metadata_nonempty_count",
    ]
    corr = rec_df[cols].corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", square=True, linewidths=0.5)
    plt.title("Correlation Heatmap of Recommendation Features")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "numeric_correlation_heatmap.png", dpi=160)
    plt.close()


def save_regression_plots(rec_df: pd.DataFrame) -> None:
    plots = [
        (
            rec_df.dropna(subset=["popularity", "unique_users"]).query("unique_users > 0"),
            "popularity",
            "unique_users",
            "Popularity vs Unique Users",
            "popularity_vs_unique_users_regplot.png",
        ),
        (
            rec_df.dropna(subset=["vote_average", "avg_user_rating"]),
            "vote_average",
            "avg_user_rating",
            "TMDB Vote Average vs MovieLens Average User Rating",
            "vote_average_vs_avg_user_rating_regplot.png",
        ),
        (
            rec_df.dropna(subset=["vote_count", "unique_users"]).query("vote_count > 0 and unique_users > 0"),
            "vote_count",
            "unique_users",
            "TMDB Vote Count vs MovieLens Unique Users",
            "vote_count_vs_unique_users_regplot.png",
        ),
    ]

    for plot_df, x_col, y_col, title, file_name in plots:
        if plot_df.empty:
            continue
        plt.figure()
        sns.regplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            scatter_kws={"alpha": 0.25, "s": 18, "color": "#4C78A8"},
            line_kws={"color": "#E45756", "linewidth": 2},
            lowess=True,
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / file_name, dpi=160)
        plt.close()


def save_genre_comparison_plots(rec_df: pd.DataFrame) -> None:
    genre_df = (
        rec_df.dropna(subset=["avg_user_rating", "primary_genre"])
        .groupby("primary_genre")
        .size()
        .reset_index(name="movie_count")
    )
    top_genres = genre_df[genre_df["movie_count"] >= 80].sort_values("movie_count", ascending=False).head(10)["primary_genre"]
    plot_df = rec_df[rec_df["primary_genre"].isin(top_genres)].copy()

    if not plot_df.empty:
        plt.figure(figsize=(12, 7))
        sns.boxplot(
            data=plot_df,
            x="primary_genre",
            y="avg_user_rating",
            order=top_genres.tolist(),
            color="#72B7B2",
            showfliers=False,
        )
        plt.title("Average User Rating by Primary Genre")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "avg_user_rating_by_genre_boxplot.png", dpi=160)
        plt.close()

    unique_user_df = (
        rec_df.dropna(subset=["unique_users", "primary_genre"])
        .groupby("primary_genre")
        .size()
        .reset_index(name="movie_count")
    )
    top_genres_users = unique_user_df[unique_user_df["movie_count"] >= 80].sort_values("movie_count", ascending=False).head(10)["primary_genre"]
    plot_users = rec_df[rec_df["primary_genre"].isin(top_genres_users)].copy()
    if not plot_users.empty:
        plt.figure(figsize=(12, 7))
        sns.violinplot(
            data=plot_users,
            x="primary_genre",
            y="unique_users",
            order=top_genres_users.tolist(),
            color="#F28E2B",
            cut=0,
        )
        plt.yscale("log")
        plt.title("Unique Users by Primary Genre")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "unique_users_by_genre_violin.png", dpi=160)
        plt.close()


def save_year_trend_plots(rec_df: pd.DataFrame) -> None:
    year_df = (
        rec_df.dropna(subset=["release_year"])
        .groupby("release_year")
        .agg(
            movie_count=("movie_id", "size"),
            mean_user_rating=("avg_user_rating", "mean"),
            mean_unique_users=("unique_users", "mean"),
        )
        .reset_index()
    )
    year_df = year_df[year_df["release_year"] >= 1980].copy()
    year_df["mean_user_rating_smooth"] = year_df["mean_user_rating"].rolling(5, min_periods=1).mean()
    year_df["mean_unique_users_smooth"] = year_df["mean_unique_users"].rolling(5, min_periods=1).mean()
    year_df["mean_user_rating_std"] = year_df["mean_user_rating"].rolling(5, min_periods=1).std().fillna(0)
    year_df["mean_unique_users_std"] = year_df["mean_unique_users"].rolling(5, min_periods=1).std().fillna(0)

    if not year_df.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        x = year_df["release_year"].to_numpy()
        y = year_df["mean_user_rating_smooth"].to_numpy()
        band = year_df["mean_user_rating_std"].to_numpy()
        ax.fill_between(x, y - band, y + band, color="#4C78A8", alpha=0.16, linewidth=0)
        ax.plot(x, y, color="#2F5D8A", linewidth=2.8, solid_capstyle="round", label="5-year moving average")
        ax.axhline(np.nanmean(y), color="#7f7f7f", linestyle="--", linewidth=1.1, label="overall mean")
        peak_idx = int(np.nanargmax(y))
        ax.scatter(x[peak_idx], y[peak_idx], color="#2F5D8A", s=28, zorder=3)
        ax.annotate(
            f"Peak: {int(x[peak_idx])}",
            xy=(x[peak_idx], y[peak_idx]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            color="#2F5D8A",
        )
        ax.set_title("5-Year Smoothed Average User Rating by Release Year", pad=10)
        ax.set_xlabel("Release year")
        ax.set_ylabel("Average user rating")
        ax.legend(frameon=False, loc="lower left")
        ax.margins(x=0.01)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "yearly_avg_user_rating_trend.png", dpi=220)
        plt.close()

        fig, ax = plt.subplots(figsize=(11, 6))
        x = year_df["release_year"].to_numpy()
        y = year_df["mean_unique_users_smooth"].to_numpy()
        band = year_df["mean_unique_users_std"].to_numpy()
        ax.fill_between(x, np.maximum(y - band, 0), y + band, color="#E45756", alpha=0.16, linewidth=0)
        ax.plot(x, y, color="#B33A3A", linewidth=2.8, solid_capstyle="round", label="5-year moving average")
        ax.axhline(np.nanmean(y), color="#7f7f7f", linestyle="--", linewidth=1.1, label="overall mean")
        peak_idx = int(np.nanargmax(y))
        ax.scatter(x[peak_idx], y[peak_idx], color="#B33A3A", s=28, zorder=3)
        ax.annotate(
            f"Peak: {int(x[peak_idx])}",
            xy=(x[peak_idx], y[peak_idx]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            color="#B33A3A",
        )
        ax.set_title("5-Year Smoothed Unique Users by Release Year", pad=10)
        ax.set_xlabel("Release year")
        ax.set_ylabel("Average unique users")
        ax.legend(frameon=False, loc="upper left")
        ax.margins(x=0.01)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "yearly_unique_users_trend.png", dpi=220)
        plt.close()


def save_metadata_richness_plot(rec_df: pd.DataFrame) -> None:
    plot_df = rec_df.dropna(subset=["avg_user_rating"]).copy()
    if plot_df.empty:
        return
    plt.figure()
    sns.boxplot(
        data=plot_df,
        x="metadata_nonempty_count",
        y="avg_user_rating",
        color="#B279A2",
        showfliers=False,
    )
    plt.title("Metadata Richness vs Average User Rating")
    plt.xlabel("Number of non-empty metadata groups")
    plt.ylabel("Average user rating")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "metadata_richness_vs_avg_user_rating.png", dpi=160)
    plt.close()


def save_rating_distributions(rec_df: pd.DataFrame) -> None:
    save_histplot(rec_df["avg_user_rating"], "Average User Rating Distribution", "Average user rating", "avg_user_rating_distribution.png")
    save_histplot(rec_df["unique_users"], "Unique Users per Movie", "Unique users", "unique_users_distribution.png")


def save_feature_count_distributions(rec_df: pd.DataFrame) -> None:
    for col, title in [
        ("genre_count", "Genre Count per Movie"),
        ("keyword_count", "Keyword Count per Movie"),
        ("cast_count", "Top Cast Count per Movie"),
        ("director_count", "Director Count per Movie"),
    ]:
        save_histplot(rec_df[col], title, col, f"{col}_distribution.png")


def save_top_feature_plots() -> None:
    for csv_name, title, out_name in [
        ("top_genres.csv", "Top Primary Genres", "top_primary_genres.png"),
        ("top_keywords.csv", "Top Keywords", "top_keywords.png"),
        ("top_cast.csv", "Top Cast Appearances", "top_cast.png"),
        ("top_directors.csv", "Top Directors", "top_directors.png"),
    ]:
        df = pd.read_csv(TABLE_DIR / csv_name).head(15)
        save_barplot(df, "count", "value", title, out_name)


def save_eda_note(rec_df: pd.DataFrame) -> None:
    content_count = int(rec_df["eligible_for_content_rec"].sum())
    cf_count = int(rec_df["eligible_for_cf"].sum())
    total = len(rec_df)
    corr = rec_df[["avg_user_rating", "vote_average", "unique_users", "vote_count", "popularity", "keyword_count"]].corr(numeric_only=True)
    rating_vote_corr = corr.loc["avg_user_rating", "vote_average"] if "avg_user_rating" in corr.index else np.nan
    user_vote_count_corr = corr.loc["unique_users", "vote_count"] if "unique_users" in corr.index else np.nan
    user_pop_corr = corr.loc["unique_users", "popularity"] if "unique_users" in corr.index else np.nan

    genre_summary = (
        rec_df.groupby("primary_genre")
        .agg(movie_count=("movie_id", "size"), mean_user_rating=("avg_user_rating", "mean"))
        .query("movie_count >= 80")
        .sort_values("mean_user_rating", ascending=False)
        .reset_index()
    )
    top_genre_line = "N/A"
    if not genre_summary.empty:
        top_genre = genre_summary.iloc[0]
        top_genre_line = f"{top_genre['primary_genre']} ({top_genre['mean_user_rating']:.2f})"

    lines = [
        "Recommendation EDA Notes",
        f"Total movies in analysis table: {total}",
        f"Movies eligible for content-based recommendation: {content_count}",
        f"Movies eligible for collaborative filtering baseline: {cf_count}",
        "",
        "Suggested recommendation path:",
    ]
    if cf_count >= 2000:
        lines.append("- Build content-based recommendation first, then add collaborative filtering as a second model.")
    else:
        lines.append("- Prioritize content-based recommendation. Collaborative filtering is possible but coverage may be weak.")
    lines.extend(
        [
            "",
            "Relationship findings:",
            f"- TMDB vote average and MovieLens average user rating move together moderately to strongly (corr={rating_vote_corr:.2f}).",
            f"- TMDB vote count and MovieLens unique users are strongly aligned (corr={user_vote_count_corr:.2f}).",
            f"- Popularity also has a positive relationship with MovieLens unique users (corr={user_pop_corr:.2f}).",
            f"- Among common genres, the highest mean MovieLens user rating is currently: {top_genre_line}.",
            "",
            "Recommended input fields for content-based recommendation:",
            "- primary_genre",
            "- genres_list",
            "- keywords_list",
            "- top_cast_list",
            "- director_list",
            "",
            "Suggested filtering before recommendation:",
            "- keep movies with at least 3 non-empty metadata groups",
            "- for collaborative filtering experiments, keep movies with at least 5 unique users",
        ]
    )
    (OUTPUT_DIR / "eda_notes.txt").write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    ensure_dirs()
    make_plot_style()

    inputs = load_inputs()
    rec_df = build_recommendation_table(inputs)

    rec_df.to_csv(OUTPUT_DIR / "recommendation_analysis_table.csv", index=False, encoding="utf-8-sig")
    rec_df.to_parquet(OUTPUT_DIR / "recommendation_analysis_table.parquet", index=False)

    save_summary_tables(rec_df)
    save_top_frequency_tables(rec_df)
    save_duplicate_title_table(rec_df)
    save_rating_distributions(rec_df)
    save_feature_count_distributions(rec_df)
    save_scatterplot(rec_df)
    save_correlation_heatmap(rec_df)
    save_regression_plots(rec_df)
    save_genre_comparison_plots(rec_df)
    save_year_trend_plots(rec_df)
    save_metadata_richness_plot(rec_df)
    save_top_feature_plots()
    save_eda_note(rec_df)

    print("Saved recommendation analysis table and EDA outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    run()
