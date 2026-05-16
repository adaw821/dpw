# Movie Data Analysis System

An end-to-end movie analytics project built on the **TMDB Movie Dataset** with two interactive Streamlit apps:

1. **Profit Prediction** — XGBoost classifier that predicts whether a movie will be profitable (ROI ≥ 2.5).
2. **Smart Recommendation** — Hybrid content-based recommender using TF-IDF semantic similarity + Jaccard overlap on genres / keywords / cast / director, blended with a quality score.

> **Live demo:** https://8x7ejjvm9jggxuzh2mkhwp.streamlit.app/).

---

## Project Structure

```
dpw/
├── app.py                              # Streamlit Cloud entrypoint (downloads data, then hands off)
├── app_home.py                         # Homepage / page router
├── success_model.py                    # Profit-prediction Streamlit page
├── recommend_model.py                  # Recommendation Streamlit page
├── page_design.py                      # Shared UI helpers (background, custom CSS)
├── build_ml_recommender.py             # Recommender core: TF-IDF, Jaccard, scoring
│
├── pipeline/                           # Data cleaning pipeline
│   ├── clean.py                        # Cleaning logic for the 5 raw TMDB tables
│   ├── clean_ratings_for_recommender.py  # Build final_ratings_raw.parquet
│   ├── data_clean_report.py            # Data quality report (missing / duplicate / outlier)
│   ├── test_clean.py                   # Unit tests for the cleaned tables
│   └── run_cleaning.py                 # Master entrypoint: clean → test → report
│
├── analysis/                           # EDA & visualization (one-off scripts)
│   ├── success_eda.py                  # Profit / ROI plots
│   └── recommender_eda.py              # Recommender data exploration
│
├── scripts/                            # Utility scripts
│   └── generate_tfidf_cache.py         # Pre-build TF-IDF pickle cache
│
├── data/
│   ├── raw/                            # Place raw TMDB CSVs here (gitignored)
│   └── processed/                      # Cleaned parquet/csv outputs (gitignored)
│
├── recommendation_outputs/
│   └── ml_recommender/
│       ├── tfidf_vectors.pkl           # Cached TF-IDF vectors (committed)
│       └── tfidf_norms.pkl             # Cached document norms (committed)
│
├── design/background.jpg               # Header background image
├── plots/                              # Profit-EDA figures
├── plots_recommend/                    # Recommender-EDA figures
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Pinned Python version (3.11)
├── README.md                           # This file
└── README.zh.md                        # Chinese version
```

---

## Quick Start (Local)

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

pip install -r requirements.txt
```

### 2. Prepare the data

Place these raw TMDB files into `data/raw/`:

```
movies_metadata.csv
ratings_small.csv
links_small.csv
keywords.csv
credits.csv
```

Then run the full cleaning pipeline:

```bash
python -m pipeline.run_cleaning
```

This produces all the cleaned tables in `data/processed/` plus `data_cleaning_log.txt` and `data_report.xlsx`.

To produce the recommender-specific ratings table:

```bash
python pipeline/clean_ratings_for_recommender.py
```

Run the recommender EDA to generate `recommendation_analysis_table.parquet`, which the recommendation app **requires** (otherwise the page raises `FileNotFoundError`):

```bash
python analysis/recommender_eda.py
```

> This writes `recommendation_analysis_table.parquet` / `.csv`, summary tables, correlation heatmaps, etc. into `recommendation_outputs/`. Requires `matplotlib`, `seaborn`, and `statsmodels` (already pinned in `requirements.txt`).

(Optional) Run the profit-model EDA, which renders the figures under `plots/`:

```bash
python analysis/success_eda.py
```

To pre-build the TF-IDF cache (recommended; the app will rebuild on the fly if missing):

```bash
python -m scripts.generate_tfidf_cache
```

### 3. Launch the app

```bash
streamlit run app_home.py
```

Open the URL Streamlit prints (usually <http://localhost:8501>).

---

## Cloud Deployment (Streamlit Community Cloud)
we deployed the local web to the cloud for easy viewing.
So you can click this link to see the web page rather than running above command on local.

```bash
https://8x7ejjvm9jggxuzh2mkhwp.streamlit.app/
```

## Profit Prediction Model

**File:** `success_model.py`

- **Target:** `is_profitable = (revenue / budget ≥ 2.5)`
- **Features:** `budget`, `genre_count`, `country_count`, `company_count`, `runtime_bin`, `primary_genre_id`, `director_win_rate`, `writer_win_rate`
- **Model:** `XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, scale_pos_weight=…)`
- **Caching:** the entire feature engineering + training pipeline is wrapped in `@st.cache_resource`, so it runs **only once per container lifetime**.

---

## Recommendation Model

**File:** `recommend_model.py`, core in `build_ml_recommender.py`

- **Content score = weighted sum of:**
  - Genre Jaccard
  - Keyword Jaccard
  - Cast Jaccard
  - Director Jaccard
  - TF-IDF cosine similarity over `recommendation_text`
- **Quality score = 0.6 × (avg_rating / 5) + 0.4 × log-scaled user count**
- **Final score = α · content + β · quality**
- Default weights live in `config.json` (or fall back to equal weights).
- Configurable in the sidebar: number of recommendations (5–20) and minimum match-score filter.

---

## Tech Stack

- **Python 3.11** (pinned via `runtime.txt`)
- **Streamlit ≥ 1.32** — UI
- **pandas / numpy / pyarrow** — data
- **scikit-learn** — `train_test_split`
- **xgboost ≥ 2.0** — profit classifier
- **gdown** — Google Drive bootstrap
- **matplotlib / seaborn / scipy** — EDA plots (only required to run `analysis/`)

---

## License

Internal project — for academic / coursework use.
