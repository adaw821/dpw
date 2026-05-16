import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import pickle

sys.path.insert(0, str(Path(__file__).parent))
from build_ml_recommender import (
    load_data, build_tfidf_vectors, build_movie_lookup, recommend
)


try:
    st.set_page_config(
        page_title="Movie Recommendation",
        page_icon="🎬",
        layout="wide"
    )
except st.errors.StreamlitAPIException:
    # 已经被 app_home.py 设置过 page_config,这里跳过即可
    pass

@st.cache_resource
def load_config():
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        weights_config = config.get("recommender_weights", {})
        weights = np.array([
            weights_config.get("genre_jaccard", 0.2),
            weights_config.get("keyword_jaccard", 0.2),
            weights_config.get("cast_jaccard", 0.2),
            weights_config.get("director_jaccard", 0.2),
            weights_config.get("semantic_similarity", 0.2),
        ])
        weights = weights / weights.sum()  
        return weights, config
    return np.array([0.2, 0.2, 0.2, 0.2, 0.2]), {}

@st.cache_resource
def load_all_data():
    df, ratings = load_data()
    df = df[(df['genres_list'].apply(len) > 0) | (df['top_cast_list'].apply(len) > 0)]
    df['title'] = df['title'].fillna('Unknown')
    df['display_title'] = df.apply(
        lambda x: f"{x['title']} ({int(x['release_year']) if pd.notna(x['release_year']) else '?'})",
        axis=1
    )
    if 'popularity' in df.columns:
        df = df.sort_values('popularity', ascending=False)
    elif 'vote_count' in df.columns:
        df = df.sort_values('vote_count', ascending=False)
    else:
        df = df.sort_values('movie_id')
    title_lookup = {row['display_title']: int(row['movie_id']) for _, row in df.iterrows()}
    id_to_title = {int(row['movie_id']): row['display_title'] for _, row in df.iterrows()}
    tfidf_path = Path(__file__).parent / "recommendation_outputs" / "ml_recommender" / "tfidf_vectors.pkl"
    norms_path = Path(__file__).parent / "recommendation_outputs" / "ml_recommender" / "tfidf_norms.pkl"

    if tfidf_path.exists() and norms_path.exists():
        with open(tfidf_path, "rb") as f:
            tfidf_vectors = pickle.load(f)
        with open(norms_path, "rb") as f:
            norms = pickle.load(f)
    else:
        tfidf_vectors, norms = build_tfidf_vectors(df)

    movie_lookup = build_movie_lookup(df)
    weights, config = load_config()
    return df, title_lookup, id_to_title, tfidf_vectors, norms, movie_lookup, weights, config

def get_movie_info(df, movie_id, id_to_title):
    movie = df[df['movie_id'] == movie_id].iloc[0]
    def safe_list(val):
        if isinstance(val, list):
            return val
        elif pd.isna(val):
            return []
        else:
            return [str(val)] if str(val).strip() else []
    def safe_float(val, default=0.0):
        if pd.isna(val):
            return default
        try:
            return float(val)
        except:
            return default
    def safe_int(val, default=0):
        if pd.isna(val):
            return default
        try:
            return int(val)
        except:
            return default
    info = {
        'movie_id': movie_id,
        'title': id_to_title.get(movie_id, 'Unknown'),
        'display_title': id_to_title.get(movie_id, 'Unknown'),
        'year': safe_int(movie.get('release_year'), '?'),
        'genres': safe_list(movie.get('genres_list')),
        'keywords': safe_list(movie.get('keywords_list')),
        'cast': safe_list(movie.get('top_cast_list')),
        'directors': safe_list(movie.get('director_list')),
        'avg_rating': safe_float(movie.get('avg_user_rating')),
        'rating_count': safe_int(movie.get('unique_users')),
        'budget': safe_float(movie.get('budget')),
        'revenue': safe_float(movie.get('revenue')),
        'popularity': safe_float(movie.get('popularity')),
        'vote_average': safe_float(movie.get('vote_average')),
        'vote_count': safe_int(movie.get('vote_count'))
    }
    return info

def main():

    st.title("Movie Recommendation")
    st.markdown("Find movies tailored to your taste")

    with st.spinner("Loading movie database..."):
        df, title_lookup, id_to_title, tfidf_vectors, norms, movie_lookup, weights, config = load_all_data()

    settings = config.get("recommendation_settings", {})
    default_top_n = settings.get("default_top_n", 10)
    min_unique_users_config = settings.get("min_unique_users", 5)

    all_movie_titles = list(title_lookup.keys())


    with st.sidebar:
        st.markdown("### Search")
        search_term = st.text_input(
            "Movie title",
            placeholder="e.g., Inception, Parasite, Toy Story",
            label_visibility="collapsed"
        )
        selected_movie = None

        if search_term:
            filtered = [t for t in all_movie_titles if search_term.lower() in t.lower()]
            if filtered:
                if len(filtered) > 100:
                    st.info(f"{len(filtered)} movies found, please refine your search.")
                    filtered = filtered[:100]
                selected_movie = st.selectbox("Select a movie", filtered, label_visibility="collapsed")
            else:
                st.warning(f"No movie found for '{search_term}'")
        else:
            st.info("Type a movie name above")
            st.markdown("**Popular examples:**")
            for ex in all_movie_titles[:8]:
                st.caption(f"• {ex}")

        st.divider()
        st.markdown("### Settings")
        top_n = st.slider("Number of recommendations", 5, 20, default_top_n)
        min_similarity = st.slider("Minimum score (%)", 0, 50, 0)


    if selected_movie:
        movie_id = title_lookup[selected_movie]
        info = get_movie_info(df, movie_id, id_to_title)


        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Selected Movie")
            st.markdown(f"### {info['title']}")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Year:** {info['year']}")
                if info['popularity'] > 0:
                    st.markdown(f"**Popularity:** {info['popularity']:.1f}")
            with col_b:
                if info['vote_count'] > 0:
                    st.markdown(f"**Score:** {info['vote_average']:.1f} ({info['vote_count']})")

            st.markdown("---")
            if info['genres']:
                st.markdown(f"**Genres:** {', '.join(info['genres'][:5])}")
            if info['cast']:
                st.markdown(f"**Cast:** {', '.join(info['cast'][:3])}")
            if info['directors']:
                st.markdown(f"**Director:** {', '.join(info['directors'])}")

        with col2:
            st.subheader("Recommend")
            if st.button("Get Recommendations", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        raw_recs = recommend(
                            df, title_lookup, tfidf_vectors, norms, weights,
                            selected_movie, top_n=top_n, min_unique_users=min_unique_users_config
                        )

                        # The UI shows a relative Match score, so the slider should filter
                        # by the same relative score instead of the raw model score.
                        if raw_recs and min_similarity > 0:
                            max_score = max([r.final_score for r in raw_recs])
                            recs = [
                                r for r in raw_recs
                                if max_score > 0 and (r.final_score / max_score) * 100 >= min_similarity
                            ]
                        else:
                            recs = raw_recs

                        st.session_state['recommendations'] = recs
                        st.session_state['query_movie'] = info['title']
                    except Exception as e:
                        st.error(f"Error: {e}")


        if st.session_state.get('recommendations'):
            recs = st.session_state['recommendations']
            query_title = st.session_state.get('query_movie', '')
            st.divider()
            st.subheader(f"Recommendations for {query_title}")

            max_score = max([r.final_score for r in recs]) if recs else 1

            for i in range(0, len(recs), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(recs):
                        rec = recs[i + j]
                        with cols[j]:
                            with st.container(border=True):
                                match_pct = (rec.final_score / max_score) * 100 if max_score > 0 else 0
                                raw_pct = rec.final_score * 100
                                if match_pct >= 80:
                                    color_icon = "🟢"
                                elif match_pct >= 60:
                                    color_icon = "🟡"
                                else:
                                    color_icon = "🔵"

                                st.markdown(f"##### {color_icon} {rec.candidate_title_year}")
                                st.markdown(f"**Match score:** `{match_pct:.1f}%`")

                                if rec.avg_user_rating > 0:
                                    stars = "⭐" * min(5, int(rec.avg_user_rating))
                                    st.markdown(f"{stars} {rec.avg_user_rating:.1f} ({int(rec.unique_users)})")


                                with st.expander("Details"):
                                    if rec.shared_genres:
                                        st.markdown(f"**Genres:** {rec.shared_genres}")
                                    if rec.shared_keywords:
                                        st.markdown(f"**Keywords:** {rec.shared_keywords}")
                                    if rec.shared_cast:
                                        st.markdown(f"**Cast:** {rec.shared_cast}")
                                    if rec.shared_directors:
                                        st.markdown(f"**Director:** {rec.shared_directors}")

                                    st.markdown("---")
                                    st.caption(f"Raw score: {raw_pct:.1f}%")
                                    st.caption(f"Content score: {rec.content_score:.2f}")
                                    st.caption(f"Quality score: {rec.quality_score:.2f}")

                                    col_d1, col_d2 = st.columns(2)
                                    with col_d1:
                                        st.caption(f"Genre: {rec.genre_jaccard:.2f}")
                                        st.caption(f"Keyword: {rec.keyword_jaccard:.2f}")
                                    with col_d2:
                                        st.caption(f"Cast: {rec.cast_jaccard:.2f}")
                                        st.caption(f"Director: {rec.director_jaccard:.2f}")
    else:
    
        st.markdown("""
        <div style="background:white; border-radius:16px; padding:3rem 2rem; text-align:center; border:1px solid #eee;">
            <div style="font-size:1.1rem; color:#3b82f6; margin-bottom:0.5rem;">Start Exploring</div>
            <div style="color:#666;">Use the sidebar to search for a movie you love, then get personalized recommendations.</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    if st.button("💰 Go to Profit Prediction"):
        st.session_state.page = "profit"
        st.rerun()

if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = []

if __name__ == "__main__":
    main()