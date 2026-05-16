import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from page_design import add_title_background, add_custom_styles
from pathlib import Path
import xgboost as xgb

try:
    st.set_page_config(
        page_title="Film Profit Prediction System",
        page_icon="🎬",
        layout="wide"
    )
except st.errors.StreamlitAPIException:
    # 已经被 app_home.py 设置过 page_config,这里跳过即可
    pass

# 1. 添加背景
add_title_background(
    title_text="🎬 Film Profit Prediction System",
    image_path="design/background.jpg",
    height=250,
    speed=40,
    opacity=0.55,
    grayscale=True
)

# 2. 注入顺滑滑块的 CSS 样式
add_custom_styles()


# ============================================================ #
# 数据加载 + 特征工程 + 模型训练（一次性，全部缓存）
# ============================================================ #
def get_processed_data_path(filename):
    project_root = Path(__file__).parent
    return project_root / "data" / "processed" / filename


@st.cache_resource(show_spinner="Loading data & training model (one-time setup)...")
def load_and_train():
    """
    把整个 pipeline 包进一个缓存函数：
    数据加载 → 特征工程 → 训练 XGBoost
    只在容器启动时跑一次，后续所有用户交互直接复用内存里的对象。
    """
    # ---------- 加载主表 ----------
    main = pd.read_parquet(get_processed_data_path('movies_main.parquet'))
    main = main[(main['budget'] > 0) & (main['revenue'] > 0)].copy()
    main['ROI'] = main['revenue'] / main['budget']
    main['is_profitable'] = (main['ROI'] >= 2.5).astype(int)

    main['runtime'] = main['runtime'].fillna(0)
    main['runtime_bin'] = pd.cut(
        main['runtime'], bins=[0, 90, 120, 500],
        labels=['Short', 'Standard', 'Long']
    )

    # ---------- Genres ----------
    genres = pd.read_parquet(get_processed_data_path('movie_genres.parquet'))
    genres = genres[genres['genre_rank'] == 1][['movie_id', 'genre_id', 'genre_name']]
    genres = genres.rename(columns={
        'genre_id': 'primary_genre_id',
        'genre_name': 'primary_genre_name'
    })

    # ---------- Countries ----------
    countries = pd.read_parquet(get_processed_data_path('movie_countries.parquet'))
    countries = countries[countries['country_rank'] == 1][['movie_id', 'country_iso']]
    countries = countries.rename(columns={'country_iso': 'primary_country_iso'})

    # ---------- Cast (只读需要的列，节省内存/时间) ----------
    cast = pd.read_csv(
        get_processed_data_path('cast.csv'),
        usecols=['movie_id', 'cast_order', 'person_id', 'person_name']
    )
    order_0 = cast[cast['cast_order'] == 0][['movie_id', 'person_id', 'person_name']] \
        .rename(columns={'person_id': 'first_actor', 'person_name': 'first_actor_name'})
    order_1 = cast[cast['cast_order'] == 1][['movie_id', 'person_id', 'person_name']] \
        .rename(columns={'person_id': 'second_actor', 'person_name': 'second_actor_name'})
    order_2 = cast[cast['cast_order'] == 2][['movie_id', 'person_id', 'person_name']] \
        .rename(columns={'person_id': 'third_actor', 'person_name': 'third_actor_name'})

    # ---------- Crew ----------
    crew = pd.read_csv(
        get_processed_data_path('crew.csv'),
        usecols=['movie_id', 'job_title', 'person_id', 'person_name']
    )
    directors = crew[crew['job_title'] == 'Director'][['movie_id', 'person_id', 'person_name']] \
        .rename(columns={'person_id': 'director_id', 'person_name': 'director_name'})
    novels = crew[crew['job_title'] == 'Novel'][['movie_id', 'person_id', 'person_name']] \
        .rename(columns={'person_id': 'novel_id', 'person_name': 'novel_name'})
    writers = crew[crew['job_title'] == 'Writer'][['movie_id', 'person_id', 'person_name']] \
        .rename(columns={'person_id': 'writer_id', 'person_name': 'writer_name'})

    main = main.merge(genres, on='movie_id', how='left')
    main = main.merge(countries, on='movie_id', how='left')
    main = main.merge(order_0, on='movie_id', how='left')
    main = main.merge(order_1, on='movie_id', how='left')
    main = main.merge(order_2, on='movie_id', how='left')
    main = main.merge(directors, on='movie_id', how='left')
    main = main.merge(novels, on='movie_id', how='left')
    main = main.merge(writers, on='movie_id', how='left')

    id_columns = ['primary_genre_id', 'first_actor', 'second_actor',
                  'third_actor', 'director_id', 'novel_id', 'writer_id']
    main[id_columns] = main[id_columns].fillna(-1)

    director_stats = main.groupby('director_id')['is_profitable'].mean().reset_index()
    director_stats = director_stats.rename(columns={'is_profitable': 'director_win_rate'})

    writer_stats = main.groupby('writer_id')['is_profitable'].mean().reset_index()
    writer_stats = writer_stats.rename(columns={'is_profitable': 'writer_win_rate'})

    main = main.merge(director_stats, on='director_id', how='left')
    main = main.merge(writer_stats, on='writer_id', how='left')

    global_win_rate = main['is_profitable'].mean()
    main['director_win_rate'] = main['director_win_rate'].fillna(global_win_rate)
    main['writer_win_rate'] = main['writer_win_rate'].fillna(global_win_rate)

    # ---------- 训练特征 ----------
    features = [
        'budget', 'genre_count', 'country_count', 'company_count', 'runtime_bin',
        'primary_genre_id', 'director_win_rate', 'writer_win_rate'
    ]
    X_raw = main[features]
    y = main['is_profitable']
    X = pd.get_dummies(X_raw, columns=['runtime_bin', 'primary_genre_id'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)

    # ---------- 构建下拉选项（向量化，比 iterrows 快 10x+）----------
    train_cols = X_train.columns.tolist()

    available_genres_with_names = []
    genres_lookup = genres.drop_duplicates(subset=['primary_genre_id']) \
        .set_index('primary_genre_id')['primary_genre_name'].to_dict()
    for col in train_cols:
        if col.startswith('primary_genre_id_'):
            genre_id = float(col.replace('primary_genre_id_', ''))
            genre_name = genres_lookup.get(genre_id, f"Unknown ({genre_id})")
            available_genres_with_names.append((genre_id, genre_name))
    available_genres_with_names.sort(key=lambda x: x[1])

    # 导演（向量化）
    dir_df = main[main['director_id'] != -1][['director_id', 'director_name']] \
        .drop_duplicates(subset=['director_id'])
    dir_df['director_name'] = dir_df['director_name'].fillna(
        dir_df['director_id'].apply(lambda i: f"Director ({i})")
    )
    available_directors_with_names = sorted(
        list(zip(dir_df['director_id'], dir_df['director_name'])),
        key=lambda x: x[1]
    )

    # 演员（向量化）
    actor_mask = (main['first_actor'] != -1) | (main['second_actor'] != -1) | (main['third_actor'] != -1)
    actor_df = main[actor_mask][['first_actor', 'first_actor_name']] \
        .drop_duplicates(subset=['first_actor'])
    actor_df['first_actor_name'] = actor_df['first_actor_name'].fillna(
        actor_df['first_actor'].apply(lambda i: f"Actor ({i})")
    )
    available_actors_with_names = sorted(
        list(zip(actor_df['first_actor'], actor_df['first_actor_name'])),
        key=lambda x: x[1]
    )

    # 预先索引到 dict，让按钮回调里查询是 O(1)
    director_winrate_lookup = main.drop_duplicates(subset=['director_id']) \
        .set_index('director_id')['director_win_rate'].to_dict()

    # 演员历史成功率（一次性算好）
    actor_winrate = {}
    for col in ['first_actor', 'second_actor', 'third_actor']:
        grp = main.groupby(col)['is_profitable'].mean()
        for aid, val in grp.items():
            if aid != -1:
                actor_winrate.setdefault(aid, val)

    return {
        'X_train_cols': train_cols,
        'model': xgb_model,
        'global_win_rate': global_win_rate,
        'genres_options': available_genres_with_names,
        'directors_options': available_directors_with_names,
        'actors_options': available_actors_with_names,
        'director_winrate_lookup': director_winrate_lookup,
        'actor_winrate_lookup': actor_winrate,
    }


# ---- 一次加载（缓存命中后是瞬间返回）----
bundle = load_and_train()
train_cols = bundle['X_train_cols']
xgb_model = bundle['model']
global_win_rate = bundle['global_win_rate']
available_genres_with_names = bundle['genres_options']
available_directors_with_names = bundle['directors_options']
available_actors_with_names = bundle['actors_options']
director_winrate_lookup = bundle['director_winrate_lookup']
actor_winrate_lookup = bundle['actor_winrate_lookup']


# ============================================================ #
# 页面交互
# ============================================================ #
st.write("Please adjust the parameters below to view the real-time prediction results.")

col1, col2 = st.columns(2)

with col1:
    genre_options = [f"{name} ({id})" for id, name in available_genres_with_names]
    selected_genre_option = st.selectbox(
        "Primary Genre (Type to search)",
        options=genre_options,
        help="You can type to search for a specific genre."
    )
    selected_genre_parts = selected_genre_option.split('(')
    selected_genre_id = float(selected_genre_parts[-1].rstrip(')'))

    budget = st.number_input("Movie Budget ($)", min_value=100000, value=50000000, step=100000)
    genre_count = st.slider("Number of Genres", min_value=1, max_value=5, value=2, step=1)
    country_count = st.slider("Number of Countries", min_value=1, max_value=5, value=1, step=1)
    company_count = st.slider("Number of Production Companies", min_value=1, max_value=10, value=2, step=1)

with col2:
    runtime_bin = st.selectbox("Runtime Category", ['Short', 'Standard', 'Long'])

    director_options = ["Not Selected (-1)"] + [f"{name} ({id})" for id, name in available_directors_with_names]
    selected_director_option = st.selectbox(
        "Select Director (Type to search)",
        options=director_options,
        help="Type the director's name to search the database."
    )

    if selected_director_option == "Not Selected (-1)":
        selected_director = -1
        director_name = "Not Selected"
    else:
        director_parts = selected_director_option.split('(')
        director_id_str = director_parts[-1].rstrip(')')
        selected_director = int(float(director_id_str))
        director_name = director_parts[0].strip()

    actor_options = ["Not Selected (-1)"] + [f"{name} ({id})" for id, name in available_actors_with_names]
    selected_actor_option = st.selectbox(
        "Select Lead Actor (Type to search)",
        options=actor_options,
        help="Type the actor's name to search the database."
    )

    if selected_actor_option == "Not Selected (-1)":
        selected_actor = -1
        actor_name = "Not Selected"
    else:
        actor_parts = selected_actor_option.split('(')
        actor_id_str = actor_parts[-1].rstrip(')')
        selected_actor = int(float(actor_id_str))
        actor_name = actor_parts[0].strip()

    # O(1) dict 查询，比之前的 DataFrame filter 快很多
    director_win_rate = global_win_rate
    if selected_director != -1:
        director_win_rate = director_winrate_lookup.get(selected_director, global_win_rate)

    writer_win_rate = global_win_rate
    if selected_actor != -1:
        writer_win_rate = actor_winrate_lookup.get(selected_actor, global_win_rate)

    if selected_director != -1:
        st.info(f"🎬 Director {director_name} (ID: {selected_director}) Historical Success Rate: {director_win_rate:.2%}")
    if selected_actor != -1:
        st.info(f"🌟 Actor {actor_name} (ID: {selected_actor}) Historical Success Rate: {writer_win_rate:.2%}")

if st.button("Start Prediction"):
    input_dict = {
        'budget': [budget],
        'genre_count': [genre_count],
        'country_count': [country_count],
        'company_count': [company_count],
        'runtime_bin_Standard': [1 if runtime_bin == 'Standard' else 0],
        'runtime_bin_Long': [1 if runtime_bin == 'Long' else 0],
        'director_win_rate': [director_win_rate],
        'writer_win_rate': [writer_win_rate]
    }

    target_col_name = None
    for col in train_cols:
        if col.startswith('primary_genre_id_') and str(selected_genre_id) in col:
            target_col_name = col
            break

    if target_col_name:
        input_dict[target_col_name] = [1]

    input_data = pd.DataFrame(input_dict)
    input_data = input_data.reindex(columns=train_cols, fill_value=0)

    prediction = xgb_model.predict(input_data)[0]
    prob = xgb_model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 Result: Likely to be a success! (Probability: {prob:.2%})")
    else:
        st.error(f"⚠️ Result: Risk of failure. (Probability: {prob:.2%})")


# 返回首页按钮
st.markdown("---")
if st.button("🏠 Back to Home"):
    st.session_state.page = "home"
    st.rerun()
