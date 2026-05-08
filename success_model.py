import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from page_design import add_title_background, add_custom_styles
from pathlib import Path
import xgboost as xgb

st.set_page_config(
    page_title="Film Profit Prediction System",
    page_icon="🎬",
    layout="wide"
)

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

# --- 数据加载与模型训练 ---
def get_processed_data_path(filename):
    project_root = Path(__file__).parent
    return project_root / "data" / "processed" / filename

# feature engineering
main = pd.read_parquet(get_processed_data_path('movies_main.parquet'))
main = main[(main['budget'] > 0) & (main['revenue'] > 0)].copy()
main['ROI'] = main['revenue'] / main['budget']
main['is_profitable'] = (main['ROI'] >= 2.5).astype(int)

main['runtime'] = main['runtime'].fillna(0)
main['runtime_bin'] = pd.cut(main['runtime'], bins=[0, 90, 120, 500], labels=['Short', 'Standard', 'Long'])

# Genres
genres = pd.read_parquet(get_processed_data_path('movie_genres.parquet'))
genres = genres[genres['genre_rank'] == 1][['movie_id', 'genre_id', 'genre_name']]
genres = genres.rename(columns={'genre_id': 'primary_genre_id', 'genre_name': 'primary_genre_name'})

# Countries
countries = pd.read_parquet(get_processed_data_path('movie_countries.parquet'))
countries = countries[countries['country_rank'] == 1][['movie_id', 'country_iso']]
countries = countries.rename(columns={'country_iso': 'primary_country_iso'})

# Cast
cast = pd.read_csv(get_processed_data_path('cast.csv'))
order_0 = cast[cast['cast_order'] == 0][['movie_id', 'person_id', 'person_name']].rename(columns={'person_id': 'first_actor', 'person_name': 'first_actor_name'})
order_1 = cast[cast['cast_order'] == 1][['movie_id', 'person_id', 'person_name']].rename(columns={'person_id': 'second_actor', 'person_name': 'second_actor_name'})
order_2 = cast[cast['cast_order'] == 2][['movie_id', 'person_id', 'person_name']].rename(columns={'person_id': 'third_actor', 'person_name': 'third_actor_name'})

# Crew 
crew = pd.read_csv(get_processed_data_path('crew.csv'))
directors = crew[crew['job_title'] == 'Director'][['movie_id', 'person_id', 'person_name']].rename(columns={'person_id': 'director_id', 'person_name': 'director_name'})
novels = crew[crew['job_title'] == 'Novel'][['movie_id', 'person_id', 'person_name']].rename(columns={'person_id': 'novel_id', 'person_name': 'novel_name'})
writers = crew[crew['job_title'] == 'Writer'][['movie_id', 'person_id', 'person_name']].rename(columns={'person_id': 'writer_id', 'person_name': 'writer_name'})

main = main.merge(genres, on='movie_id', how='left')
main = main.merge(countries, on='movie_id', how='left')
main = main.merge(order_0, on='movie_id', how='left')
main = main.merge(order_1, on='movie_id', how='left')
main = main.merge(order_2, on='movie_id', how='left')
main = main.merge(directors, on='movie_id', how='left')
main = main.merge(novels, on='movie_id', how='left')
main = main.merge(writers, on='movie_id', how='left')

# missing rate process
id_columns = ['primary_genre_id', 'first_actor', 'second_actor', 'third_actor', 'director_id', 'novel_id', 'writer_id']
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

# add features
features = [
    'budget', 'genre_count', 'country_count', 'company_count', 'runtime_bin',
    'primary_genre_id', 'director_win_rate', 'writer_win_rate'
]

X_raw = main[features]
y = main['is_profitable']

X = pd.get_dummies(X_raw, columns=['runtime_bin', 'primary_genre_id'], drop_first=True)

# split data 20% is testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       
    random_state=42,   
    stratify=y           
)

# XG_boost model initialization
weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=200,          
    learning_rate=0.05,       
    max_depth=5,               
    scale_pos_weight=weight,   
    random_state=42,
    eval_metric='logloss'     
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# --- 页面交互部分 ---
st.write("Please adjust the parameters below to view the real-time prediction results.")

# 获取训练集中所有出现过的 primary_genre_id 和名称
available_genres_with_names = []
for col in X_train.columns:
    if col.startswith('primary_genre_id_'):
        genre_id_str = col.replace('primary_genre_id_', '')
        genre_id = float(genre_id_str)
        genre_name = genres[genres['primary_genre_id'] == genre_id]['primary_genre_name'].iloc[0] if len(genres[genres['primary_genre_id'] == genre_id]) > 0 else f"Unknown Genre ({genre_id})"
        available_genres_with_names.append((genre_id, genre_name))
available_genres_with_names.sort(key=lambda x: x[1]) # 按名称排序方便搜索

# 获取所有可用的导演和演员名称及ID
available_directors_with_names = []
for _, row in main[main['director_id'] != -1].drop_duplicates(subset=['director_id']).iterrows():
    if pd.notna(row['director_name']):
        available_directors_with_names.append((row['director_id'], row['director_name']))
    else:
        available_directors_with_names.append((row['director_id'], f"Director ({row['director_id']})"))
# 按名字排序
available_directors_with_names.sort(key=lambda x: x[1])

available_actors_with_names = []
for _, row in main[(main['first_actor'] != -1) | (main['second_actor'] != -1) | (main['third_actor'] != -1)].drop_duplicates(subset=['first_actor']).iterrows():
    if pd.notna(row['first_actor_name']):
        available_actors_with_names.append((row['first_actor'], row['first_actor_name']))
    else:
        available_actors_with_names.append((row['first_actor'], f"Actor ({row['first_actor']})"))
# 按名字排序
available_actors_with_names.sort(key=lambda x: x[1])

# 1. 创建页面输入组件
col1, col2 = st.columns(2)

with col1:
    # 类型选择 - 支持搜索
    genre_options = [f"{name} ({id})" for id, name in available_genres_with_names]
    selected_genre_option = st.selectbox(
        "Primary Genre (Type to search)", 
        options=genre_options,
        help="You can type to search for a specific genre."
    )
    # 解析选中的类型ID
    selected_genre_parts = selected_genre_option.split('(')
    selected_genre_id = float(selected_genre_parts[-1].rstrip(')'))
    
    budget = st.number_input("Movie Budget ($)", min_value=100000, value=50000000, step=100000)
    
    genre_count = st.slider("Number of Genres", min_value=1, max_value=5, value=2, step=1)
    country_count = st.slider("Number of Countries", min_value=1, max_value=5, value=1, step=1)
    company_count = st.slider("Number of Production Companies", min_value=1, max_value=10, value=2, step=1)

with col2:
    runtime_bin = st.selectbox("Runtime Category", ['Short', 'Standard', 'Long'])
    
    # 导演选择 - 支持搜索
    director_options = ["Not Selected (-1)"] + [f"{name} ({id})" for id, name in available_directors_with_names]
    selected_director_option = st.selectbox(
        "Select Director (Type to search)", 
        options=director_options,
        help="Type the director's name to search the database."
    )
    
    # 解析选中的导演ID
    if selected_director_option == "Not Selected (-1)":
        selected_director = -1
        director_name = "Not Selected"
    else:
        director_parts = selected_director_option.split('(')
        director_id_str = director_parts[-1].rstrip(')')
        selected_director = int(float(director_id_str))
        director_name = director_parts[0].strip()
    
    # 演员选择 - 支持搜索
    actor_options = ["Not Selected (-1)"] + [f"{name} ({id})" for id, name in available_actors_with_names]
    selected_actor_option = st.selectbox(
        "Select Lead Actor (Type to search)", 
        options=actor_options,
        help="Type the actor's name to search the database."
    )
    
    # 解析选中的演员ID
    if selected_actor_option == "Not Selected (-1)":
        selected_actor = -1
        actor_name = "Not Selected"
    else:
        actor_parts = selected_actor_option.split('(')
        actor_id_str = actor_parts[-1].rstrip(')')
        selected_actor = int(float(actor_id_str))
        actor_name = actor_parts[0].strip()
    
    # 根据选择的导演和演员获取历史成功率
    director_win_rate = 0.5
    if selected_director != -1:
        director_data = main[main['director_id'] == selected_director]
        if not director_data.empty:
            director_win_rate = director_data['director_win_rate'].iloc[0] if pd.notna(director_data['director_win_rate'].iloc[0]) else global_win_rate
        else:
            director_win_rate = global_win_rate
    
    writer_win_rate = 0.5
    if selected_actor != -1:
        actor_data = main[(main['first_actor'] == selected_actor) | (main['second_actor'] == selected_actor) | (main['third_actor'] == selected_actor)]
        if not actor_data.empty:
            actor_win_rate = actor_data['is_profitable'].mean()
            writer_win_rate = actor_win_rate
        else:
            writer_win_rate = global_win_rate

    # 显示所选人员的历史数据
    if selected_director != -1:
        st.info(f"🎬 Director {director_name} (ID: {selected_director}) Historical Success Rate: {director_win_rate:.2%}")
    if selected_actor != -1:
        st.info(f"🌟 Actor {actor_name} (ID: {selected_actor}) Historical Success Rate: {writer_win_rate:.2%}")

# 2. 点击预测按钮
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
    for col in X_train.columns:
        if col.startswith('primary_genre_id_') and str(selected_genre_id) in col:
            target_col_name = col
            break
            
    if target_col_name:
        input_dict[target_col_name] = [1]
    
    input_data = pd.DataFrame(input_dict)
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
    
    prediction = xgb_model.predict(input_data)[0]
    prob = xgb_model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.success(f"🎉 Result: Likely to be a success! (Probability: {prob:.2%})")
    else:
        st.error(f"⚠️ Result: Risk of failure. (Probability: {prob:.2%})")
