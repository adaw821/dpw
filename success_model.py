import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import streamlit as st


"""feature engineering"""
main = pd.read_parquet(r'C:\Users\imada\Desktop\dpw\data\processed\movies_main.parquet')
main = main[(main['budget'] > 0) & (main['revenue'] > 0)].copy()
main['ROI'] = main['revenue'] / main['budget']
main['is_profitable'] = (main['ROI'] >= 2.5).astype(int)

main['runtime'] = main['runtime'].fillna(0)
main['runtime_bin'] = pd.cut(main['runtime'], bins=[0, 90, 120, 500], labels=['Short', 'Standard', 'Long'])

# Genres
genres = pd.read_parquet(r'C:\Users\imada\Desktop\dpw\data\processed\movie_genres.parquet')
genres = genres[genres['genre_rank'] == 1][['movie_id', 'genre_id']]
genres = genres.rename(columns={'genre_id': 'primary_genre_id'})

# Countries
countries = pd.read_parquet(r"C:\Users\imada\Desktop\dpw\data\processed\movie_countries.parquet")
countries = countries[countries['country_rank'] == 1][['movie_id', 'country_iso']]
countries = countries.rename(columns={'country_iso': 'primary_country_iso'})

# Cast
cast = pd.read_csv(r"C:\Users\imada\Desktop\dpw\data\processed\cast.csv")
order_0 = cast[cast['cast_order'] == 0][['movie_id', 'person_id']].rename(columns={'person_id': 'first_actor'})
order_1 = cast[cast['cast_order'] == 1][['movie_id', 'person_id']].rename(columns={'person_id': 'second_actor'})
order_2 = cast[cast['cast_order'] == 2][['movie_id', 'person_id']].rename(columns={'person_id': 'third_actor'})

# Crew 
crew = pd.read_csv(r"C:\Users\imada\Desktop\dpw\data\processed\crew.csv")
directors = crew[crew['job_title'] == 'Director'][['movie_id', 'person_id']].rename(columns={'person_id': 'director_id'})
novels = crew[crew['job_title'] == 'Novel'][['movie_id', 'person_id']].rename(columns={'person_id': 'novel_id'})
writers = crew[crew['job_title'] == 'Writer'][['movie_id', 'person_id']].rename(columns={'person_id': 'writer_id'})


main = main.merge(genres, on='movie_id', how='left')
main = main.merge(countries, on='movie_id', how='left')
main = main.merge(order_0, on='movie_id', how='left')
main = main.merge(order_1, on='movie_id', how='left')
main = main.merge(order_2, on='movie_id', how='left')
main = main.merge(directors, on='movie_id', how='left')
main = main.merge(novels, on='movie_id', how='left')
main = main.merge(writers, on='movie_id', how='left')

#missing rate process
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

#split data 20% is testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       
    random_state=42,   
    stratify=y           
)

print(f"training set: {X_train.shape[0]}")
print(f"testing set: {X_test.shape[0]}")

weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {weight:.2f}")

#XG_boost model initialization
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

print("\nreport")
print(classification_report(y_test, y_pred_xgb))

"""简单试了一下参数需要改 不然没有可读性"""
st.title("🎬 电影盈利预测系统 (ROI ≥ 2.5)")
st.write("调整下方参数，查看 XGBoost 模型的实时预测结果。")

# 获取训练集中所有出现过的 primary_genre_id
# 从 X_train 的列名中提取出以 primary_genre_id_ 开头的列，获取它代表的实际 ID
available_genres = []
for col in X_train.columns:
    if col.startswith('primary_genre_id_'):
        # 截取 '_' 后面的数字部分
        genre_id_str = col.replace('primary_genre_id_', '')
        available_genres.append(float(genre_id_str)) # 你的原始数据可能是浮点数

# 排序一下，让下拉菜单更好看，并补充一个提示选项
available_genres = sorted(available_genres)

# 1. 创建页面输入组件
selected_genre_id = st.selectbox("主要类型 ID (Primary Genre ID)", available_genres)
budget = st.number_input("电影预算 ($)", min_value=100000, value=50000000)
genre_count = st.slider("类型数量", 1, 5, 2)
country_count = st.slider("国家数量", 1, 5, 1)
company_count = st.slider("制作公司数量", 1, 10, 2)
runtime_bin = st.selectbox("时长分类", ['Short', 'Standard', 'Long'])
director_win_rate = st.slider("导演历史胜率", 0.0, 1.0, 0.5)
writer_win_rate = st.slider("编剧历史胜率", 0.0, 1.0, 0.5)

# 2. 点击预测按钮
if st.button("start predict"):
    
    # 将所有的输入装入字典 (注意这里改用 input_dict)
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
    
    # 处理用户选择的分类 ID (由于 get_dummies 会将列名转为浮点数字符串，比如 primary_genre_id_12.0)
    # 这一步需要精确匹配 X_train 里的列名格式
    # 我们遍历 X_train 找到对应的列
    target_col_name = None
    for col in X_train.columns:
        if col.startswith('primary_genre_id_') and str(selected_genre_id) in col:
            target_col_name = col
            break
            
    if target_col_name:
        input_dict[target_col_name] = [1]
    
    # 转换成 DataFrame
    input_data = pd.DataFrame(input_dict)
    
    # 🌟 自动对齐列名：把缺少的列补 0，并按照 X_train 的顺序排列 🌟
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
    
    # 3. 进行预测
    prediction = xgb_model.predict(input_data)[0]
    prob = xgb_model.predict_proba(input_data)[0][1]
    
    # 4. 显示结果
    if prediction == 1:
        st.success(f"result: maybe success！(possibility: {prob:.2%})")
    else:
        st.error(f"result: maybe failed (possibility: {prob:.2%})")