import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

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