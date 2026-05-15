import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

INPUT_RATINGS = BASE_DIR / 'data' / 'raw' / 'ratings_small.csv'
INPUT_LINKS = BASE_DIR / 'data' / 'raw' / 'links_small.csv'
INPUT_MOVIES = BASE_DIR / 'data' / 'processed' / 'movies_main.parquet'
OUTPUT_PATH = BASE_DIR / 'data' / 'processed' / 'final_ratings_raw.parquet'

print('Loading ratings...')
ratings = pd.read_csv(INPUT_RATINGS)

print(f'Original rows: {len(ratings)}')
print(f'Rating range: {ratings["rating"].min()} - {ratings["rating"].max()}')

print('Cleaning ratings...')
ratings = ratings[(ratings['rating'] >= 0.5) & (ratings['rating'] <= 5.0)]
ratings['rating_date'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings = ratings.sort_values('timestamp').drop_duplicates(
    subset=['userId', 'movieId'], keep='last'
)
print(f'After dedup: {len(ratings)} rows')

print('Loading links...')
links = pd.read_csv(INPUT_LINKS)

print('Merging ratings with links...')
ratings_with_tmdb = ratings.merge(links[['movieId', 'tmdbId']], on='movieId', how='inner')
ratings_with_tmdb = ratings_with_tmdb.dropna(subset=['tmdbId'])
ratings_with_tmdb['tmdbId'] = ratings_with_tmdb['tmdbId'].astype(int)
print(f'After merge: {len(ratings_with_tmdb)} rows')

print('Loading movies...')
movies = pd.read_parquet(INPUT_MOVIES)

print('Merging with movies...')
final_ratings = ratings_with_tmdb.merge(
    movies, left_on='tmdbId', right_on='movie_id', how='inner'
)
print(f'Final rows: {len(final_ratings)}')
print(f'Unique movies: {final_ratings["movie_id"].nunique()}')
print(f'Unique users: {final_ratings["userId"].nunique()}')

print(f'Saving to {OUTPUT_PATH}...')
final_ratings.to_parquet(OUTPUT_PATH, index=False)

print('Done.')