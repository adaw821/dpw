import os
import sys
import ast
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(os.getcwd())
#address
INPUT_DIR = PROJECT_ROOT / "data" / "raw"#data/raw
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"#data/processed
LOG_FILE = PROJECT_ROOT / "data_cleaning_log.txt"#cleaning log

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DataCleanLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.logs = []
        self.start_time = datetime.now()

    def log(self, action, details):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {action}: {details}"
        print(msg)
        self.logs.append(msg)

    def save(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Data Cleaning Master Log\n")
            f.write(f"Start time: {self.start_time}\n")
            f.write(f"End time: {datetime.now()}\n")
            f.write("=" * 80 + "\n")
            for log in self.logs:
                f.write(log + "\n")
        print(f"\n All logs saved to: {self.log_file}")



def safe_json_parse(json_str):
    """Json String extracter"""
    if pd.isna(json_str) or json_str in ('', '[]', '{}'):
        return []
    try:
        parsed = ast.literal_eval(json_str)
        if isinstance(parsed, (list, dict)):
            return parsed
        return []
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, (list, dict)):
                return parsed
            return []
        except (json.JSONDecodeError, TypeError):
            return []

def extract_collection_info(collection):
    """extract movie series ID and name"""
    if not collection:
        return None, None
    if isinstance(collection, list) and len(collection) > 0:
        collection = collection[0]
    if isinstance(collection, dict):
        return collection.get('id'), collection.get('name')
    return None, None

def clean_currency(col):
    """clean the currency"""
    return (col.astype(str)
            .str.replace(r'[\$,]', '', regex=True)
            .str.replace(r'^$', '0', regex=True)
            .pipe(lambda x: pd.to_numeric(x, errors='coerce'))
            .fillna(0))


def clean_metadata(logger):
    input_file = INPUT_DIR / "movies_metadata.csv"
    logger.log("---- [MODULE 1] ----", "Starting movies_metadata cleaning")
    
    if not input_file.exists():
        logger.log("ERROR", f"File not found: {input_file}")
        return

    df = pd.read_csv(input_file, low_memory=False, on_bad_lines='skip')
    logger.log("Read movies_metadata", f"Rows: {len(df)}")

    # format movie_id
    df['movie_id'] = pd.to_numeric(df['id'], errors='coerce')
    df = df.dropna(subset=['movie_id'])
    df['movie_id'] = df['movie_id'].astype(int)

    # format date
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year

    # format currency
    df['budget'] = clean_currency(df['budget'])
    df['revenue'] = clean_currency(df['revenue'])

    # turn to JSON
    df['genres_parsed'] = df['genres'].apply(safe_json_parse)
    df['countries_parsed'] = df['production_countries'].apply(safe_json_parse)
    df['companies_parsed'] = df['production_companies'].apply(safe_json_parse)
    df['collection_parsed'] = df['belongs_to_collection'].apply(safe_json_parse)
    
    collection_info = df['collection_parsed'].apply(extract_collection_info)
    df['collection_id'] = collection_info.apply(lambda x: x[0])
    df['collection_name'] = collection_info.apply(lambda x: x[1])

    # clean numerical column
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df.loc[df['runtime'] <= 0, 'runtime'] = np.nan
    for col in ['popularity', 'vote_average', 'vote_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # drop duplicate
    df = df.drop_duplicates(subset=['movie_id'], keep='first')

    # contruct Main Table
    movies_main = df[['movie_id', 'title', 'release_year', 'runtime',
                      'budget', 'revenue', 'popularity', 'vote_average',
                      'vote_count', 'collection_id', 'collection_name']].copy()
    movies_main['genre_count'] = df['genres_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    movies_main['country_count'] = df['countries_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    movies_main['company_count'] = df['companies_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)


    # process invalid years and data
    movies_main = movies_main[(movies_main['release_year'] >= 1850) & (movies_main['release_year'] <= 2030)]
    movies_main = movies_main[(movies_main['budget'] >= 0) & (movies_main['revenue'] >= 0)]

    # contruct subtable: Genres
    genre_records = [{'movie_id': row['movie_id'], 'genre_name': g.get('name'), 'genre_id': g.get('id'), 'genre_rank': rank}
                     for _, row in df.iterrows() if isinstance(row['genres_parsed'], list)
                     for rank, g in enumerate(row['genres_parsed'], 1) if isinstance(g, dict) and g.get('name')]
    movie_genres = pd.DataFrame(genre_records) if genre_records else pd.DataFrame(columns=['movie_id', 'genre_name', 'genre_id', 'genre_rank'])

    # contruct subtable: Countries
    country_records = [{'movie_id': row['movie_id'], 'country_iso': c.get('iso_3166_1'), 'country_name': c.get('name'), 'country_rank': rank}
                       for _, row in df.iterrows() if isinstance(row['countries_parsed'], list)
                       for rank, c in enumerate(row['countries_parsed'], 1) if isinstance(c, dict) and c.get('name')]
    movie_countries = pd.DataFrame(country_records) if country_records else pd.DataFrame(columns=['movie_id', 'country_iso', 'country_name', 'country_rank'])

    # contruct subtable: Companies
    company_records = [{'movie_id': row['movie_id'], 'company_id': comp.get('id'), 'company_name': comp.get('name'), 'company_rank': rank}
                       for _, row in df.iterrows() if isinstance(row['companies_parsed'], list)
                       for rank, comp in enumerate(row['companies_parsed'], 1) if isinstance(comp, dict) and comp.get('name')]
    movie_companies = pd.DataFrame(company_records) if company_records else pd.DataFrame(columns=['movie_id', 'company_id', 'company_name', 'company_rank'])

    # Parquet
    movies_main.to_parquet(OUTPUT_DIR / "movies_main.parquet", index=False)
    movie_genres.to_parquet(OUTPUT_DIR / "movie_genres.parquet", index=False)
    movie_countries.to_parquet(OUTPUT_DIR / "movie_countries.parquet", index=False)
    movie_companies.to_parquet(OUTPUT_DIR / "movie_companies.parquet", index=False)
    logger.log("Metadata Success", "Saved movies_main and 3 related parquet files")



def clean_ratings_and_links(logger):
    logger.log("---- [MODULE 2] ----", "Starting ratings & links cleaning")
    ratings_file = INPUT_DIR / "ratings_small.csv"
    links_file = INPUT_DIR / "links_small.csv"

    if not ratings_file.exists() or not links_file.exists():
        logger.log("ERROR", "ratings_small.csv or links_small.csv not found")
        return

    ratings = pd.read_csv(ratings_file)
    links = pd.read_csv(links_file)

    # drop duplicates
    ratings = ratings[(ratings['rating'] >= 0.5) & (ratings['rating'] <= 5.0)]
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings = ratings.sort_values('timestamp').drop_duplicates(subset=['userId', 'movieId'], keep='last')

    # merge tmdbId
    ratings_with_tmdb = ratings.merge(links[['movieId', 'tmdbId']], on='movieId', how='inner').dropna(subset=['tmdbId'])
    ratings_with_tmdb['tmdbId'] = ratings_with_tmdb['tmdbId'].astype(int)

    # aggregation
    ratings_agg = ratings_with_tmdb.groupby('tmdbId')['rating'].agg(['mean', 'count']).reset_index()
    ratings_agg.columns = ['movie_id', 'avg_user_rating', 'user_rating_count']
    ratings_agg['avg_user_rating'] = ratings_agg['avg_user_rating'].round(2)

    ratings_agg.to_parquet(OUTPUT_DIR / "ratings_clean.parquet", index=False)
    logger.log("Ratings Success", f"Saved ratings_clean.parquet with {len(ratings_agg)} rows")


def clean_keywords(logger):
    logger.log("---- [MODULE 3] ----", "Starting keywords cleaning")
    input_file = INPUT_DIR / "keywords.csv"

    if not input_file.exists():
        logger.log("ERROR", "keywords.csv not found")
        return

    try:
        df = pd.read_csv(input_file, header=None, names=['id', 'keywords_str'], encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, header=None, names=['id', 'keywords_str'], encoding='latin-1')

    df['keywords_parsed'] = df['keywords_str'].apply(safe_json_parse)

    rows = [{'movie_id': row['id'], 'keyword_id': k['id'], 'keyword_name': k['name']}
            for _, row in df.iterrows() if isinstance(row['keywords_parsed'], list)
            for k in row['keywords_parsed'] if isinstance(k, dict) and 'id' in k and 'name' in k]

    keywords_long_df = pd.DataFrame(rows, columns=['movie_id', 'keyword_id', 'keyword_name'])
    keywords_long_df.to_csv(OUTPUT_DIR / "keywords_long.csv", index=False)
    logger.log("Keywords Success", f"Saved keywords_long.csv with {len(keywords_long_df)} rows")


def clean_credits(logger):
    logger.log("---- [MODULE 4] ----", "Starting credits cleaning")
    input_file = INPUT_DIR / "credits.csv"

    if not input_file.exists():
        logger.log("ERROR", "credits.csv not found")
        return

    df_credits = pd.read_csv(input_file, low_memory=False)

    # process cast for top 5 data
    cast_rows = []
    crew_rows = []

    for _, row in df_credits.iterrows():
        movie_id = row['id']
        
        # analyze Cast
        cast_list = safe_json_parse(row['cast'])
        for member in cast_list[:5]:
            if isinstance(member, dict) and all(k in member for k in ['id', 'name', 'character', 'order']):
                cast_rows.append({
                    'movie_id': movie_id, 'person_id': member['id'],
                    'person_name': member['name'], 'character_name': member['character'],
                    'cast_order': member['order']
                })
        
        # analyze Crew
        crew_list = safe_json_parse(row['crew'])
        for member in crew_list:
            if isinstance(member, dict) and all(k in member for k in ['id', 'name', 'job', 'department']):
                job = member['job'].lower()
                dept = member['department'].lower()
                
                if (dept == 'directing' and job == 'director') or \
                   (dept == 'writing' and job in ['screenplay', 'writer', 'story', 'novel', 'author', 'book']):
                    crew_rows.append({
                        'movie_id': movie_id, 'person_id': member['id'],
                        'person_name': member['name'], 'job_title': member['job'],
                        'department': member['department']
                    })

    pd.DataFrame(cast_rows).to_csv(OUTPUT_DIR / "cast.csv", index=False)
    pd.DataFrame(crew_rows).to_csv(OUTPUT_DIR / "crew.csv", index=False)
    logger.log("Credits Success", f"Saved cast.csv ({len(cast_rows)} rows) and crew.csv ({len(crew_rows)} rows)")



if __name__ == "__main__":
    print(f"Starting Master Data Cleaning Pipeline...")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}\n")
    
    global_logger = DataCleanLogger(LOG_FILE)
    
    try:
        clean_metadata(global_logger)
        clean_ratings_and_links(global_logger)
        clean_keywords(global_logger)
        clean_credits(global_logger)
    except Exception as e:
        global_logger.log("CRITICAL ERROR", str(e))
    finally:
        global_logger.save()