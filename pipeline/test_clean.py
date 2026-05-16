import os
import unittest
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

class TestDataCleaning(unittest.TestCase):

    def test_01_output_files_exist(self):
        """check whether all the expected files output successfully"""
        expected_files = [
            "movies_main.parquet",
            "movie_genres.parquet",
            "movie_countries.parquet",
            "movie_companies.parquet",
            "ratings_clean.parquet",
            "keywords_long.csv",
            "cast.csv",
            "crew.csv"
        ]
        for file_name in expected_files:
            file_path = PROCESSED_DIR / file_name
            self.assertTrue(file_path.exists(), f"loss file: {file_name}")
            self.assertGreater(os.path.getsize(file_path), 0, f"file is empty: {file_name}")

    def test_02_movies_main_validity(self):
        """check movies_main's data quality"""
        file_path = PROCESSED_DIR / "movies_main.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self.assertGreater(len(df), 0, "movies_main dont have any data")
            self.assertEqual(df['movie_id'].isnull().sum(), 0, "movie_id exist empty value")
            self.assertTrue(df['movie_id'].is_unique, "movie_id exist duplicates value")

    def test_03_ratings_validity(self):
        """check numerical logic"""
        file_path = PROCESSED_DIR / "ratings_clean.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            # check average range whether is valid
            invalid_ratings = df[(df['avg_user_rating'] < 0.5) | (df['avg_user_rating'] > 5.0)]
            self.assertEqual(len(invalid_ratings), 0, "exist invalid rate out of 0.5-5.0")

    def test_04_credits_validity(self):
        """check credits validity"""
        file_path = PROCESSED_DIR / "cast.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            expected_columns = ['movie_id', 'person_id', 'person_name', 'character_name', 'cast_order']
            for col in expected_columns:
                self.assertIn(col, df.columns, f"cast.csv loss column: {col}")

    def test_05_movies_business_logic(self):
        """check movies_main financial and date logic"""
        file_path = PROCESSED_DIR / "movies_main.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            
            # years validity check
            valid_years = df['release_year'].dropna()
            invalid_years = valid_years[(valid_years < 1850) | (valid_years > 2030)]
            self.assertEqual(len(invalid_years), 0, f"exist invalid release_year: {invalid_years.unique()}")
            
            # budget and revenue cant be negative
            self.assertTrue((df['budget'] >= 0).all(), "budget contains negative values")
            self.assertTrue((df['revenue'] >= 0).all(), "revenue contains negative values")

    def test_06_crew_filter_logic(self):
        """check if crew.csv only contains directing and writing departments"""
        file_path = PROCESSED_DIR / "crew.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            
            actual_depts = set(df['department'].str.lower().dropna().unique())
            
            # based on logic in clean.py
            expected_depts = {'directing', 'writing'}
            
            self.assertTrue(actual_depts.issubset(expected_depts), 
                            f"crew.csv contains unexpected departments: {actual_depts - expected_depts}")

if __name__ == '__main__':
    unittest.main(verbosity=2)