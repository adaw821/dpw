import os
import unittest
import pandas as pd
from pathlib import Path

# 获取项目根目录和处理后的数据目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

class TestDataCleaning(unittest.TestCase):

    def test_01_output_files_exist(self):
        """校验所有预期的输出文件是否都已生成"""
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
            self.assertTrue(file_path.exists(), f"缺失文件: {file_name}")
            self.assertGreater(os.path.getsize(file_path), 0, f"文件为空: {file_name}")

    def test_02_movies_main_validity(self):
        """校验主表 movies_main 的基础数据质量"""
        file_path = PROCESSED_DIR / "movies_main.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            # 1. 校验是否有数据
            self.assertGreater(len(df), 0, "主表没有任何数据")
            # 2. 校验主键 (movie_id) 是否有空值
            self.assertEqual(df['movie_id'].isnull().sum(), 0, "movie_id 存在空值")
            # 3. 校验主键是否唯一
            self.assertTrue(df['movie_id'].is_unique, "movie_id 存在重复值")

    def test_03_ratings_validity(self):
        """校验评分表的数据逻辑"""
        file_path = PROCESSED_DIR / "ratings_clean.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            # 校验平均分是否在合理区间 [0.5, 5.0]
            invalid_ratings = df[(df['avg_user_rating'] < 0.5) | (df['avg_user_rating'] > 5.0)]
            self.assertEqual(len(invalid_ratings), 0, "存在超出 0.5-5.0 范围的异常评分")

    def test_04_credits_validity(self):
        """校验演职员表的字段完整性"""
        file_path = PROCESSED_DIR / "cast.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            # 校验关键字段是否存在
            expected_columns = ['movie_id', 'person_id', 'person_name', 'character_name', 'cast_order']
            for col in expected_columns:
                self.assertIn(col, df.columns, f"cast.csv 缺失列: {col}")

if __name__ == '__main__':
    unittest.main(verbosity=2)