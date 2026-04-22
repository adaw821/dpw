import pandas as pd
import glob

def get_data_quality_report(filepath):
    """计算缺失率 异常值 重复数据"""
    df = pd.read_csv(filepath)

    try:
        missing_rate = df.isnull().mean() * 100
        duplicate_rate = df.duplicated().mean()

        numeric_df = df.select_dtypes(include=['numbers'])
        z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std()
        outliers_rate = (z_scores.abs() > 3).mean()
    except Exception as e:
        return f"报错信息：{str(e)}"
def file_lists(folder_path):
    print("正在查看文件夹")
    try:
        files = glob.glob(f"{folder_path}/*.csv")
    except Exception as e:
        return f"报错信息：{str(e)}"
