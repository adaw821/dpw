import os
import glob
import pandas as pd
import numpy as np

def safe_read_csv(file_path):
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    print(f"cannot recognize {os.path.basename(file_path)}")
    return pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace', low_memory=False)

def read_data(file_path):
    """read by extension"""
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.csv':
        return safe_read_csv(file_path)
    elif ext.lower() == '.parquet':
        return pd.read_parquet(file_path)
    return None

def calculate_folder_metrics(folder_path):
    """calculate data based on folders"""
    files = glob.glob(f"{folder_path}/*.csv") + glob.glob(f"{folder_path}/*.parquet")
    if not files:
        print(f"folder {folder_path} is empty")
        return 0, 0, 0, 0

    total_cells = 0
    total_missing = 0
    total_rows = 0
    total_duplicates = 0
    total_numeric_cells = 0
    total_outliers = 0
    
    for f in files:
        df = read_data(f)
        if df is None or df.empty:
            continue
            
        # space turn to NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)
        
        # 1. plus missing value 
        cells = df.shape[0] * df.shape[1]
        missing = df.isnull().sum().sum()
        total_cells += cells
        total_missing += missing
        
        # 2. plus duplicated rows
        rows = df.shape[0]
        duplicates = df.duplicated().sum()
        total_rows += rows
        total_duplicates += duplicates
        
        # 3. plus outliers
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            total_numeric_cells += len(col_data)
            if col_data.std() > 0:
                z_scores = (col_data - col_data.mean()) / col_data.std()
                total_outliers += (z_scores.abs() > 3).sum()
                
    # calcukate total rate
    missing_rate = total_missing / total_cells if total_cells > 0 else 0
    duplicate_rate = total_duplicates / total_rows if total_rows > 0 else 0
    outliers_rate = total_outliers / total_numeric_cells if total_numeric_cells > 0 else 0
    
    return missing_rate, duplicate_rate, outliers_rate, total_rows

def compare_folders_quality(orig_folder, clean_folder):

    orig_name = os.path.basename(os.path.normpath(orig_folder))
    clean_name = os.path.basename(os.path.normpath(clean_folder))
    
    orig_missing, orig_dup, orig_out, orig_rows = calculate_folder_metrics(orig_folder)
    clean_missing, clean_dup, clean_out, clean_rows = calculate_folder_metrics(clean_folder)

    report_list = [{
        f'【{orig_name}】totoal rows': orig_rows,
        f'【{clean_name}】total rows': clean_rows,
        
        f'【{orig_name}】missing rate': f"{orig_missing:.2%}",
        f'【{clean_name}】missing rate': f"{clean_missing:.2%}",
        'missing rate improvement': f"{(orig_missing - clean_missing):.2%}",
        
        f'【{orig_name}】duplicate rate': f"{orig_dup:.2%}",
        f'【{clean_name}】duplicate rate': f"{clean_dup:.2%}",
        
        f'【{orig_name}】outlier rate': f"{orig_out:.2%}",
        f'【{clean_name}】outlier rate': f"{clean_out:.2%}"
    }]
        
    return pd.DataFrame(report_list)


