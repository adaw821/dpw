import os
import glob
import pandas as pd
import numpy as np

def safe_read_csv(file_path):
    """尝试使用多种常见编码读取CSV，解决乱码和类型混合警告"""
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    print(f"⚠️ 无法识别 {os.path.basename(file_path)} 的编码，强制读取。")
    return pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace', low_memory=False)

def read_data(file_path):
    """根据后缀自动读取数据"""
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.csv':
        return safe_read_csv(file_path)
    elif ext.lower() == '.parquet':
        return pd.read_parquet(file_path)
    return None

def calculate_folder_metrics(folder_path):
    """计算整个文件夹内所有数据的宏观质量指标"""
    files = glob.glob(f"{folder_path}/*.csv") + glob.glob(f"{folder_path}/*.parquet")
    if not files:
        print(f"文件夹 {folder_path} 为空")
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
            
        # 将空白字符转为 NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)
        
        # 1. 累加缺失值
        cells = df.shape[0] * df.shape[1]
        missing = df.isnull().sum().sum()
        total_cells += cells
        total_missing += missing
        
        # 2. 累加重复行 (按单表内重复计算)
        rows = df.shape[0]
        duplicates = df.duplicated().sum()
        total_rows += rows
        total_duplicates += duplicates
        
        # 3. 累加异常值
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            total_numeric_cells += len(col_data)
            if col_data.std() > 0:
                z_scores = (col_data - col_data.mean()) / col_data.std()
                total_outliers += (z_scores.abs() > 3).sum()
                
    # 所有文件遍历完毕，开始计算整个文件夹的综合比率
    missing_rate = total_missing / total_cells if total_cells > 0 else 0
    duplicate_rate = total_duplicates / total_rows if total_rows > 0 else 0
    outliers_rate = total_outliers / total_numeric_cells if total_numeric_cells > 0 else 0
    
    return missing_rate, duplicate_rate, outliers_rate, total_rows

def compare_folders_quality(orig_folder, clean_folder):
    """提取文件夹名，对比旧文件夹和新文件夹的全局指标"""
    
    # 提取纯文件夹名字 (比如: 'original_data' 和 'cleaned_data')
    orig_name = os.path.basename(os.path.normpath(orig_folder))
    clean_name = os.path.basename(os.path.normpath(clean_folder))
    
    # 分别对两个文件夹进行“大盘汇总”
    orig_missing, orig_dup, orig_out, orig_rows = calculate_folder_metrics(orig_folder)
    clean_missing, clean_dup, clean_out, clean_rows = calculate_folder_metrics(clean_folder)
    
    # 动态使用文件夹名字作为表头
    report_list = [{
        '评估层级': '文件夹全局汇总',
        f'【{orig_name}】总行数': orig_rows,
        f'【{clean_name}】总行数': clean_rows,
        
        f'【{orig_name}】缺失率': f"{orig_missing:.2%}",
        f'【{clean_name}】缺失率': f"{clean_missing:.2%}",
        '缺失率改善': f"{(orig_missing - clean_missing):.2%}",
        
        f'【{orig_name}】重复率': f"{orig_dup:.2%}",
        f'【{clean_name}】重复率': f"{clean_dup:.2%}",
        
        f'【{orig_name}】异常率': f"{orig_out:.2%}",
        f'【{clean_name}】异常率': f"{clean_out:.2%}"
    }]
        
    return pd.DataFrame(report_list)


