import pandas as pd
from data_clean_report import*

orig_folder_path =  "C:/Users/imada/Desktop/dpw/raw_data"
clean_folder_path = "C:/Users/imada/Desktop/dpw/updated_data"

final_report = compare_folders_quality(orig_folder_path, clean_folder_path)
    
if final_report is not None:
    print(final_report)
    final_report.to_excel('全局数据质量对比报告.xlsx', index=False)
    print("\n报告已成功导出为: 全局数据质量对比报告.xlsx")
