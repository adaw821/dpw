import pandas as pd
import unittest

from clean import (
    INPUT_DIR, OUTPUT_DIR, LOG_FILE, DataCleanLogger, 
    clean_metadata, clean_ratings_and_links, clean_keywords, clean_credits
)
from data_clean_report import compare_folders_quality

from test_clean import TestDataCleaning 

print("start data cleaning")
if __name__ == "__main__":
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
        
    print("running unit tests")
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)

    print("create data quality report")
    final_report = compare_folders_quality(str(INPUT_DIR), str(OUTPUT_DIR))
    
    if final_report is not None:
        print(final_report.to_string()) 
        final_report.to_excel('data_report.xlsx', index=False)
        print("\nReport output successfully as: data_report.xlsx")