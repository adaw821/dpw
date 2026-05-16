"""
Master pipeline: data cleaning + unit tests + quality report.

Run from project root:
    python -m pipeline.run_cleaning
"""
import sys
import unittest
from pathlib import Path

# Allow `python pipeline/run_cleaning.py` direct invocation as well
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.clean import (
    INPUT_DIR, OUTPUT_DIR, LOG_FILE, DataCleanLogger,
    clean_metadata, clean_ratings_and_links, clean_keywords, clean_credits,
)
from pipeline.data_clean_report import compare_folders_quality
from pipeline.test_clean import TestDataCleaning  # noqa: F401  (registered for unittest discovery)


def main() -> None:
    print("start data cleaning")
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
    unittest.main(module="pipeline.test_clean",
                  argv=["first-arg-is-ignored"],
                  exit=False, verbosity=2)

    print("create data quality report")
    final_report = compare_folders_quality(str(INPUT_DIR), str(OUTPUT_DIR))

    if final_report is not None:
        print(final_report.to_string())
        report_path = Path(__file__).resolve().parent.parent / "data_report.xlsx"
        final_report.to_excel(report_path, index=False)
        print(f"\nReport output successfully as: {report_path}")


if __name__ == "__main__":
    main()
