# 1.Project Structure
This project focuses on data cleaning for the TMDB movie dataset. The project directory structure is as follows：
```
├── data/
│   ├── raw/                 # raw data
│   └── processed/           # clean data
├── test_clean.py        # test the data after cleaning
├── clean.py                 # data cleaning code 
├── data_cleaning_log.txt    # data cleaning log
├── data_clean_report.py     # data quality report
├── data_report.xlsx         
└── README.md                # data cleaning work instruction
```
```bash
note: if want to run the code, must ensure that raw/ and processed/ folder in the data/ directory (folder name cannot be changed)
```

# 2. Data Cleaning Logic
```
This project processes 5 discrete raw datasets through clean.py to generate well-structured data analysis tables.
```

## movies_metadata:
```
Filtered out invalid movie_ids and formatted the release_date.
Applied regex cleaning to abnormal currency fields (budget, revenue).
Parsed and split nested JSON strings (genres, countries, companies) into independent dimensional sub-tables.
```
## ratings & links:
```
Filtered out abnormal ratings outside the 0.5-5.0 range.
Deduplicated by userId and movieId, keeping only the latest rating.
Mapped to a unified tmdbId via links_small.csv, and aggregated the data to calculate the average score and the total number of ratings for each movie.
```

## keywords:
```
Compatibly parsed abnormal encodings, extracted keyword_id and keyword_name from the JSON, and flattened them into a long table structure.
```

## credits:
```
Extracted the top 5 main actors from the cast.
Filtered the crew list, retaining only key positions (e.g., Director, Writer).
```

# 3. Cleaning Test
```
confirms the generation of all output files, ensures that none of them are empty.
ensure the id not duplicates or empty
check logical validity(like budger cannot be negative)
ensure only include the Directing and Writing departments.
```

# 5. Data Quality Report
```
contain duplicate rate, missing rate, outlier rate of the raw data and clean data, and do comparison of these 2 data
```
# 4. Run Instructions
Environment Requirements: Python 3.8+, pandas, numpy, pyarrow
```bash
python main.py
# 6. 前端运行
```bash
streamlit run success_model.py
