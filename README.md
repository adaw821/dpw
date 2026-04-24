# 1.Project Structure
This project focuses on data cleaning for the TMDB movie dataset. The project directory structure is as follows：
```├── data/
│   ├── raw/                 # raw data
│   └── processed/           # clean data
├── test_clean.py        # test the data after cleaning
├── clean.py                 # data cleaning code 
├── data_cleaning_log.txt    # data cleaning log
└── README.md                # data cleaning work instruction
```

# 2. Data Cleaning Logic
This project processes 5 discrete raw datasets through clean.py to generate well-structured data analysis tables.

## movies_metadata:
Filtered out invalid movie_ids and formatted the release_date.
Applied regex cleaning to abnormal currency fields (budget, revenue).
Parsed and split nested JSON strings (genres, countries, companies) into independent dimensional sub-tables.

## ratings & links:
Filtered out abnormal ratings outside the 0.5-5.0 range.
Deduplicated by userId and movieId, keeping only the latest rating.
Mapped to a unified tmdbId via links_small.csv, and aggregated the data to calculate the average score and the total number of ratings for each movie.

## keywords:
Compatibly parsed abnormal encodings, extracted keyword_id and keyword_name from the JSON, and flattened them into a long table structure.

## credits:
Extracted the top 5 main actors from the cast.
Filtered the crew list, retaining only key positions (e.g., Director, Writer).

# 3. Run Instructions
Environment Requirements: Python 3.8+, pandas, numpy, pyarrow
Bash
python clean.py
