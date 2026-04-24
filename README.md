1.项目结构
本项目是针对TMDB电影数据集的项目清洗，项目目录结构如下：
├── data/
│   ├── raw/                 # 存放原始数据
│   └── processed/           # 清洗后的干净数据
├── tests/
│   └── test_clean.py        # 数据基础校验脚本
├── clean.py                 # 数据清洗脚本 
├── data_cleaning_log.txt    # 自动生成的清洗日志
└── README.md                # 清洗工程文档
2. 清洗逻辑说明
本工程将原始的 5 份离散数据集，通过clean.py清洗得到结构清晰的数据分析表

movies_metadata:
过滤了无效的 movie_id，格式化了 release_date。
对异常的货币字段 (budget, revenue) 进行了正则清洗。
将嵌套的 JSON 字符串（类型、国家、公司）解析拆分成了独立的维度子表。

ratings & links:
过滤掉 0.5-5.0 范围外的异常评分。
按 userId 和 movieId 去重，保留最新评分。
通过 links_small.csv 映射为统一的 tmdbId，并聚合计算每部电影的平均分与评价人数。

keywords:
兼容解析了异常编码，提取 JSON 中的 keyword_id 和 keyword_name，展平为长表结构。

credits:
提取 cast 中的前 5 位主要演员。
过滤 crew 列表，仅保留关键职位（导演 Director、编剧 Writer 等）。

3. 运行指南
环境要求: Python 3.8+, pandas, numpy, pyarrow
Bash
python clean.py
