# 电影分析项目 

基于 **TMDB 电影数据集** 的端到端电影分析项目，包含两个交互式 Streamlit 应用：

1. **盈利预测**：用 XGBoost 分类器预测电影是否盈利（ROI ≥ 2.5）。
2. **智能推荐**：混合式内容推荐器，融合 TF-IDF 语义相似度 + 类型/关键词/演员/导演的 Jaccard 重合度，再叠加质量得分。

> **在线演示**：已部署到 Streamlit Community Cloud（首次冷启动会从 Google Drive 拉取数据）。

---

## 项目结构

```
dpw/
├── app.py                              # Streamlit Cloud 入口（下载数据后交给 app_home）
├── app_home.py                         # 首页 / 页面路由
├── success_model.py                    # 盈利预测页面
├── recommend_model.py                  # 推荐系统页面
├── page_design.py                      # 共享 UI 工具（背景、自定义 CSS）
├── build_ml_recommender.py             # 推荐核心算法：TF-IDF、Jaccard、打分
│
├── pipeline/                           # 数据清洗流水线
│   ├── clean.py                        # 5 张原始 TMDB 表的清洗逻辑
│   ├── clean_ratings_for_recommender.py  # 生成 final_ratings_raw.parquet
│   ├── data_clean_report.py            # 数据质量报告（缺失/重复/异常）
│   ├── test_clean.py                   # 清洗结果单元测试
│   └── run_cleaning.py                 # 主入口：清洗 → 测试 → 报告
│
├── analysis/                           # EDA 与可视化（一次性脚本）
│   ├── success_eda.py                  # 盈利 / ROI 相关图表
│   └── recommender_eda.py              # 推荐数据探索
│
├── scripts/                            # 工具脚本
│   └── generate_tfidf_cache.py         # 预生成 TF-IDF 缓存
│
├── data/
│   ├── raw/                            # 放原始 TMDB CSV（已 gitignore）
│   └── processed/                      # 清洗后的 parquet/csv（已 gitignore）
│
├── recommendation_outputs/
│   └── ml_recommender/
│       ├── tfidf_vectors.pkl           # TF-IDF 向量缓存（已提交）
│       └── tfidf_norms.pkl             # 文档范数缓存（已提交）
│
├── design/background.jpg               # 页眉背景图
├── plots/                              # 盈利 EDA 图
├── plots_recommend/                    # 推荐 EDA 图
├── requirements.txt                    # Python 依赖
├── runtime.txt                         # 锁定 Python 版本（3.11）
├── README.md                           # 英文版
└── README.zh.md                        # 本文件
```

---

## 本地快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. 准备数据

把以下 TMDB 原始文件放进 `data/raw/`：

```
movies_metadata.csv
ratings_small.csv
links_small.csv
keywords.csv
credits.csv
```

运行完整清洗流水线：

```bash
python -m pipeline.run_cleaning
```

会在 `data/processed/` 生成所有清洗后的表，外加 `data_cleaning_log.txt` 和 `data_report.xlsx`。

生成推荐系统专用的评分表：

```bash
python pipeline/clean_ratings_for_recommender.py
```

运行推荐系统 EDA，生成推荐应用必需的分析表 `recommendation_analysis_table.parquet`（**必须**，否则推荐页会报 `FileNotFoundError`）：

```bash
python analysis/recommender_eda.py
```

> 这一步会在 `recommendation_outputs/` 下产出 `recommendation_analysis_table.parquet` / `.csv`、汇总表、相关性热图等。需要先装好 `matplotlib`、`seaborn`、`statsmodels`（已在 `requirements.txt` 中）。

（可选）运行盈利模型 EDA，产出 `plots/` 下的可视化：

```bash
python analysis/success_eda.py
```

预生成 TF-IDF 缓存（推荐；不生成的话应用会现场重建）：

```bash
python -m scripts.generate_tfidf_cache
```

### 3. 启动应用

```bash
streamlit run app_home.py
```

打开终端里输出的 URL（一般是 <http://localhost:8501>）。

---

## 云端部署（Streamlit Community Cloud）

云端入口是 **`app.py`**。冷启动时它会：

1. 检查 `data/processed/`，缺文件就用 `gdown` 从 Google Drive 下载。
2. （可选）下载缓存好的 TF-IDF pickle。
3. 把控制权交给 `app_home.py` 渲染首页 / 路由到子页面。

### 配置 Google Drive

1. 把清洗好的 `data/processed/` 整个文件夹上传 Google Drive。
2. 右键 → 共享 → **"知道链接的任何人 - 查看者"**。
3. 复制链接里的 folder ID：`https://drive.google.com/drive/folders/<FOLDER_ID>`。
4. 直接改 `app.py` 里的 `GDRIVE_CONFIG`，**或者**在 Streamlit secrets 里加：

   ```toml
   [gdrive]
   data_folder_id = "<FOLDER_ID>"
   tfidf_folder_id = ""
   data_zip_id = ""
   ```

5. Streamlit Cloud → New app → **Main file path** 填 `app.py` → Deploy。

---

## 流水线细节

### 数据清洗（`pipeline/clean.py`）

| 模块 | 输入 | 输出 |
|---|---|---|
| `clean_metadata` | `movies_metadata.csv` | `movies_main.parquet`、`movie_genres.parquet`、`movie_countries.parquet`、`movie_companies.parquet` |
| `clean_ratings_and_links` | `ratings_small.csv`、`links_small.csv` | `ratings_clean.parquet` |
| `clean_keywords` | `keywords.csv` | `keywords_long.csv` |
| `clean_credits` | `credits.csv` | `cast.csv`、`crew.csv` |

要点：
- 货币字段用正则去掉 `$`、`,`。
- 嵌套 JSON 字段（genres / countries / companies / collection / cast / crew）用容错的 `safe_json_parse` 解析（先 `ast.literal_eval`，再 `json.loads`）。
- crew 只保留 **Directing** + **Writing** 部门。
- 每部电影只保留前 5 位主演。
- 日期 / 时长 / 热度等列强制转数值并验证（`1850 ≤ release_year ≤ 2030`）。

### 质量报告（`pipeline/data_clean_report.py`）

对比 `data/raw/` 和 `data/processed/`：
- 缺失率（NaN 单元格 / 总单元格）
- 重复行率
- 异常率（数值列 |z-score| > 3）

输出：`data_report.xlsx`。

### 单元测试（`pipeline/test_clean.py`）

六项断言：
1. 所有预期的输出文件存在且非空。
2. `movies_main.movie_id` 唯一且无缺失。
3. 所有 `avg_user_rating` 落在 `[0.5, 5.0]`。
4. `cast.csv` 含所有必需列。
5. `release_year`、`budget`、`revenue` 通过逻辑边界检查。
6. `crew.csv` 仅包含 Directing / Writing 部门。

---

## 盈利预测模型

**文件**：`success_model.py`

- **目标**：`is_profitable = (revenue / budget ≥ 2.5)`
- **特征**：`budget`、`genre_count`、`country_count`、`company_count`、`runtime_bin`、`primary_genre_id`、`director_win_rate`、`writer_win_rate`
- **模型**：`XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, scale_pos_weight=…)`
- **缓存**：整个特征工程 + 训练流程包在 `@st.cache_resource` 里，**容器生命周期内只跑一次**。

---

## 推荐模型

**文件**：`recommend_model.py`，核心算法在 `build_ml_recommender.py`

- **内容得分 = 五项加权和**：
  - 类型 Jaccard
  - 关键词 Jaccard
  - 演员 Jaccard
  - 导演 Jaccard
  - `recommendation_text` 上的 TF-IDF 余弦相似度
- **质量得分 = 0.6 ×（平均评分 / 5）+ 0.4 × log 缩放后的用户数**
- **最终得分 = α · 内容得分 + β · 质量得分**
- 默认权重读 `config.json`，缺失则均分。
- 侧栏可调：推荐数量（5–20）、最低匹配分阈值。

---

## 性能优化说明

- XGBoost 模型用 `@st.cache_resource` 缓存——第一次进盈利页约 5–10 秒，之后所有交互都是瞬时的。
- TF-IDF 缓存（`tfidf_vectors.pkl`、`tfidf_norms.pkl`）已直接随仓库提交，推荐页冷启动从 ~10 秒降到 ~1 秒。
- `cast.csv` / `crew.csv` 用 `usecols=` 只读必要列。
- 查询表（`director_winrate_lookup`、`actor_winrate_lookup`）一次性构建成 Python `dict`，每次交互查询都是 O(1)。

---

## 技术栈

- **Python 3.11**（通过 `runtime.txt` 锁定）
- **Streamlit ≥ 1.32** — UI
- **pandas / numpy / pyarrow** — 数据
- **scikit-learn** — `train_test_split`
- **xgboost ≥ 2.0** — 盈利分类器
- **gdown** — Google Drive 拉取
- **matplotlib / seaborn / scipy** — EDA 图（仅 `analysis/` 需要）

---

## 许可

内部项目，仅供学术 / 课程使用。
