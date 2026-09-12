# Zepto Data & AI Platform — Capstone Project

A comprehensive data engineering, analytics, and AI platform demonstrating end-to-end skills in web scraping, ETL pipelines, exploratory data analysis, predictive modeling, and AI-powered support systems.

## Project Architecture

```
zepto-data-ai-platform/
├── .gitignore
├── requirements.txt                 ← Consolidated dependencies
├── README.md                        ← This file
├── data_pipeline/                   ← Module 1: Web Scraping & ETL
│   ├── pipeline.py                  ← Scraper + Cleaner + SQLite Loader
│   ├── queries.py                   ← SQL Queries + Pandas Merge Equivalence
│   ├── zepto_catalog.db             ← Normalized SQLite Database
│   ├── requirements.txt
│   └── README.md
├── analytics/                       ← Module 2: EDA & Predictive Modeling
│   ├── 01_eda.ipynb                 ← Exploratory Data Analysis
│   ├── 02_modeling.ipynb            ← Classification & Regression Modeling
│   ├── titanic.csv                  ← Offline Titanic Dataset
│   ├── best_pipeline.joblib         ← Serialized ML Pipeline
│   ├── test_pipeline.py             ← Pipeline Reload Verification
│   ├── requirements.txt
│   ├── charts/                      ← All generated visualizations
│   └── README.md
└── support_assistant/               ← Module 3: RAG Support Assistant
    ├── docs/                        ← 8 Zepto Policy Documents
    │   ├── doc_01.txt .. doc_08.txt
    ├── chroma_db/                   ← Persisted ChromaDB Vector Store
    ├── ingest.py                    ← Document Ingestion & Embedding
    ├── rag_engine.py                ← LangGraph StateGraph RAG Pipeline
    ├── main.py                      ← FastAPI Application
    ├── test_api.py                  ← API Integration Tests
    ├── requirements.txt
    ├── Dockerfile                   ← Docker Isolation
    └── README.md
```

## Quick Start

### Prerequisites
- Python 3.11+
- pip
- Docker (optional, for containerized deployment)

### Installation
```bash
# Clone and install all dependencies
cd zepto-data-ai-platform
pip install -r requirements.txt
```

### Execution Order

#### 1. Data Pipeline
```bash
# Scrape books, clean data, and load into SQLite
python data_pipeline/pipeline.py

# Run SQL queries and pandas merge equivalence tests
python data_pipeline/queries.py
```

#### 2. Analytics Pipeline
```bash
# Run EDA (creates titanic.csv + charts)
jupyter nbconvert --to notebook --execute analytics/01_eda.ipynb
# OR: python analytics/01_eda.py

# Run Modeling (creates best_pipeline.joblib + metrics)
jupyter nbconvert --to notebook --execute analytics/02_modeling.ipynb
# OR: python analytics/02_modeling.py

# Verify pipeline
python analytics/test_pipeline.py
```

#### 3. Support Assistant
```bash
# Ingest policy documents into ChromaDB
python support_assistant/ingest.py

# Start the API server
cd support_assistant
MOCK_LLM=1 uvicorn main:app --host 0.0.0.0 --port 7860 &

# Run API tests
python test_api.py

# Docker (optional)
docker build -t zepto-support .
docker run -d -p 7860:7860 --name zepto-test zepto-support
docker stop zepto-test && docker rm zepto-test
```

## Module Details

### Module 1: Data Pipeline (25 marks)
- **Web Scraper**: Extracts ≥ 60 books across ≥ 3 categories from [books.toscrape.com](http://books.toscrape.com/)
- **Conversion**: `1 GBP = 105.50 INR` (fixed baseline constant)
- **Cleaning**: Median imputation for numeric anomalies; row dropping for missing identifiers
- **Storage**: Normalized 2-table SQLite schema (`categories` + `books` with PK/FK)
- **Queries**: 5 SQL queries (SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, JOIN) + `pd.merge()` equivalence

### Module 2: Analytics Pipeline (50 marks)
- **EDA**: Profiling, threshold-based cleaning, IQR outliers, skewness, 6×6 heatmap, 4 multivariate charts
- **Classification**: Logistic Regression, Decision Tree, Random Forest with full metric suite
- **Imbalance**: Baseline vs `class_weight='balanced'` vs SMOTE (via `imblearn.Pipeline`)
- **Tuning**: `GridSearchCV` over `n_estimators`, `max_depth`, `max_features` + OOB score
- **Regression**: Linear regression predicting `fare` with MAE, RMSE, R², Adj R² + heteroscedasticity analysis
- **Serialization**: `best_pipeline.joblib` with reload verification

### Module 3: Support Assistant (25 marks)
- **RAG Pipeline**: LangGraph `StateGraph` with 3 nodes + conditional intent routing
- **Vector Store**: ChromaDB with `all-MiniLM-L6-v2` embeddings from 8 policy documents
- **API**: FastAPI `POST /ask` with Pydantic validation
- **Modes**: `MOCK_LLM=1` (default, keyword routing + canned answers) and `MOCK_LLM=0` (real LLM with structured prompts)
- **Docker**: Isolated container serving on port 7860

## Git Workflow
```bash
# Feature branch with genuine incremental commits
git checkout -b feature/zepto-platform-build

# Commit 1: Data pipeline implementation
# Commit 2: Analytics & modeling pipeline
# Commit 3: Support assistant service

# Clean non-fast-forward merge
git checkout main
git merge --no-ff feature/zepto-platform-build -m "Merge feature/zepto-platform-build into main"
```

## Technology Stack
| Category | Technologies |
|----------|-------------|
| Scraping | requests, BeautifulSoup4, lxml |
| Data | pandas, numpy, SQLite3 |
| ML | scikit-learn, imbalanced-learn |
| Visualization | matplotlib, seaborn |
| NLP/Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Orchestration | LangGraph |
| API | FastAPI, Uvicorn |
| Validation | Pydantic |
| Container | Docker |
