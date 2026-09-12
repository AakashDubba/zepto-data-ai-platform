# Data Pipeline Module

## Overview
Web scraper and data pipeline that extracts book listings from [books.toscrape.com](http://books.toscrape.com/), cleans and transforms the data, and loads it into a normalized SQLite database.

## Conversion Constant
- **1 GBP = 105.50 INR** (fixed baseline conversion rate used throughout the project)

## Architecture
```
books.toscrape.com  →  Scraper  →  Raw DataFrame  →  Cleaner  →  SQLite DB
                      (requests     (title, price,    (price_gbp,  (categories
                       + BS4)        star_rating,      price_inr,   + books
                                     availability,     rating,      with PK/FK)
                                     category)         in_stock)
```

## Database Schema (Normalized 2-Table Design)
```sql
categories(
    category_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)

books(
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    rating      INTEGER NOT NULL,
    in_stock    INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
)
```

## Data Cleaning & Imputation Strategy
1. **Price**: Stripped currency symbols (£) and converted to float. Invalid/missing prices are imputed with **median imputation** to avoid outlier skew (mean would be distorted by very expensive books).
2. **Star Rating**: Text mapping ("One"→1, "Two"→2, ..., "Five"→5). Missing ratings imputed with **median** (rounded to nearest integer).
3. **Availability**: Regex match for "In stock" text → boolean `True`/`False`.
4. **GBP → INR**: Computed as `price_inr = price_gbp × 105.50`.
5. **Row Dropping**: Rows with missing `title` or `category` (essential identifiers) are dropped. This is justified because these are non-imputable categorical identifiers — median/mode imputation would produce meaningless results.

## SQL Queries (`queries.py`)
Five distinct queries demonstrating:
| # | Clauses Used | Description |
|---|-------------|-------------|
| 1 | SELECT, WHERE, ORDER BY | High-rated books (rating ≥ 4) |
| 2 | ORDER BY, LIMIT | Top 10 most expensive books |
| 3 | DISTINCT | All unique category names |
| 4 | JOIN, IN, BETWEEN | Books in price/rating range with category names |
| 5 | JOIN | Full book listing with category names |

- Queries 4 and 5 use `pd.read_sql()` to load SQL results into pandas DataFrames.
- Query 5 (JOIN) is replicated using `pd.merge()` with programmatic equivalence assertion via `pd.testing.assert_frame_equal()`.

## Installation & Usage
```bash
cd data_pipeline
pip install -r requirements.txt

# Run the pipeline (scrape → clean → store)
python pipeline.py

# Run SQL queries and equivalence tests
python queries.py
```

## Output
- `zepto_catalog.db` — Normalized SQLite database with ≥ 60 books across ≥ 3 categories
