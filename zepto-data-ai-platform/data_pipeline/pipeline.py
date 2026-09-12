"""
Zepto Data Pipeline — Web Scraper, Cleaner, and SQLite Loader
=============================================================
Scrapes books from http://books.toscrape.com/ across multiple categories,
cleans/transforms the data, and loads it into a normalized SQLite database
(zepto_catalog.db) with two tables: categories and books.

Conversion constant: 1 GBP = 105.50 INR
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
import os
import sys
import math

# Fix Windows encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL = "http://books.toscrape.com/"
GBP_TO_INR = 105.50  # Fixed conversion constant: 1 GBP = 105.50 INR
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zepto_catalog.db")
MIN_BOOKS = 60
MIN_CATEGORIES = 3

RATING_MAP = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
}


# ── Scraper ──────────────────────────────────────────────────────────────────
def get_category_urls(limit=None):
    """Return a list of (category_name, category_url) tuples from the sidebar."""
    resp = requests.get(BASE_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    sidebar = soup.select("ul.nav-list > li > ul > li > a")
    cats = []
    for a_tag in sidebar:
        name = a_tag.text.strip()
        url = BASE_URL + a_tag["href"]
        cats.append((name, url))
    if limit:
        cats = cats[:limit]
    return cats


def scrape_category(category_name, category_url, books_list):
    """Scrape all books from a single category (handles pagination)."""
    page_url = category_url
    while page_url:
        resp = requests.get(page_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        for article in soup.select("article.product_pod"):
            title_tag = article.select_one("h3 a")
            title = title_tag["title"] if title_tag else "Unknown"

            price_tag = article.select_one("p.price_color")
            raw_price = price_tag.text.strip() if price_tag else ""

            rating_tag = article.select_one("p.star-rating")
            star_classes = rating_tag["class"] if rating_tag else []
            star_text = [c for c in star_classes if c.lower() != "star-rating"]
            raw_rating = star_text[0] if star_text else ""

            avail_tag = article.select_one("p.instock.availability")
            raw_availability = avail_tag.text.strip() if avail_tag else ""

            books_list.append({
                "title": title,
                "price": raw_price,
                "star_rating": raw_rating,
                "availability": raw_availability,
                "category": category_name,
            })

        # Pagination
        next_btn = soup.select_one("li.next > a")
        if next_btn:
            # Resolve relative URL
            page_url = page_url.rsplit("/", 1)[0] + "/" + next_btn["href"]
        else:
            page_url = None


def scrape_books():
    """Scrape ≥ 60 books across ≥ 3 categories."""
    categories = get_category_urls()
    books = []

    # We'll iterate through categories until we have enough books & categories
    categories_used = set()
    for cat_name, cat_url in categories:
        scrape_category(cat_name, cat_url, books)
        categories_used.add(cat_name)
        if len(books) >= MIN_BOOKS and len(categories_used) >= MIN_CATEGORIES:
            break

    print(f"[SCRAPE] Total books scraped: {len(books)}")
    print(f"[SCRAPE] Categories scraped: {len(categories_used)} -> {sorted(categories_used)}")
    return pd.DataFrame(books)


# ── Cleaning & Transformation ───────────────────────────────────────────────
def clean_price(raw_price):
    """Strip currency symbols and convert to float. Returns NaN on failure."""
    try:
        cleaned = re.sub(r"[^\d.]", "", str(raw_price))
        return float(cleaned) if cleaned else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def clean_rating(raw_rating):
    """Convert text rating ('One'..'Five') to integer 1-5. Returns NaN on failure."""
    try:
        return RATING_MAP.get(str(raw_rating).strip().lower(), float("nan"))
    except (ValueError, TypeError):
        return float("nan")


def clean_availability(raw_avail):
    """Convert availability text to boolean (True if 'in stock' appears)."""
    return bool(re.search(r"in stock", str(raw_avail), re.IGNORECASE))


def clean_dataframe(df):
    """Apply all cleaning and transformation steps."""
    print(f"\n[CLEAN] Raw dataframe shape: {df.shape}")

    # ── Price cleaning ──
    df["price_gbp"] = df["price"].apply(clean_price)

    # ── Rating cleaning ──
    df["rating"] = df["star_rating"].apply(clean_rating)

    # ── Availability cleaning ──
    df["in_stock"] = df["availability"].apply(clean_availability)

    # ── Numeric anomaly handling: median imputation ──
    # For price_gbp: if NaN, impute with median
    price_median = df["price_gbp"].median()
    nan_prices = df["price_gbp"].isna().sum()
    if nan_prices > 0:
        print(f"[CLEAN] Imputing {nan_prices} missing prices with median = {price_median:.2f}")
        df["price_gbp"].fillna(price_median, inplace=True)

    # For rating: if NaN, impute with median (rounded to int)
    rating_median = df["rating"].median()
    nan_ratings = df["rating"].isna().sum()
    if nan_ratings > 0:
        print(f"[CLEAN] Imputing {nan_ratings} missing ratings with median = {rating_median}")
        df["rating"].fillna(round(rating_median), inplace=True)

    df["rating"] = df["rating"].astype(int)

    # ── GBP to INR conversion ──
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR
    df["price_inr"] = df["price_inr"].round(2)

    # ── Drop any remaining fully-invalid rows ──
    before = len(df)
    df.dropna(subset=["title", "category"], inplace=True)
    after = len(df)
    if before != after:
        print(f"[CLEAN] Dropped {before - after} rows with missing title/category")

    print(f"[CLEAN] Cleaned dataframe shape: {df.shape}")
    print(f"[CLEAN] Conversion constant used: 1 GBP = {GBP_TO_INR} INR")
    print(f"[CLEAN] Sample cleaned data:")
    print(df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]].head())

    return df


# ── SQLite Storage ───────────────────────────────────────────────────────────
def create_database(df):
    """Create normalized 2-table SQLite database with PK/FK constraints."""
    # Remove old DB if exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    # ── Create categories table ──
    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    # ── Create books table with FK reference ──
    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
    """)

    # ── Insert categories ──
    unique_cats = df["category"].unique()
    for cat in unique_cats:
        cursor.execute("INSERT INTO categories (category_name) VALUES (?)", (cat,))
    conn.commit()

    # ── Build category name → id map ──
    cat_map = {}
    for row in cursor.execute("SELECT category_id, category_name FROM categories"):
        cat_map[row[1]] = row[0]

    # ── Insert books ──
    for _, row in df.iterrows():
        cursor.execute(
            "INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id) VALUES (?, ?, ?, ?, ?, ?)",
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                int(row["rating"]),
                int(row["in_stock"]),
                cat_map[row["category"]],
            ),
        )
    conn.commit()

    # ── Verification ──
    book_count = cursor.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    cat_count = cursor.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    print(f"\n[DB] Database created at: {DB_PATH}")
    print(f"[DB] Total books inserted: {book_count}")
    print(f"[DB] Total categories inserted: {cat_count}")

    assert book_count >= MIN_BOOKS, f"Expected ≥ {MIN_BOOKS} books, got {book_count}"
    assert cat_count >= MIN_CATEGORIES, f"Expected ≥ {MIN_CATEGORIES} categories, got {cat_count}"

    conn.close()
    return book_count, cat_count


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Zepto Data Pipeline — Books to Scrape")
    print("=" * 60)

    # Step 1: Scrape
    df_raw = scrape_books()

    # Step 2: Clean
    df_clean = clean_dataframe(df_raw)

    # Step 3: Store in SQLite
    book_count, cat_count = create_database(df_clean)

    print("\n" + "=" * 60)
    print(f"  Pipeline Complete: {book_count} books, {cat_count} categories")
    print("=" * 60)


if __name__ == "__main__":
    main()
