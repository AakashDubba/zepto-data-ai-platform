"""
Zepto Data Pipeline — SQL Queries & Pandas Equivalence Testing
==============================================================
Executes 5 distinct SQL queries against zepto_catalog.db covering:
  SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, JOIN
Reads results into pandas DataFrames via pd.read_sql().
Replicates the JOIN query using pd.merge() and asserts equivalence.
"""

import sqlite3
import pandas as pd
import os
import sys

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zepto_catalog.db")


def run_queries():
    """Execute and display 5 distinct SQL queries + pandas merge equivalence check."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    print("=" * 70)
    print("  Zepto SQL Queries — zepto_catalog.db")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────────
    # QUERY 1: SELECT / WHERE — Books with rating ≥ 4
    # ──────────────────────────────────────────────────────────────────────
    q1 = """
    SELECT book_id, title, rating, price_gbp
    FROM books
    WHERE rating >= 4
    ORDER BY rating DESC
    """
    print("\n── QUERY 1: SELECT/WHERE — Books with rating ≥ 4 (ORDER BY rating DESC) ──")
    df_q1 = pd.read_sql(q1, conn)
    print(df_q1.to_string(index=False))
    print(f"  → {len(df_q1)} rows returned\n")

    # ──────────────────────────────────────────────────────────────────────
    # QUERY 2: ORDER BY + LIMIT — Top 10 most expensive books
    # ──────────────────────────────────────────────────────────────────────
    q2 = """
    SELECT book_id, title, price_gbp, price_inr
    FROM books
    ORDER BY price_gbp DESC
    LIMIT 10
    """
    print("── QUERY 2: ORDER BY + LIMIT — Top 10 most expensive books ──")
    df_q2 = pd.read_sql(q2, conn)
    print(df_q2.to_string(index=False))
    print(f"  → {len(df_q2)} rows returned\n")

    # ──────────────────────────────────────────────────────────────────────
    # QUERY 3: DISTINCT — Unique categories
    # ──────────────────────────────────────────────────────────────────────
    q3 = """
    SELECT DISTINCT category_name
    FROM categories
    ORDER BY category_name
    """
    print("── QUERY 3: DISTINCT — All unique categories ──")
    df_q3 = pd.read_sql(q3, conn)
    print(df_q3.to_string(index=False))
    print(f"  → {len(df_q3)} distinct categories\n")

    # ──────────────────────────────────────────────────────────────────────
    # QUERY 4: IN + BETWEEN — Books in specific categories with price range
    # ──────────────────────────────────────────────────────────────────────
    q4 = """
    SELECT b.book_id, b.title, b.price_gbp, b.rating, c.category_name
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    WHERE b.price_gbp BETWEEN 10.00 AND 40.00
      AND b.rating IN (3, 4, 5)
    ORDER BY b.price_gbp
    """
    print("── QUERY 4: IN + BETWEEN — Books priced £10-£40 with rating 3-5 ──")
    df_q4 = pd.read_sql(q4, conn)
    print(df_q4.to_string(index=False))
    print(f"  → {len(df_q4)} rows returned\n")

    # ──────────────────────────────────────────────────────────────────────
    # QUERY 5: JOIN — All books with their category names (pd.read_sql)
    # ──────────────────────────────────────────────────────────────────────
    q5_join = """
    SELECT b.book_id, b.title, b.price_gbp, b.price_inr, b.rating, b.in_stock,
           c.category_name
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    ORDER BY b.book_id
    """
    print("── QUERY 5: JOIN — Books with category names (via SQL JOIN) ──")
    df_q5_sql = pd.read_sql(q5_join, conn)
    print(df_q5_sql.head(10).to_string(index=False))
    print(f"  → {len(df_q5_sql)} total rows\n")

    # ──────────────────────────────────────────────────────────────────────
    # PANDAS MERGE EQUIVALENCE CHECK
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Pandas pd.merge() Equivalence Test")
    print("=" * 70)

    # Read raw tables into DataFrames
    df_books = pd.read_sql("SELECT * FROM books", conn)
    df_categories = pd.read_sql("SELECT * FROM categories", conn)

    # Replicate the JOIN query using pd.merge
    df_merged = pd.merge(
        df_books, df_categories,
        on="category_id",
        how="inner"
    )

    # Select and order columns to match SQL JOIN output
    df_merged = df_merged[["book_id", "title", "price_gbp", "price_inr",
                            "rating", "in_stock", "category_name"]]
    df_merged = df_merged.sort_values("book_id").reset_index(drop=True)

    # Also reset index on SQL result for comparison
    df_q5_compare = df_q5_sql.reset_index(drop=True)

    # Assert equivalence
    print("\n[TEST] Comparing SQL JOIN result vs pd.merge() result...")
    print(f"  SQL JOIN shape:   {df_q5_compare.shape}")
    print(f"  pd.merge shape:   {df_merged.shape}")

    assert df_q5_compare.shape == df_merged.shape, \
        f"Shape mismatch: SQL {df_q5_compare.shape} vs merge {df_merged.shape}"

    # Compare values
    pd.testing.assert_frame_equal(
        df_q5_compare, df_merged,
        check_dtype=False,
        check_exact=False,
        atol=0.01,
    )
    print("  ✅ PASS — SQL JOIN and pd.merge() produce equivalent results!")
    print(f"  Verified: {len(df_merged)} rows match across all columns\n")

    conn.close()
    print("=" * 70)
    print("  All queries executed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_queries()
