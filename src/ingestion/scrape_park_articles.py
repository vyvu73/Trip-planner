"""
scrape_park_articles.py   (file 3 of 3)

Goal:
    Get the article links from find_park_article_links.py, scrape each
    article with Firecrawl, and save it to raw_web_page with its park name.

No JSON file:
    The links come straight from get_park_article_links().

Before running:
    - ingest_raw_web_page.py has saved the hub page .md file
    - Your local Postgres (raw_data_nps) is running

Usage:
    uv run scrape_park_articles.py
"""

import os

import psycopg2
from dotenv import load_dotenv
from firecrawl import Firecrawl

from find_park_article_links import get_park_article_links   # <-- file 2


# ============================================================
# Settings``
# ============================================================

SOURCE = "visitcalifornia"

# Start small to test. Set to None to scrape every link.
MAX_ARTICLES = None


# ============================================================
# Setup
# ============================================================

load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not FIRECRAWL_API_KEY:
    raise ValueError("FIRECRAWL_API_KEY is missing from .env")

firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)


def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        database="trip_planner",
    )

def url_exists(conn, url):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM raw_web_page
            WHERE url = %s;
            """,
            (url,),
        )

        return cur.fetchone() is not None
# ============================================================
# Scrape one article
# ============================================================

def scrape_article(url):
    """Download one page as markdown. Returns (title, markdown)."""
    result = firecrawl.scrape(url, formats=["markdown"], only_main_content=True)
    title = getattr(result.metadata, "title", None) if result.metadata else None
    return title, (result.markdown or "")


# ============================================================
# Save one page (same insert as your original ingestion script)
# ============================================================

def insert_web_page(conn, page):
    """Re-running updates the row instead of making a duplicate."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO raw_web_page (source, park_name, title, url, content)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (url)
            DO UPDATE SET
                source     = EXCLUDED.source,
                park_name  = COALESCE(raw_web_page.park_name, EXCLUDED.park_name),
                title      = COALESCE(EXCLUDED.title, raw_web_page.title),
                content    = EXCLUDED.content,
                crawled_at = NOW()
            RETURNING id;
            """,
            (page["source"], page["park_name"], page["title"],
             page["url"], page["content"]),
        )
        return cur.fetchone()[0]


# ============================================================
# Main
# ============================================================

def main():
    links = get_park_article_links()
    print(f"Found {len(links)} article link(s)")

    if MAX_ARTICLES is not None:
        links = links[:MAX_ARTICLES]
        print(f"Testing with the first {len(links)} (MAX_ARTICLES = {MAX_ARTICLES})")

    saved = skipped = failed = 0
    conn = get_db_connection()

    try:
        for i, item in enumerate(links, start=1):
            url = item["url"]
            park_name = item["park_name"]     # <-- inherited from the hub section

            print(f"\n[{i}/{len(links)}] {park_name}\n  {url}")

            if url_exists(conn, url):
                print("  SKIPPED: already exists in database")
                skipped += 1
                continue
            # One bad page shouldn't stop the whole run
            try:
                title, markdown = scrape_article(url)
            except Exception as error:
                print(f"  FAILED: {error}")
                failed += 1
                continue

            if not markdown.strip():
                print("  SKIPPED: empty page")
                skipped += 1
                continue

            row_id = insert_web_page(conn, {
                "source": SOURCE,
                "park_name": park_name,
                "title": title,
                "url": url,
                "content": markdown,
            })
            conn.commit()     # keep each save, even if a later one fails

            print(f"  SAVED: id={row_id}, {len(markdown.split())} words")
            saved += 1

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"\nSaved: {saved}   Skipped: {skipped}   Failed: {failed}")


if __name__ == "__main__":
    main()