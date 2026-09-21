
"""
ingest_raw_web_page.py   (file 1 of 3)

Goal:
    Download the hub page with Firecrawl and save it as a markdown (.md)
    file, so find_park_article_links.py can read it.

Uses 1 Firecrawl credit each time you run it.
You only need to run it again if the hub page changes.

Usage:
    uv run ingest_raw_web_page.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from firecrawl import Firecrawl


# ============================================================
# Settings
# ============================================================

HUB_URL = "https://www.visitcalifornia.com/things-to-do/national-parks/"

OUTPUT_FOLDER = Path("crawl_test_output")

# find_park_article_links.py looks for a file starting with "01-"
OUTPUT_FILE = OUTPUT_FOLDER / "01-hub-page.md"


# ============================================================
# Setup
# ============================================================

load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not FIRECRAWL_API_KEY:
    raise ValueError("FIRECRAWL_API_KEY is missing from .env")

firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)


# ============================================================
# Main
# ============================================================

def main():
    print(f"Downloading: {HUB_URL}")

    # scrape = download just this one page (no link following)
    result = firecrawl.scrape(
        HUB_URL,
        formats=["markdown"],
        only_main_content=True,
    )

    markdown = result.markdown or ""

    if not markdown.strip():
        raise ValueError("The hub page came back empty. Nothing saved.")

    OUTPUT_FOLDER.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(markdown, encoding="utf-8")

    title = getattr(result.metadata, "title", None) if result.metadata else None

    print(f"Title: {title or '(no title)'}")
    print(f"Size:  {len(markdown)} characters, {len(markdown.split())} words")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()