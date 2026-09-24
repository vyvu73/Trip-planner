"""
ingest_tripadvisor.py

Fetch reviews for every location in TRIPADVISOR_LOCATIONS and store them in
raw_tripadvisor_review. Re-runs are safe: reviews already stored are skipped.

Run:
    uv run python src/ingestion/ingest_tripadvisor.py
"""

import os
import time
import requests
from dotenv import load_dotenv

from chunk_web_pages import get_db_connection
from park_names import normalize_park
from tripadvisor_locations import TRIPADVISOR_LOCATIONS

load_dotenv()

API_KEY = os.getenv("TRIPADVISOR_API_KEY")
BASE_URL = "https://terra.tripadvisor.com/api"
FETCH_DELAY_SECONDS = 1

HEADERS = {
    "X-API-Key": API_KEY,
    "accept": "application/json",
}


def ensure_table(conn):
    """
    Create raw_tripadvisor_review if it doesn't exist. review_id is UNIQUE
    so re-runs can insert with ON CONFLICT DO NOTHING.
    """

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_tripadvisor_review (
                id             serial PRIMARY KEY,
                review_id      text UNIQUE NOT NULL,
                location_id    text NOT NULL,
                location_name  text,
                park_name      text,
                rating         int,
                title          text,
                content        text,
                trip_type      text,
                travel_date    text,
                published_date text,
                url            text,
                created_utc    timestamptz DEFAULT now()
            );
        """)

    conn.commit()


def search_location(query):
    """
    Search Tripadvisor for a location by name.
    Used by resolve_tripadvisor_ids.py only; fetch runs never search.
    """

    url = f"{BASE_URL}/locations/search"

    params = {
        "query": query,
        "country_code": "US",
        "category": "ATTRACTION",
        "size": 5,
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
    )

    response.raise_for_status()

    return response.json()


def fetch_reviews(location_id):
    """
    Fetch available English reviews for one Tripadvisor location.
    """

    url = f"{BASE_URL}/locations/{location_id}/reviews"

    params = {
        "language": "en",
        "sort_by": "MOST_RECENT",
        "page": 1,
        "size": 20,
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
    )

    response.raise_for_status()

    return response.json()


def get_primary_name(location):
    """
    Get the primary English name from a Tripadvisor location.
    """

    names = location.get("names", [])

    for name in names:
        if name.get("primary"):
            return name.get("value")

    if names:
        return names[0].get("value")

    return None


def first_value(items):
    """
    title/text come as [{"language": "en", "value": "...", "primary": true}].
    Return the primary value, else the first, else "".
    """

    if not items:
        return ""

    primary = next((item for item in items if item.get("primary")), items[0])

    return primary.get("value", "")


def to_row(review, entry):
    """
    One API review -> one raw_tripadvisor_review row.
    """

    _, park_name = normalize_park(entry["park_name"])

    return {
        "review_id": str(review["id"]),              # API returns an int
        "location_id": entry["location_id"],
        "location_name": entry["location_name"],     # our name, not the API's
        "park_name": park_name,
        "rating": review.get("rating"),
        "title": first_value(review.get("title")),
        "content": first_value(review.get("text")),
        "trip_type": review.get("trip_type"),
        "travel_date": review.get("travel_date"),
        "published_date": review.get("publish_ts"),
        "url": review.get("url"),
    }


def save_reviews(conn, rows):
    """
    Insert rows, skipping review_ids already stored.
    Returns how many were new (RETURNING only yields rows that were inserted).
    """

    new = 0

    with conn.cursor() as cur:
        for row in rows:
            cur.execute("""
                INSERT INTO raw_tripadvisor_review (
                    review_id,
                    location_id,
                    location_name,
                    park_name,
                    rating,
                    title,
                    content,
                    trip_type,
                    travel_date,
                    published_date,
                    url
                )
                VALUES (
                    %(review_id)s,
                    %(location_id)s,
                    %(location_name)s,
                    %(park_name)s,
                    %(rating)s,
                    %(title)s,
                    %(content)s,
                    %(trip_type)s,
                    %(travel_date)s,
                    %(published_date)s,
                    %(url)s
                )
                ON CONFLICT (review_id) DO NOTHING
                RETURNING id;
            """, row)

            if cur.fetchone():
                new += 1

    return new


def main():
    conn = get_db_connection()

    total_fetched = 0
    total_new = 0

    try:
        ensure_table(conn)

        for entry in TRIPADVISOR_LOCATIONS:
            name = entry["location_name"]

            if entry["location_id"] is None:
                print(f"WARNING: skipping {name} -- no location_id")
                continue

            try:
                reviews = fetch_reviews(entry["location_id"]).get("data", [])
            except requests.RequestException as error:
                # One bad id or timeout shouldn't stop the other locations.
                print(f"ERROR: {name}: {error}")
                continue

            new = save_reviews(conn, [to_row(review, entry) for review in reviews])
            conn.commit()

            total_fetched += len(reviews)
            total_new += new

            print(f"{entry['park_name']} | {name}: fetched {len(reviews)} / new {new}")

            time.sleep(FETCH_DELAY_SECONDS)

        print(f"\nFinished: fetched {total_fetched} / new {total_new}")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
