import os
import requests
import psycopg2
from dotenv import load_dotenv
from src.ingestion.nps_transformers import (
    transform_park,
    transform_article,
    transform_place,
    transform_thing_to_do,
    transform_campground,
    transform_visitor_centers,
    transform_tour,
    transform_fees_passes,
    transform_parking_lot
)

load_dotenv()
PAGE_SIZE = 500
API_KEY = os.getenv("NPS_API_KEY")
BASE_URL = os.getenv("NPS_BASE_URL")
ENDPOINTS_TO_INGEST = {
    "parks": transform_park,
    "thingstodo":transform_thing_to_do,
    "campgrounds":transform_campground,
    "visitorcenters": transform_visitor_centers,
    "places": transform_place,
    "articles": transform_article,
    "tours":transform_tour,
    "feespasses":transform_fees_passes,
    "parkinglots": transform_parking_lot,
}
PARK_CODES = [
    "chis",  # Channel Islands
    "deva",  # Death Valley
    "jotr",  # Joshua Tree
    "kica",  # Kings Canyon
    "lavo",  # Lassen Volcanic
    "pinn",  # Pinnacles
    "redw",  # Redwood
    "seki",  # Sequoia
    "yose",  # Yosemite
]

def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        database="raw_data_nps"
    )

def fetch_endpoint(endpoint, params=None):
    headers = {"X-Api-Key": API_KEY}
    params = dict(params or {})
    params["limit"] = PAGE_SIZE
    start = 0
    records = []

    while True:
        params["start"] = start
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers,
                                params=params, timeout=30)
        response.raise_for_status()
        body = response.json()

        for entry in body["data"]:
            if isinstance(entry, list):
                records.extend(entry)
            else:
                records.append(entry)

        start += PAGE_SIZE
        if start >= int(body.get("total", 0)):
            break
    return records

def insert_nps_record(cursor, record):
    cursor.execute(
        "INSERT INTO raw_data_nps (source_id, park_name, page_type, title, content, url) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (page_type, source_id) DO UPDATE SET "
        "park_name = EXCLUDED.park_name, title = EXCLUDED.title, "
        "content = EXCLUDED.content, url = EXCLUDED.url",
        (record["source_id"], record["park_name"], record["page_type"],
         record["title"], record["content"], record["url"])
    )

def ingest_endpoint(endpoint, transformer, params=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        data = fetch_endpoint(endpoint, params)
        for item in data:
            record = transformer(item)
            if not record["content"].strip():
                continue
            insert_nps_record(cursor, record)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Ingestion failed for {endpoint}: {e}")
    finally:
        cursor.close()
        conn.close()

def test_endpoint(endpoint="parks", park_code="chis"):
    transformer = ENDPOINTS_TO_INGEST[endpoint]
    records = fetch_endpoint(endpoint, {"parkCode": park_code})
    print(f"Fetched {len(records)} records from {endpoint} for {park_code}")

    if not records:
        return

    # Confirm the filter worked (should all be the same park)
    print("parkCode values:", {r.get("parkCode") for r in records})

    # Check the transformer output on the first few records
    for item in records[:3]:
        record = transformer(item)
        print(record["title"], "|", record["url"], "|", len(record["content"] or ""), "chars")

    # Check every record has what the DB insert needs
    transformed = [transformer(item) for item in records]
    missing = [t for t in transformed if not t.get("source_id")]
    empty = [t for t in transformed if not t["content"].strip()]
    dupes = len(transformed) - len({t["source_id"] for t in transformed})
    print(f"Missing source_id: {len(missing)} | empty content: {len(empty)} | duplicate source_id: {dupes}")


def main():
    for park_code in PARK_CODES:
        for endpoint, transformer in ENDPOINTS_TO_INGEST.items():
            print(f"Ingesting {endpoint} for {park_code}")
            ingest_endpoint(endpoint, transformer, {"parkCode": park_code})

if __name__ == "__main__":
    for ep in ENDPOINTS_TO_INGEST:
        print(f"\n=== {ep} ===")
        try:
            test_endpoint(ep, "chis")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")