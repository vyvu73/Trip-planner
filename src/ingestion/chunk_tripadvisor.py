"""
chunk_tripadvisor.py

One documents_chunks row per raw_tripadvisor_review row. Reviews are well
under MAX_TOKENS, so nothing is split.

Upserts on (source_table, source_id, chunk_index) instead of delete + insert,
so re-runs keep existing embeddings. A chunk's embedding is only reset to
NULL when its content changed, and load_chunk.py re-embeds just that one.

Run:
    uv run python src/ingestion/chunk_tripadvisor.py
"""

import json

from chunk_web_pages import MIN_CHUNK_TOKENS, count_tokens, get_db_connection
from park_names import normalize_park


SOURCE_TYPE = "tripadvisor"
SOURCE_TABLE = "raw_tripadvisor_review"


def fetch_reviews(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                review_id,
                location_id,
                location_name,
                park_name,
                rating,
                title,
                content,
                trip_type,
                travel_date,
                url
            FROM raw_tripadvisor_review
            WHERE content IS NOT NULL
              AND TRIM(content) <> ''
            ORDER BY id;
        """)

        columns = [desc[0] for desc in cur.description]

        return [
            dict(zip(columns, row))
            for row in cur.fetchall()
        ]


def build_content(review):
    """Review title + blank line + review text."""
    title = (review["title"] or "").strip()
    text = review["content"].strip()
    return f"{title}\n\n{text}" if title else text


def save_chunk(conn, review, content):
    park_code, park_name = normalize_park(review["park_name"])

    metadata = {
        "park_code": park_code,
        "page_type": "review",
        "location_id": review["location_id"],
        "location_name": review["location_name"],
        "rating": review["rating"],
        "trip_type": review["trip_type"],
        "travel_date": review["travel_date"],
    }

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents_chunks (
                source_type,
                source_table,
                source_id,
                chunk_index,
                park_name,
                title,
                url,
                content,
                metadata,
                token_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (source_table, source_id, chunk_index) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                park_name   = EXCLUDED.park_name,
                title       = EXCLUDED.title,
                url         = EXCLUDED.url,
                content     = EXCLUDED.content,
                metadata    = EXCLUDED.metadata,
                token_count = EXCLUDED.token_count,
                embedding   = CASE
                    WHEN documents_chunks.content IS DISTINCT FROM EXCLUDED.content
                    THEN NULL
                    ELSE documents_chunks.embedding
                END;
        """, (
            SOURCE_TYPE,
            SOURCE_TABLE,
            review["review_id"],
            0,
            park_name,
            review["location_name"],
            review["url"],
            content,
            json.dumps(metadata),
            count_tokens(content),
        ))


def main():
    conn = get_db_connection()

    chunked = 0
    skipped = 0

    try:
        reviews = fetch_reviews(conn)
        print(f"Found {len(reviews)} review(s)")

        for review in reviews:
            content = build_content(review)

            # Drop one-liners like "Amazing!! Must see" -- noise, not information.
            if count_tokens(content) < MIN_CHUNK_TOKENS:
                skipped += 1
                continue

            save_chunk(conn, review, content)
            chunked += 1

        conn.commit()
        print(f"Finished: chunked {chunked} / skipped {skipped} (too short)")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
