import os
import psycopg2
import argparse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL")
FETCH_BATCH = 500
EMBED_BATCH = 100

def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        database="trip_planner"
    )


def build_embed_text(source_type, park_name, title, content, metadata):
    """
    The text actually sent to the embedding model: a short header of
    context (title, park, section) plus the chunk's content.

    The header is never stored in the DB -- content stays clean, this
    is only what OpenAI sees. Without it, a chunk like "Reservations are
    not required this year" embeds with no idea which park it's about.
    """

    lines = [f'title: "{title}"', f"park: {park_name}"]

    page_type = metadata.get("page_type")
    if page_type:
        lines.append(f"page_type: {page_type}")

    if source_type == "web_page":
        heading_path = metadata.get("heading_path") or []
        if heading_path:
            lines.append(f"section: {' > '.join(heading_path)}")

    elif source_type == "nps":
        section = metadata.get("section")
        part, parts = metadata.get("part"), metadata.get("parts")
        if section:
            suffix = f" (part {part} of {parts})" if parts and parts > 1 else ""
            lines.append(f"section: {section}{suffix}")

    header = "\n".join(lines)

    return f"---\n{header}\n---\n{content}"


def fetch_unembedded(conn, limit):
    #select rows with missing embedding.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, source_type, park_name, title, content, metadata
            FROM documents_chunks
            WHERE embedding IS NULL
            ORDER BY id
            LIMIT %s;
        """, (limit,))

        return cur.fetchall()
    
def embed_texts(texts):
    """
    Send a batch of texts to OpenAI, return their vectors in the same order.

    Raises RuntimeError with a clear message on failure. Rows in this batch
    stay embedding IS NULL either way, so a later re-run just retries them.
    """
    try:
        response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    except Exception as error:
        raise RuntimeError(f"OpenAI embedding call failed: {error}") from error

    return [item.embedding for item in response.data]

def to_vector_literal(vector):
    """psycopg2 has no native pgvector type; send it as text and cast in SQL."""
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


def save_embeddings(conn, ids, vectors):
    with conn.cursor() as cur:
        for chunk_id, vector in zip(ids, vectors):
            cur.execute(
                "UPDATE documents_chunks SET embedding = %s::vector WHERE id = %s;",
                (to_vector_literal(vector), chunk_id),
            )
    conn.commit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="embed at most this many chunks (for testing)")
    args = parser.parse_args()

    conn = get_db_connection()
    done = 0

    try:
        while True:
            remaining = FETCH_BATCH if args.limit is None else args.limit - done
            if remaining <= 0:
                break

            rows = fetch_unembedded(conn, min(FETCH_BATCH, remaining))
            if not rows:
                break

            for i in range(0, len(rows), EMBED_BATCH):
                batch = rows[i:i + EMBED_BATCH]

                texts = [
                    build_embed_text(source_type, park_name, title, content, metadata)
                    for _, source_type, park_name, title, content, metadata in batch
                ]

                try:
                    vectors = embed_texts(texts)
                except RuntimeError as error:
                    print(f"\n{error}")
                    print(f"Stopping after {done} chunk(s). Already-embedded "
                          "rows are saved -- re-run to pick up where this left off.")
                    return

                save_embeddings(conn, [row[0] for row in batch], vectors)

                done += len(batch)
                print(f"Embedded {done} chunk(s)...")

    finally:
        conn.close()

    print(f"\nFinished: {done} chunk(s) embedded.")


if __name__ == "__main__":
    main()