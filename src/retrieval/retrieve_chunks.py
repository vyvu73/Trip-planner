#takes the user query and returns relevant chunks
import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from src.ingestion.park_names import PARK_DISPLAY_NAMES, detect_park_code
from src.vector_store.load_chunk import to_vector_literal
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL")

REVIEW_SOURCE_TYPE = "tripadvisor"
# A review is kept only if it is at most this much farther (cosine distance)
# than the best fact chunk. Factual questions have a near-exact fact match,
# so reviews trail far behind; experience questions don't. Tuned on the
# plan's threshold prompts: "yes" margins <= 0.103, "no" margins >= 0.241.
REVIEW_MAX_MARGIN = 0.15

def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        database="trip_planner"
    )
def generate_embedding(text):
    ## embed the query
    ##condition to check if query is within scope.
    try:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=text)
    except Exception as error:
        raise RuntimeError(f"OpenAI embedding call failed: {error}") from error 
    return response.data[0].embedding


def run_vector_query(conn, vector, park_name, reviews, limit):
    """
    Nearest chunks to `vector`, either reviews only or everything but reviews,
    optionally scoped to one park.
    """
    conditions = ["source_type = %s" if reviews else "source_type <> %s"]
    params = [vector, REVIEW_SOURCE_TYPE]

    if park_name:
        conditions.append("park_name = %s")
        params.append(park_name)

    params += [vector, limit]

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, source_type, title, park_name, content, metadata,
                   embedding <=> %s AS distance
            FROM documents_chunks
            WHERE {" AND ".join(conditions)}
            ORDER BY embedding <=> %s
            LIMIT %s;
        """, params)

        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def search_chunks(conn, query_vector, question, k=7, review_k=2):
    """
    Vector search over documents_chunks, optionally scoped to one park.

    Reviews are searched separately so they neither crowd out facts nor
    vanish under ~3,600 fact chunks: up to review_k reviews are kept, and only
    if within REVIEW_MAX_MARGIN of the best fact. Unused review slots are backfilled with
    facts, so k chunks come back whenever that many exist.

    detect_park_code works in codes (matches NPS's own parkCode, and is
    immune to punctuation/casing in the display name). The SQL filter
    itself is on park_name -- that column is normalized to one value per
    park and already has its own index.
    """
    park_code = detect_park_code(question)
    park_name = PARK_DISPLAY_NAMES.get(park_code) if park_code else None

    vector = to_vector_literal(query_vector)          # from load_chunk.py

    facts = run_vector_query(conn, vector, park_name, reviews=False, limit=k)
    best_fact = facts[0]["distance"] if facts else None

    reviews = [
        chunk
        for chunk in run_vector_query(conn, vector, park_name, reviews=True, limit=review_k)
        if best_fact is None or chunk["distance"] - best_fact < REVIEW_MAX_MARGIN
    ]

    return facts[:k - len(reviews)] + reviews


def retrieve_chunks(question, k=7):
    """
    One-call entry point for chatpot.py: embed the question, open a
    connection, run the (optionally park-scoped) vector search, and
    return the chunk rows.
    """
    conn = get_db_connection()

    try:
        query_vector = generate_embedding(question)
        return search_chunks(conn, query_vector, question, k=k)
    finally:
        conn.close()
