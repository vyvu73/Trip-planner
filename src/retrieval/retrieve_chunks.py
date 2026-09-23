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


def search_chunks(conn, query_vector, question, k=5):
    """
    Vector search over documents_chunks, optionally scoped to one park.

    detect_park_code works in codes (matches NPS's own parkCode, and is
    immune to punctuation/casing in the display name). The SQL filter
    itself is on park_name -- that column is normalized to one value per
    park and already has its own index.
    """
    park_code = detect_park_code(question)
    park_name = PARK_DISPLAY_NAMES.get(park_code) if park_code else None

    vector = to_vector_literal(query_vector)          # from load_chunk.py
    where_clause = "WHERE park_name = %s" if park_name else ""

    params = [vector]
    if park_name:
        params.append(park_name)
    params += [vector, k]

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, title, park_name, content, embedding <=> %s AS distance
            FROM documents_chunks
            {where_clause}
            ORDER BY embedding <=> %s
            LIMIT %s;
        """, params)

        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def retrieve_chunks(question, k=5):
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
