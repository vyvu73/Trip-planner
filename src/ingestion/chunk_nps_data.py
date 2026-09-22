"""
chunk_nps_data.py

Goal:
    Read raw_data_nps, chunk each record, and save the chunks to
    documents_chunks (source_type = 'nps').

Strategy (see docs / guideline):
    - Keep a record whole when it is <= MAX_TOKENS.
    - Split longer records on labelled fields ("Directions:", "Stops:", ...),
      then on lines, then by tokens as a last resort.
    - No minimum chunk size: short NPS records are real content.
    - content is clean text only. The context header (title, park, section)
      is built later, at embedding time, from the columns and metadata.

Usage (run from src/ingestion, like chunk_web_pages.py):
    python chunk_nps_data.py --dry-run     # print stats, write nothing
    python chunk_nps_data.py               # write chunks to documents_chunks
"""

import argparse
import json
import re
from collections import Counter

from chunk_web_pages import count_tokens, get_db_connection, split_text_by_tokens
from park_names import normalize_park


# ============================================================
# Settings
# ============================================================

MAX_TOKENS = 500

# Records of these types are always split field by field once they are
# over MAX_TOKENS, instead of packing several fields into one chunk.
SPLIT_BY_FIELD = {"park", "fees_passes"}

# Article records hold only a short listing description (about 290 chars).
LOW_PRIORITY_TYPES = {"article"}

# Token budget kept free when splitting a block: room for the
# "Label (continued): " prefix, and for decode/re-encode drift of 1-2 tokens.
CONTINUATION_PREFIX_TOKENS = 12

# A block is a repeat if its first N characters appear in a longer block.
REPEAT_MATCH_CHARS = 60


# ============================================================
# Database
# ============================================================

def fetch_nps_records(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, source_id, park_name, page_type, title, url, content
            FROM raw_data_nps
            WHERE content IS NOT NULL
              AND TRIM(content) <> ''
            ORDER BY id;
        """)

        columns = [desc[0] for desc in cur.description]

        return [dict(zip(columns, row)) for row in cur.fetchall()]


def save_chunks(conn, record, chunks):
    """
    Replace all chunks for this raw record. Deleting first avoids stale
    chunks when a record now produces fewer chunks than before.
    """

    source_table = "raw_data_nps"
    source_id = str(record["id"])            # not nps source_id: that is only
                                             # unique together with page_type
    park_code, park_name = normalize_park(record["park_name"])

    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM documents_chunks
            WHERE source_table = %s
              AND source_id = %s;
        """, (source_table, source_id))

        for chunk_index, chunk in enumerate(chunks):
            metadata = {
                "park_code": park_code,
                "page_type": record["page_type"],
                "nps_source_id": record["source_id"],
                "section": chunk["section"],
                "part": chunk_index + 1,
                "parts": len(chunks),
            }

            if record["page_type"] in LOW_PRIORITY_TYPES:
                metadata["priority"] = "low"

            cur.execute("""
                INSERT INTO documents_chunks (
                    source_type, source_table, source_id, chunk_index,
                    park_name, title, url, content, metadata, token_count
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s);
            """, (
                "nps",
                source_table,
                source_id,
                chunk_index,
                park_name,
                record["title"],
                record["url"] or None,       # tours and parking lots have no url
                chunk["content"],
                json.dumps(metadata),
                count_tokens(chunk["content"]),
            ))


# ============================================================
# Step 1: Break content into labelled blocks
# ============================================================

# "Directions: ...", "Entrance fees:\n..." -> label at the start of a block
LABEL_RE = re.compile(r"^([A-Z][A-Za-z &/'’-]{1,40}):")


def parse_blocks(content):
    """
    The transformers join fields with blank lines, so each field is one
    block. Returns [{"label": "Directions" | None, "text": "..."}].

    Blocks that are only a label ("Difficulty:" with no value) are dropped.
    """

    blocks = []

    for raw in re.split(r"\n\s*\n", content):
        text = raw.strip()

        if not text:
            continue

        match = LABEL_RE.match(text)

        if match and not text[match.end():].strip():
            continue

        blocks.append({
            "label": match.group(1) if match else None,
            "text": text,
        })

    return blocks


def drop_repeated_blocks(blocks):
    """
    For place records the transformer joins listingDescription and bodyText,
    and the short description usually reappears at the start of the body.
    Drop a block whose opening text is already inside a longer block.
    """

    kept = []

    for i, block in enumerate(blocks):
        opening = block["text"][:REPEAT_MATCH_CHARS]

        repeated = len(opening) >= REPEAT_MATCH_CHARS and any(
            j != i
            and len(other["text"]) > len(block["text"])
            and opening in other["text"]
            for j, other in enumerate(blocks)
        )

        if not repeated:
            kept.append(block)

    return kept


# ============================================================
# Step 2: Split one oversized block
# ============================================================

def split_block(label, text, max_tokens=MAX_TOKENS):
    """
    Split a block that is over max_tokens: pack whole lines first, and cut
    a single over-long line by tokens (with overlap) as a last resort.

    Continuation pieces start with "Label (continued): " so they stay
    understandable on their own.
    """

    if count_tokens(text) <= max_tokens:
        return [text]

    limit = max_tokens - CONTINUATION_PREFIX_TOKENS

    pieces = []
    current = []

    def flush():
        if current:
            pieces.append("\n".join(current))
            current.clear()

    for line in text.split("\n"):
        if count_tokens(line) > limit:
            flush()
            pieces.extend(split_text_by_tokens(line, max_tokens=limit))
            continue

        if current and count_tokens("\n".join(current + [line])) > limit:
            flush()

        current.append(line)

    flush()

    if label:
        for i in range(1, len(pieces)):
            pieces[i] = f"{label} (continued): {pieces[i]}"

    return pieces


# ============================================================
# Step 3: Chunk one record
# ============================================================

def chunk_record(record):
    """
    Returns [{"content": str, "section": str | None}, ...]
    """

    blocks = drop_repeated_blocks(parse_blocks(record["content"] or ""))

    if not blocks:
        return []

    whole = "\n\n".join(block["text"] for block in blocks)

    # Rule 1: a record that fits stays one chunk.
    if count_tokens(whole) <= MAX_TOKENS:
        return [{"content": whole, "section": None}]

    # Rule 2: split oversized blocks, then decide how to group the pieces.
    pieces = []

    for block in blocks:
        for text in split_block(block["label"], block["text"]):
            pieces.append((block["label"], text))

    if record["page_type"] in SPLIT_BY_FIELD:
        return [{"content": text, "section": label} for label, text in pieces]

    # Everything else: pack whole blocks together up to MAX_TOKENS.
    chunks = []
    current_texts = []
    current_label = None

    for label, text in pieces:
        too_big = count_tokens("\n\n".join(current_texts + [text])) > MAX_TOKENS

        if current_texts and too_big:
            chunks.append({
                "content": "\n\n".join(current_texts),
                "section": current_label,
            })
            current_texts = []

        if not current_texts:
            current_label = label

        current_texts.append(text)

    if current_texts:
        chunks.append({
            "content": "\n\n".join(current_texts),
            "section": current_label,
        })

    return chunks


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="chunk and print stats, but write nothing")
    args = parser.parse_args()

    conn = get_db_connection()

    try:
        records = fetch_nps_records(conn)
        print(f"Found {len(records)} NPS record(s)")

        total_chunks = 0
        over_max = 0
        by_type = Counter()
        unknown_parks = Counter()

        for record in records:
            chunks = chunk_record(record)

            if not chunks:
                continue

            if normalize_park(record["park_name"])[0] is None:
                unknown_parks[record["park_name"]] += 1

            over_max += sum(count_tokens(c["content"]) > MAX_TOKENS for c in chunks)
            total_chunks += len(chunks)
            by_type[record["page_type"]] += len(chunks)

            if not args.dry_run:
                save_chunks(conn, record, chunks)
                conn.commit()

        mode = "DRY RUN (nothing written)" if args.dry_run else "Saved"
        print(f"\n{mode}: {total_chunks} chunk(s), {over_max} over {MAX_TOKENS} tokens")

        for page_type, count in by_type.most_common():
            print(f"  {page_type:<15} {count}")

        if unknown_parks:
            print(f"\nPark names with no park_code: {dict(unknown_parks)}")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
