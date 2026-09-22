import os
import re
import json
import psycopg2
import tiktoken

from dotenv import load_dotenv

from park_names import normalize_park
from clean_web_markdown import (
    clean_markdown,
    derive_page_type,
    is_dead_page,
)


load_dotenv()


MAX_TOKENS = 500
OVERLAP_TOKENS = 50
MIN_CHUNK_TOKENS = 20   # drop fragments smaller than this (unless it's the only chunk)

# Sections that are link lists / media embeds, not article text.
DROP_SECTION_RE = re.compile(
    r"^(find more things to do|podcasts|more things to do\b.*|.*\bvideos)$",
    re.IGNORECASE,
)

encoder = tiktoken.encoding_for_model("text-embedding-3-small")


# ---------------------------------------------------------
# Database
# ---------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        database="trip_planner",
    )


def fetch_web_pages(conn):
    """
    Retrieve raw web pages that contain content.
    """

    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                id,
                source,
                park_name,
                page_type,
                title,
                url,
                content
            FROM raw_web_page
            WHERE content IS NOT NULL
              AND TRIM(content) <> ''
            ORDER BY id;
        """)

        columns = [desc[0] for desc in cur.description]

        return [
            dict(zip(columns, row))
            for row in cur.fetchall()
        ]


# ---------------------------------------------------------
# Token helpers
# ---------------------------------------------------------

def count_tokens(text):
    return len(encoder.encode(text))


def split_text_by_tokens(
    text,
    max_tokens=MAX_TOKENS,
    overlap_tokens=OVERLAP_TOKENS
):
    """
    Fallback splitter for a single paragraph that is larger
    than MAX_TOKENS.

    Example:
        chunk 1 -> tokens 0-499
        chunk 2 -> tokens 450-949
    """

    tokens = encoder.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))

        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(tokens):
            break

        start = end - overlap_tokens

    return chunks


# ---------------------------------------------------------
# Markdown structure parsing
# ---------------------------------------------------------

def parse_markdown_sections(markdown):
    """
    Split Markdown into sections while preserving heading hierarchy.

    Example:

        # Plan Your Visit
        ## Hiking
        ### Mist Trail

    produces a heading path like:

        ["Plan Your Visit", "Hiking", "Mist Trail"]
    """

    sections = []

    # Example:
    # {
    #     1: "Plan Your Visit",
    #     2: "Hiking",
    #     3: "Mist Trail"
    # }
    headings = {}

    current_content = []
    current_heading_path = []

    def save_section():
        content = "\n".join(current_content).strip()

        if content:
            sections.append({
                "heading_path": current_heading_path.copy(),
                "content": content
            })

    for line in markdown.splitlines():
        heading_match = re.match(
            r"^(#{1,6})\s+(.+?)\s*$",
            line
        )

        if heading_match:
            # Save the text belonging to the previous section.
            save_section()
            current_content.clear()

            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            headings[level] = heading_text

            # If we move back up the heading hierarchy,
            # remove child headings.
            for existing_level in list(headings):
                if existing_level > level:
                    del headings[existing_level]

            current_heading_path = [
                headings[key]
                for key in sorted(headings)
            ]

        else:
            current_content.append(line)

    # Save the final section.
    save_section()

    return sections


# ---------------------------------------------------------
# Paragraph chunking
# ---------------------------------------------------------

def split_into_paragraphs(text):
    """
    Split section content at blank lines.
    """

    paragraphs = re.split(r"\n\s*\n", text)

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def chunk_section(section, max_tokens=MAX_TOKENS):
    """
    Build chunks from paragraphs without exceeding MAX_TOKENS.

    Paragraph boundaries are preserved whenever possible.
    A single oversized paragraph falls back to token-based splitting.
    """

    paragraphs = split_into_paragraphs(section["content"])

    chunk_texts = []
    current_paragraphs = []

    for paragraph in paragraphs:

        # One paragraph is already too large.
        if count_tokens(paragraph) > max_tokens:
            if current_paragraphs:
                chunk_texts.append(
                    "\n\n".join(current_paragraphs)
                )
                current_paragraphs = []

            chunk_texts.extend(
                split_text_by_tokens(paragraph, max_tokens)
            )

            continue

        candidate = "\n\n".join(
            current_paragraphs + [paragraph]
        )

        if count_tokens(candidate) <= max_tokens:
            current_paragraphs.append(paragraph)

        else:
            if current_paragraphs:
                chunk_texts.append(
                    "\n\n".join(current_paragraphs)
                )

            current_paragraphs = [paragraph]

    if current_paragraphs:
        chunk_texts.append(
            "\n\n".join(current_paragraphs)
        )

    return [
        {
            "heading_path": section["heading_path"],
            "content": text
        }
        for text in chunk_texts
    ]


# ---------------------------------------------------------
# Chunk complete web page
# ---------------------------------------------------------

def chunk_web_page(page):
    """
    Clean and chunk one raw_web_page row.
    """

    # Dead pages that the scraper saved (e.g. "404 Error").
    if is_dead_page(page["title"]):
        return []

    markdown = clean_markdown(page["content"])

    if not markdown:
        return []

    sections = parse_markdown_sections(markdown)

    sections = [
        section for section in sections
        if not (
            section["heading_path"]
            and DROP_SECTION_RE.match(section["heading_path"][-1])
        )
    ]

    chunks = []

    for section in sections:
        chunks.extend(
            chunk_section(section)
        )

    # Drop tiny fragments (photo credits, captions, one-line leftovers).
    substantial = [
        c for c in chunks
        if count_tokens(c["content"]) >= MIN_CHUNK_TOKENS
    ]
    if substantial:
        chunks = substantial

    # Fallback for content where section parsing produced nothing.
    if not chunks:
        fallback_section = {
            "heading_path": [],
            "content": markdown
        }

        chunks = chunk_section(fallback_section)

    return chunks


# ---------------------------------------------------------
# Save chunks
# ---------------------------------------------------------

def save_chunks(conn, page, chunks):
    """
    Replace all existing chunks for this raw page and insert
    the newly generated chunks.

    Deleting first prevents stale chunks if a page used to
    produce more chunks than it does after re-processing.
    """

    source_table = "raw_web_page"
    source_id = str(page["id"])
    park_code, park_name = normalize_park(page["park_name"])

    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM documents_chunks
            WHERE source_table = %s
              AND source_id = %s;
        """, (
            source_table,
            source_id
        ))

        for chunk_index, chunk in enumerate(chunks):
            content = chunk["content"]

            metadata = {
                "park_code": park_code,
                "heading_path": chunk["heading_path"],
                "page_type": derive_page_type(page),
                "original_source": page["source"]
            }

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
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb,
                    %s
                );
            """, (
                "web_page",
                source_table,
                source_id,
                chunk_index,
                park_name,
                page["title"],
                page["url"],
                content,
                json.dumps(metadata),
                count_tokens(content)
            ))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    conn = get_db_connection()

    try:
        pages = fetch_web_pages(conn)

        print(f"Found {len(pages)} web page(s)")

        for page_number, page in enumerate(pages, start=1):
            print(
                f"\n[{page_number}/{len(pages)}] "
                f"{page['park_name']} - {page['title']}"
            )

            chunks = chunk_web_page(page)

            print(f"Created {len(chunks)} chunk(s)")

            save_chunks(
                conn,
                page,
                chunks
            )

            conn.commit()

        print("\nFinished chunking web pages.")

    except Exception as error:
        conn.rollback()
        print(f"\nError: {error}")
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
