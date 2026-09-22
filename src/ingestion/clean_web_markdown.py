"""
clean_web_markdown.py

Shared cleanup for scraped visitcalifornia.com pages (Firecrawl markdown).

Used by:
    scrape_park_articles.py  -> cleans the page BEFORE saving to raw_web_page
    chunk_web_pages.py       -> re-applies it (safe to run twice) before chunking

Removes: site header/nav above the article, footer/newsletter/related-article
carousels below it, images, photo credits, map URLs. Turns [text](url) into text.
"""

import re
from urllib.parse import urlparse


# The first heading that starts the page footer. Everything from here down
# (related-article carousels, newsletter form, country dropdown) is not content.
FOOTER_HEADINGS = {
    "official resources",
    "related articles",
    "planning resources",
    "explore by interest",
    "subscribe to our newsletter",
}


# Page type
# ---------------------------------------------------------

def derive_page_type(page):
    """
    Use the stored page_type if set; otherwise take the first URL path
    segment, e.g. visitcalifornia.com/experience/... -> "experience".
    """

    if page.get("page_type"):
        return page["page_type"]

    segments = urlparse(page["url"] or "").path.strip("/").split("/")

    return segments[0] or None


# ---------------------------------------------------------


# Markdown cleanup
# ---------------------------------------------------------

# Photo credits, e.g. "benedek/Getty Images", "Getty Images/iStockPhoto"
CREDIT_RE = re.compile(
    r"^.{0,80}\b(?:Getty Images|iStock(?:Photo)?|Alamy|Shutterstock)\b.{0,20}$",
    re.IGNORECASE,
)


def is_boilerplate_line(line):
    """
    Return True for common navigation / advertising lines
    that should not become RAG chunks.

    Keep these rules conservative so real article text is
    not accidentally removed.
    """

    stripped = line.strip()

    if not stripped:
        return False

    if "access_token=" in stripped or "api.mapbox.com" in stripped:
        return True

    if CREDIT_RE.match(stripped):
        return True

    if stripped.lower() == "1  mapbox homepage" or stripped.endswith("Mapbox homepage"):
        return True

    if stripped.startswith("Please select:"):
        return True

    exact_patterns = [
        r"^Advertisement$",
        r"^Close$",
        r"^\\+$",                                   # stray backslash lines
        r"^Use keyboard arrow keys to move through items\.?$",
        r"^\[Skip to content\]\(.+\)$",
        r"^\[Travel Alerts\]\(.+\)$",
        r"^\[Search\]\(.+\)$",
        r"^\[Visit California, Return Home\]\(.+\)$",
    ]

    return any(
        re.match(pattern, stripped, re.IGNORECASE)
        for pattern in exact_patterns
    )


def strip_page_chrome(markdown):
    """
    Remove the site header above the article and the footer below it.

    - Header: everything before the first H1 (nav, search, ads, breadcrumbs).
    - Footer: everything from the first footer heading onward.
    """

    lines = markdown.split("\n")

    # Header: start at the first H1 (all pages have one). If a page has
    # none, keep everything and let the line-level cleanup handle it.
    for i, line in enumerate(lines):
        if re.match(r"^#\s+\S", line):
            lines = lines[i:]
            break

    # Footer: cut at the first footer heading.
    for i, line in enumerate(lines):
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading and heading.group(1).strip().lower() in FOOTER_HEADINGS:
            lines = lines[:i]
            break

    return "\n".join(lines)


# A "related article" card in a carousel:
#   - [Title](url)
#   ![](image)
#   ### Title
#   one-line blurb
TEASER_CARD_RE = re.compile(
    r"^- \[[^\]]+\]\([^\n]*\)[ \t]*\n"      # - [Title](url)
    r"!\[[^\]]*\]\([^\n]*\)[ \t]*\n"        # ![](image)
    r"\s*###[ \t]+[^\n]+\n"                    # ### Title
    r"(?:\s*[^\s#-][^\n]*\n)?",               # optional blurb
    re.MULTILINE,
)

# "[View All Restaurants](url)" on its own line
VIEW_ALL_RE = re.compile(r"^\[View All[^\]]*\]\([^\n]*\)[ \t]*$", re.MULTILINE | re.IGNORECASE)


def remove_teaser_cards(markdown):
    markdown = TEASER_CARD_RE.sub("", markdown)
    return VIEW_ALL_RE.sub("", markdown)


def simplify_links_and_images(markdown):
    """
    Images carry no text for embedding, and link URLs waste tokens.

        ![alt](url)      -> removed
        [text](url)      -> text

    URLs may contain one level of parentheses (e.g. Mapbox map URLs).
    """

    url = r"\((?:[^()\n]|\([^()\n]*\))*\)"

    markdown = re.sub(r"!\[[^\]]*\]" + url, "", markdown)
    markdown = re.sub(r"\[([^\]]+)\]" + url, r"\1", markdown)

    return markdown


def clean_markdown(markdown):
    """
    Clean scraped Markdown while preserving useful structure,
    especially Markdown headings and paragraph boundaries.
    """

    if not markdown:
        return ""

    # Normalize line endings.
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")

    markdown = strip_page_chrome(markdown)
    markdown = remove_teaser_cards(markdown)
    markdown = simplify_links_and_images(markdown)

    cleaned_lines = []

    for line in markdown.splitlines():
        # Remove trailing whitespace but keep leading Markdown syntax.
        line = line.rstrip()

        if is_boilerplate_line(line):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # Collapse 3+ blank lines down to 2.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


# ---------------------------------------------------------


def is_dead_page(title):
    """Pages the scraper saved but that are really error pages, e.g. '404 | Visit California'."""

    return (title or "").strip().lower().startswith("404")
