"""
find_park_article_links.py   (file 2 of 3)

Goal:
    Read the hub page .md file (saved by ingest_raw_web_page.py) and find
    the article links in each park section. Each link gets the park name
    of the section it was found in.

No JSON file:
    scrape_park_articles.py imports get_park_article_links() from this
    file and uses the list directly.

No Firecrawl, no database. Free to run as often as you like.

Run it on its own to just look at the links:
    uv run find_park_article_links.py
"""

import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit


# ============================================================
# Settings
# ============================================================

HUB_FILE_FOLDER = Path("crawl_test_output")
HUB_URL = "https://www.visitcalifornia.com/things-to-do/national-parks/"

# Only keep links to real pages on the main site.
# (drupal-prod.visitcalifornia.com is where the pictures live.)
ALLOWED_HOSTS = {"www.visitcalifornia.com", "visitcalifornia.com"}

# Links ending in these are files (pictures, PDFs, media), not pages.
FILE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg",
                   ".pdf", ".mp3", ".mp4")

PARK_ALIASES = {
    "Yosemite National Park": ["yosemite", "half dome", "glacier point"],
    "Sequoia & Kings Canyon National Parks": ["sequoia", "kings canyon"],
    "Redwood National and State Parks": ["redwood"],
    "Lassen Volcanic National Park": ["lassen"],
    "Pinnacles National Park": ["pinnacles"],
    "Channel Islands National Park": ["channel islands"],
    "Joshua Tree National Park": ["joshua tree"],
    "Death Valley National Park": ["death valley"],
}

# Sections we never take links from (junk + non-park places).
SKIP_HEADING_WORDS = ["video", "podcast", "newsletter", "subscribe",
                      "events in california", "military sites",
                      "alcatraz", "point reyes"]


# ============================================================
# Step 1: Read the hub page .md file
# ============================================================

def load_hub_markdown():
    files = sorted(HUB_FILE_FOLDER.glob("01-*.md"))
    if not files:
        raise FileNotFoundError(
            f"No hub page found in {HUB_FILE_FOLDER}/. "
            "Run ingest_raw_web_page.py first."
        )
    return files[0].read_text(encoding="utf-8")


# ============================================================
# Step 2: Split the page into sections by heading
# ============================================================

HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def clean_heading(text):
    """'**Yosemite National Park**' -> 'Yosemite National Park'"""
    return text.strip().strip("*").strip()


def split_into_sections(markdown_text):
    """
    Each section keeps its heading path, for example:
        ["National Parks", "Featured National Parks",
         "Things to Do in Yosemite National Park"]
    """
    matches = list(HEADER_PATTERN.finditer(markdown_text))
    sections, stack = [], []

    for i, match in enumerate(matches):
        level = len(match.group(1))

        # Going back up to the same or higher level? Remove old headings.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, clean_heading(match.group(2))))

        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        sections.append({
            "path": [heading for _, heading in stack],
            "text": markdown_text[match.start():end],
        })

    return sections


# ============================================================
# Step 3: Figure out the park, and clean up links
# ============================================================

def find_park(text):
    """Return the park whose name appears in the text, or None. Never guesses."""
    lowered = text.lower().replace("-", " ")
    for park_name, aliases in PARK_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return park_name
    return None


# Matches the "(url)" part of markdown links: [text](url) and ![alt](url)
LINK_URL_PATTERN = re.compile(r"\]\(\s*([^)\s]+)")


def clean_url(raw_url):
    """Turn a raw link into a full page URL, or None if we don't want it."""
    parts = urlsplit(urljoin(HUB_URL, raw_url))   # handles "/experience/..." links

    if parts.netloc not in ALLOWED_HOSTS:
        return None                                # other site or picture server
    if parts.path.lower().endswith(FILE_EXTENSIONS):
        return None                                # picture / file

    # Drop ?query and #anchor, use one trailing-slash style
    url = urlunsplit(("https", "www.visitcalifornia.com",
                      parts.path.rstrip("/") + "/", "", ""))

    return None if url == HUB_URL else url         # skip link back to the hub


# ============================================================
# The main function (scrape_park_articles.py imports this)
# ============================================================

def get_park_article_links():
    """
    Returns a list like:
        [{"url": "https://www.visitcalifornia.com/experience/...",
          "park_name": "Yosemite National Park"}, ...]
    """
    markdown_text = load_hub_markdown()

    links = {}          # url -> park_name
    conflicts = set()   # urls found under two different parks

    for section in split_into_sections(markdown_text):
        joined_path = " ".join(section["path"]).lower()

        if any(word in joined_path for word in SKIP_HEADING_WORDS):
            continue                               # junk / non-park

        park_name = find_park(joined_path)
        if park_name is None:
            continue                               # unresolved

        for raw_url in LINK_URL_PATTERN.findall(section["text"]):
            url = clean_url(raw_url)
            if url is None:
                continue

            # Safety check: if the URL names a DIFFERENT park, skip it.
            park_in_url = find_park(urlsplit(url).path)
            if park_in_url and park_in_url != park_name:
                continue

            if url in links and links[url] != park_name:
                conflicts.add(url)
            links.setdefault(url, park_name)

    # Links that could belong to two parks are dropped, not guessed.
    for url in conflicts:
        del links[url]

    return [{"url": url, "park_name": park} for url, park in links.items()]


# ============================================================
# Run on its own: just print the links
# ============================================================

def main():
    links = get_park_article_links()

    print(f"Found {len(links)} article link(s):\n")
    for park_name in PARK_ALIASES:
        park_links = [item for item in links if item["park_name"] == park_name]
        if park_links:
            print(park_name)
            for item in park_links:
                print(f"   {item['url']}")
            print()


if __name__ == "__main__":
    main()