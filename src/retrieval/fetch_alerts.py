# Fetch live NPS alerts for the supported parks, cached in process memory
import os
import threading
import time

import requests
from dotenv import load_dotenv

from src.ingestion.park_names import PARK_DISPLAY_NAMES
load_dotenv()

API_KEY = os.getenv("NPS_API_KEY")
BASE_URL = os.getenv("NPS_BASE_URL")
ALERT_PARK_CODES = list(PARK_DISPLAY_NAMES)   # 8 codes; Kings Canyon is under seki

CACHE_TTL_SECONDS = 3600
FAILURE_BACKOFF_SECONDS = 3600
TIMEOUT_SECONDS = 5
PAGE_SIZE = 500

# Most plan-changing first; unknown categories sort last.
CATEGORY_ORDER = {"Park Closure": 0, "Danger": 1, "Caution": 2, "Information": 3}

# alerts is None when the last fetch failed.
cache = {"fetched_at": None, "alerts": None}
cache_lock = threading.Lock()


def fetch_all_alerts():
    """One request for every supported park. Errors propagate.
    /alerts only honors the first code in a comma-separated parkCode list,
    so fetch all of California and keep the supported parks."""
    if not API_KEY or not BASE_URL:
        raise RuntimeError("NPS_API_KEY or NPS_BASE_URL is not set")
    response = requests.get(
        f"{BASE_URL}/alerts",
        headers={"X-Api-Key": API_KEY},
        params={"stateCode": "CA", "limit": PAGE_SIZE},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return [alert for alert in response.json()["data"] if alert.get("parkCode") in ALERT_PARK_CODES]


def cache_expired(now):
    if cache["fetched_at"] is None:
        return True
    max_age = FAILURE_BACKOFF_SECONDS if cache["alerts"] is None else CACHE_TTL_SECONDS
    return now - cache["fetched_at"] >= max_age


def get_alerts(park_codes):
    """
    Alerts for park_codes, most severe first.
    returns: list of alert dicts, or None if live alerts are unavailable.
    get_alerts([]) just warms the cache.
    """
    with cache_lock:
        now = time.time()
        if cache_expired(now):
            try:
                cache["alerts"] = fetch_all_alerts()
            except Exception as error:
                print(f"get_alerts failed: {error}")
                cache["alerts"] = None
            cache["fetched_at"] = now
        alerts = cache["alerts"]

    if alerts is None:
        return None
    matching = [alert for alert in alerts if alert.get("parkCode") in park_codes]
    return sorted(matching, key=lambda alert: CATEGORY_ORDER.get(alert.get("category"), 99))


def format_alerts(alerts):
    """Render alerts as prompt text, one block per alert."""
    blocks = []
    for alert in alerts:
        park = PARK_DISPLAY_NAMES.get(alert.get("parkCode"), alert.get("parkCode"))
        updated = (alert.get("lastIndexedDate") or "")[:10]   # "2026-09-20 14:03:11.0" -> date
        header = f"[{park}] {alert.get('category')}: {alert.get('title')}"
        if updated:
            header += f" (updated {updated})"
        block = f"{header}\n{alert.get('description', '').strip()}"
        if alert.get("url"):
            block += f"\nMore: {alert['url']}"
        blocks.append(block)
    return "\n\n".join(blocks)


def alerts_context(park_codes):
    """Prompt text for the alerts section; "" when no park applies."""
    if not park_codes:
        return ""
    alerts = get_alerts(park_codes)
    if alerts is None:
        pages = ", ".join(f"https://www.nps.gov/{code}/planyourvisit/conditions.htm" for code in park_codes)
        return f"Live alerts are unavailable right now. Tell the user to check {pages} before going."
    if not alerts:
        names = ", ".join(PARK_DISPLAY_NAMES[code] for code in park_codes)
        return f"No active NPS alerts for {names}."
    return format_alerts(alerts)
