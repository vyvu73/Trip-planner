# Estimate the gas cost of a drive from EIA weekly retail gasoline prices
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

EIA_API_KEY = os.getenv("EIA_API_KEY")
EIA_URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
# EIA is slow and uneven (1-22s seen for the same query), hence the long
# timeout, the cache, and the warm-up at app start.
TIMEOUT_SECONDS = 30
DEFAULT_MPG = 25
# EIA publishes weekly; refetching more often than this gains nothing.
CACHE_TTL_SECONDS = 6 * 3600

# States EIA prices on their own (duoarea codes checked 2026-09-24).
STATE_AREAS = {
    "CA": "SCA", "CO": "SCO", "FL": "SFL", "MA": "SMA", "MN": "SMN",
    "NY": "SNY", "OH": "SOH", "TX": "STX", "WA": "SWA",
}

# Every other state falls back to its PADD region. PADD 5 uses the
# "except California" series so California's prices don't skew it.
STATE_TO_PADD = {
    **dict.fromkeys(["CT", "ME", "MA", "NH", "RI", "VT", "DE", "DC", "MD", "NJ", "NY",
                     "PA", "FL", "GA", "NC", "SC", "VA", "WV"], "R10"),
    **dict.fromkeys(["IL", "IN", "IA", "KS", "KY", "MI", "MN", "MO", "NE", "ND", "SD",
                     "OH", "OK", "TN", "WI"], "R20"),
    **dict.fromkeys(["AL", "AR", "LA", "MS", "NM", "TX"], "R30"),
    **dict.fromkeys(["CO", "ID", "MT", "UT", "WY"], "R40"),
    **dict.fromkeys(["AK", "AZ", "HI", "NV", "OR", "WA"], "R5XCA"),
}

AREA_LABELS = {
    "SCA": "California", "SCO": "Colorado", "SFL": "Florida", "SMA": "Massachusetts",
    "SMN": "Minnesota", "SNY": "New York", "SOH": "Ohio", "STX": "Texas", "SWA": "Washington",
    "R10": "East Coast (PADD 1)", "R20": "Midwest (PADD 2)", "R30": "Gulf Coast (PADD 3)",
    "R40": "Rocky Mountain (PADD 4)", "R5XCA": "West Coast except California (PADD 5)",
    "NUS": "U.S.",
}

# area -> (fetched_at, price, week)
price_cache = {}


def fetch_price(area):
    """Latest weekly regular-gasoline price for one EIA area, or None if EIA has none."""
    response = requests.get(
        EIA_URL,
        params={
            "api_key": EIA_API_KEY,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[product][]": "EPMR",
            "facets[duoarea][]": area,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 1,
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    rows = response.json()["response"]["data"]
    if not rows or rows[0].get("value") is None:
        return None
    return float(rows[0]["value"]), rows[0]["period"]


def gas_price(state):
    """Most specific price for the state: state, then PADD region, then U.S.
    Returns (price, area_label, week). Errors propagate."""
    state = (state or "").strip().upper()
    areas = [STATE_AREAS.get(state), STATE_TO_PADD.get(state), "NUS"]

    for area in [area for area in areas if area]:
        cached = price_cache.get(area)
        if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
            return cached[1], AREA_LABELS[area], cached[2]
        result = fetch_price(area)
        if result:
            price, week = result
            price_cache[area] = (time.time(), price, week)
            return price, AREA_LABELS[area], week
    raise LookupError("EIA returned no gasoline price")


def describe_error(error):
    """Error text safe to log: the EIA key is in the request URL, and
    requests puts that URL in HTTP and connection error messages."""
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return f"HTTP {error.response.status_code}"
    if isinstance(error, requests.RequestException):
        return type(error).__name__
    return str(error)


def warm_gas_price():
    """Fetch California's price once so the first cost question is fast."""
    if not EIA_API_KEY:
        return
    try:
        gas_price("CA")
    except Exception as error:
        print(f"warm_gas_price failed: {describe_error(error)}")


def get_transportation_cost(miles, origin_state, mpg=None):
    """Tool: gas cost of a drive, one way and round trip, with the working shown."""
    if not miles or miles <= 0:
        return "Miles must be a positive number from get_route."
    assumed = not mpg or mpg <= 0
    mpg = DEFAULT_MPG if assumed else mpg
    if not EIA_API_KEY:
        return "Gas prices are unavailable right now."

    try:
        price, area, week = gas_price(origin_state)
    except Exception as error:
        print(f"get_transportation_cost failed: {describe_error(error)}")
        return "Gas prices are unavailable right now."

    gallons = miles / mpg
    one_way = gallons * price
    mpg_note = f"{mpg:g} mpg (assumed default; the user can give their car's mpg)" if assumed else f"{mpg:g} mpg (from the user)"
    return "\n".join([
        "Gas cost estimate, regular gasoline:",
        f"- {miles:.0f} miles / {mpg_note} = {gallons:.1f} gallons",
        f"- Price: ${price:.2f}/gal, {area} average, EIA week of {week}",
        f"- One way: ${one_way:.2f}",
        f"- Round trip: ${one_way * 2:.2f}",
        "Gas only: excludes tolls, parking, park entrance fees, and wear on the car.",
    ])
