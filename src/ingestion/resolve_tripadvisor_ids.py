"""
resolve_tripadvisor_ids.py

One-off helper: for every TRIPADVISOR_LOCATIONS entry without a location_id,
search Tripadvisor and print the top 3 candidates. Prints only -- paste the
chosen id into tripadvisor_locations.py by hand.

Run:
    uv run python src/ingestion/resolve_tripadvisor_ids.py
"""

import time

from ingest_tripadvisor import get_primary_name, search_location
from park_names import PARK_ALIASES, normalize_park
from tripadvisor_locations import TRIPADVISOR_LOCATIONS


TOP_N = 3
SEARCH_DELAY_SECONDS = 1   # search endpoint allows about 1 request/second


def build_query(entry):
    """
    Search matches on the start of the name, so extra words (e.g. the park
    name) return nothing. Search the plain name and judge the park by geo.
    """
    return entry["location_name"]


def in_park(entry, location):
    """True if the candidate's geo/address mentions one of the park's aliases."""
    park_code, _ = normalize_park(entry["park_name"])
    addresses = location.get("addresses") or [{}]
    where = f"{location.get('geo')} {addresses[0].get('formatted')}".lower()
    return any(alias in where for alias in PARK_ALIASES[park_code])


def print_candidates(entry, query, results):
    print(f"\n{entry['location_name']}  ({entry['park_name']})")
    print(f"query: {query}")
    print("-" * 70)

    if not results:
        print("No locations found.")
        return

    for result in results[:TOP_N]:
        location = result["location"]
        addresses = location.get("addresses") or [{}]
        overall = location.get("traveler_ratings", {}).get("overall", {})

        marker = "*" if in_park(entry, location) else " "

        print(
            f"{marker} id={location.get('id')}  "
            f"name={get_primary_name(location)!r}  "
            f"geo={location.get('geo')!r}  "
            f"address={addresses[0].get('formatted')!r}  "
            f"reviews={overall.get('count')}"
        )


def main():
    missing = [entry for entry in TRIPADVISOR_LOCATIONS if entry["location_id"] is None]
    print(f"{len(missing)} location(s) without an id  (* = geo/address matches the park)")

    for entry in missing:
        query = build_query(entry)
        data = search_location(query)
        print_candidates(entry, query, data.get("data", []))
        time.sleep(SEARCH_DELAY_SECONDS)


if __name__ == "__main__":
    main()
