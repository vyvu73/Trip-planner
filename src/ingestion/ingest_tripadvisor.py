import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TRIPADVISOR_API_KEY")
BASE_URL = "https://terra.tripadvisor.com/api"

HEADERS = {
    "X-API-Key": API_KEY,
    "accept": "application/json",
}


ATTRACTIONS = [
    "Yosemite National Park",
    "Glacier Point",
    "Yosemite Falls",
    "Tunnel View",
    "Half Dome",
    "Mariposa Grove",
]


def search_location(query):
    """
    Search Tripadvisor for a location by name.
    """

    url = f"{BASE_URL}/locations/search"

    params = {
        "query": query,
        "country_code": "US",
        "category": "ATTRACTION",
        "size": 5,
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
    )

    response.raise_for_status()

    return response.json()


def fetch_reviews(location_id):
    """
    Fetch available English reviews for one Tripadvisor location.
    """

    url = f"{BASE_URL}/locations/{location_id}/reviews"

    params = {
        "language": "en",
        "sort_by": "MOST_RECENT",
        "page": 1,
        "size": 20,
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
    )

    response.raise_for_status()

    return response.json()


def get_primary_name(location):
    """
    Get the primary English name from a Tripadvisor location.
    """

    names = location.get("names", [])

    for name in names:
        if name.get("primary"):
            return name.get("value")

    if names:
        return names[0].get("value")

    return None


def print_search_results(query, data):
    """
    Print locations returned by Tripadvisor search.
    """

    print(f"\nSEARCH: {query}")
    print("-" * 70)

    results = data.get("data", [])

    if not results:
        print("No locations found.")
        return

    for result in results:
        location = result["location"]

        print("ID:", location.get("id"))
        print("NAME:", get_primary_name(location))
        print("AREA:", location.get("geo"))

        rating = (
            location
            .get("traveler_ratings", {})
            .get("overall", {})
        )

        print("RATING:", rating.get("rating"))
        print("REVIEW COUNT:", rating.get("count"))

        print("-" * 70)


def print_reviews(location_id, data):
    """
    Print the reviews returned for one location.
    """

    reviews = data.get("data", [])

    print(f"\nREVIEWS FOR LOCATION {location_id}")
    print(f"Available reviews: {len(reviews)}")
    print("=" * 70)

    for review in reviews:

        titles = review.get("title", [])
        texts = review.get("text", [])

        title = titles[0]["value"] if titles else ""
        text = texts[0]["value"] if texts else ""

        print("REVIEW ID:", review.get("id"))
        print("RATING:", review.get("rating"))
        print("TRIP TYPE:", review.get("trip_type"))
        print("TRAVEL DATE:", review.get("travel_date"))
        print("TITLE:", title)
        print("TEXT:", text)
        print("URL:", review.get("url"))

        print("=" * 70)


def main():

    for attraction in ATTRACTIONS:

        search_data = search_location(attraction)

        print_search_results(
            attraction,
            search_data,
        )

        results = search_data.get("data", [])

        if not results:
            # Search endpoint has a stricter rate limit.
            time.sleep(1)
            continue

        # For now, just use the FIRST search result.
        # Later you can improve how the correct POI is selected.
        location = results[0]["location"]

        location_id = location["id"]
        location_name = get_primary_name(location)

        print(
            f"\nUsing: {location_name} "
            f"(location_id={location_id})"
        )

        reviews = fetch_reviews(location_id)

        print_reviews(
            location_id,
            reviews,
        )

        # Tripadvisor search is limited to about
        # 1 request per second, so don't hammer it.
        time.sleep(1)


if __name__ == "__main__":
    main()

    