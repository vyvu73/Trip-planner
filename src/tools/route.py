# Driving distance and time between two places, via OpenRouteService
import os

import requests
from dotenv import load_dotenv

from src.ingestion.park_names import detect_park_code

load_dotenv()

ORS_API_KEY = (os.getenv("ORS_API_KEY") or "").strip()
ORS_BASE_URL = "https://api.openrouteservice.org"
TIMEOUT_SECONDS = 10
METERS_PER_MILE = 1609.344
# A geocoded trailhead or town may sit off the road network; let ORS snap
# each point to a road within this distance.
SNAP_RADIUS_METERS = 5000
# Place names repeat across the country ("Glacier Point" exists in Colorado).
# A place that names a park is searched only within this radius of it;
# anything else is biased toward central California.
PARK_SEARCH_RADIUS_KM = 50
CALIFORNIA_FOCUS = (37.0, -119.5)

# Where a drive "to the park" ends (lat, lon, label). Coordinates checked
# against OpenStreetMap on 2026-09-24 (see plans/travel-costs.md).
PARK_ROUTE_POINTS = {
    "chis": (34.2485, -119.2666, "Ventura Harbor (boats to Channel Islands leave here; boat fare not included)"),
    "deva": (36.4614, -116.8670, "Furnace Creek Visitor Center, Death Valley"),
    "jotr": (34.1339, -116.3156, "Joshua Tree Visitor Center"),
    "lavo": (40.4378, -121.5338, "Kohm Yah-mah-nee Visitor Center, Lassen Volcanic"),
    "pinn": (36.4937, -121.1464, "Pinnacles Visitor Center (east side)"),
    "redw": (41.2869, -124.0909, "Kuchel Visitor Center, Redwood"),
    "seki": (36.4909, -118.8252, "Foothills Visitor Center, Sequoia"),
    "yose": (37.7487, -119.5872, "Yosemite Valley Visitor Center"),
}


class PlaceNotFound(Exception):
    pass


def geocode(place):
    """ORS geocoder, US only. Returns (lat, lon, label, state_abbrev)."""
    params = {"text": place, "size": 1, "boundary.country": "US"}
    park_code = detect_park_code(place)
    if park_code:
        lat, lon, _ = PARK_ROUTE_POINTS[park_code]
        params.update({"boundary.circle.lat": lat, "boundary.circle.lon": lon,
                       "boundary.circle.radius": PARK_SEARCH_RADIUS_KM})
    else:
        params.update({"focus.point.lat": CALIFORNIA_FOCUS[0], "focus.point.lon": CALIFORNIA_FOCUS[1]})

    # Key in a header, not the URL, so it never shows up in logged errors.
    response = requests.get(
        f"{ORS_BASE_URL}/geocode/search",
        headers={"Authorization": ORS_API_KEY},
        params=params,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    features = response.json()["features"]
    if not features:
        raise PlaceNotFound(place)
    lon, lat = features[0]["geometry"]["coordinates"]
    properties = features[0]["properties"]
    return lat, lon, properties.get("label", place), properties.get("region_a")


def resolve(place):
    """A park code maps to its fixed point; anything else is geocoded."""
    code = place.strip().lower()
    if code in PARK_ROUTE_POINTS:
        lat, lon, label = PARK_ROUTE_POINTS[code]
        return lat, lon, label, "CA"
    return geocode(place)


def drive(start, end):
    """ORS driving route between two (lat, lon, ...) points. Returns (miles, seconds)."""
    response = requests.post(
        f"{ORS_BASE_URL}/v2/directions/driving-car",
        headers={"Authorization": ORS_API_KEY},
        json={
            "coordinates": [[start[1], start[0]], [end[1], end[0]]],
            "radiuses": [SNAP_RADIUS_METERS, SNAP_RADIUS_METERS],
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    summary = response.json()["routes"][0]["summary"]
    return summary.get("distance", 0) / METERS_PER_MILE, summary.get("duration", 0)


def get_route(origin, destination):
    """Tool: driving distance and time as text, or a plain message on failure."""
    if not ORS_API_KEY:
        return "Driving routes are unavailable right now."
    try:
        start = resolve(origin)
        end = resolve(destination)
        miles, seconds = drive(start, end)
    except PlaceNotFound as error:
        return f"Could not find a US place matching '{error}'. Ask the user for a US city or a more specific place."
    except requests.HTTPError as error:
        print(f"get_route failed: {error}")
        if error.response is not None and error.response.status_code == 404:
            return f"No driving route found between {origin} and {destination}."
        return "Driving routes are unavailable right now."
    except Exception as error:
        print(f"get_route failed: {error}")
        return "Driving routes are unavailable right now."

    hours, minutes = divmod(round(seconds / 60), 60)
    duration = f"{hours} h {minutes} min" if hours else f"{minutes} min"
    lines = [
        f"Driving route from {start[2]} to {end[2]}: {miles:.0f} miles, "
        f"about {duration} without stops or traffic.",
        "The route ignores traffic and seasonal road closures (e.g. Tioga Road in winter).",
    ]
    if start[3]:
        lines.append(f"Origin state: {start[3]}")
    return "\n".join(lines)
