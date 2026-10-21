# Fetch a live forecast for a supported park from the National Weather Service
import requests

from src.ingestion.park_names import PARK_DISPLAY_NAMES

NWS_BASE_URL = "https://api.weather.gov"
# NWS rejects requests without a User-Agent that identifies the app.
HEADERS = {"User-Agent": "trip-planner (https://github.com/vyvu73/Trip-planner)"}
TIMEOUT_SECONDS = 10
FORECAST_PERIODS = 8   # ~4 days: each day has a daytime and a night period

# One representative point per park (lat, lon, label). Weather varies a lot
# with elevation, so the label tells the model where the forecast applies.
PARK_FORECAST_POINTS = {
    "chis": (34.0497, -119.5560, "Scorpion Anchorage, Santa Cruz Island"),
    "deva": (36.4614, -116.8670, "Furnace Creek"),
    "jotr": (34.0128, -116.1681, "Hidden Valley"),
    "lavo": (40.4378, -121.5338, "Kohm Yah-mah-nee Visitor Center"),
    "pinn": (36.4937, -121.1464, "Pinnacles Visitor Center (east side)"),
    "redw": (41.2869, -124.0909, "Orick / Kuchel Visitor Center"),
    "seki": (36.5647, -118.7735, "Giant Forest, Sequoia"),
    "yose": (37.7487, -119.5872, "Yosemite Valley"),
}


def fetch_forecast_periods(lat, lon):
    """Two NWS calls: point -> forecast URL -> forecast periods. Errors propagate."""
    point = requests.get(f"{NWS_BASE_URL}/points/{lat},{lon}",
                         headers=HEADERS, timeout=TIMEOUT_SECONDS)
    point.raise_for_status()
    forecast_url = point.json()["properties"]["forecast"]

    forecast = requests.get(forecast_url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    forecast.raise_for_status()
    return forecast.json()["properties"]["periods"]


def get_weather(park_code):
    """Tool: forecast text for one park, or a plain message when unavailable."""
    if park_code not in PARK_FORECAST_POINTS:
        return f"No forecast location for park code '{park_code}'."
    lat, lon, label = PARK_FORECAST_POINTS[park_code]
    park = PARK_DISPLAY_NAMES[park_code]

    try:
        periods = fetch_forecast_periods(lat, lon)
    except Exception as error:
        print(f"get_weather failed: {error}")
        return (f"Weather is unavailable right now for {park}. "
                f"Tell the user to check https://www.nps.gov/{park_code}/planyourvisit/weather.htm")

    lines = [f"NWS forecast for {label} ({park}). Covers only the next few days; "
             f"conditions differ at other elevations in the park."]
    for period in periods[:FORECAST_PERIODS]:
        rain = (period.get("probabilityOfPrecipitation") or {}).get("value")
        rain_text = f", {rain}% chance of precipitation" if rain else ""
        lines.append(f"- {period['name']}: {period['temperature']}°{period['temperatureUnit']}, "
                     f"{period['shortForecast']}, wind {period['windSpeed']}{rain_text}")
    return "\n".join(lines)
