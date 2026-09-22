"""
nps_transformers.py
 
Transform functions that convert raw NPS API JSON records into the flat
record shape expected by fetch_nps.py's insert_nps_record():
 
    {
        "park_name": str,
        "page_type": str,
        "title": str,
        "content": str,
        "url": str,
    }
 
Each transform_* function corresponds to one NPS API endpoint and is meant
to be robust to missing/optional fields, since not every record populates
every field consistently.
"""

import re
from src.ingestion.format.format_park_data import (
    join_parts, clean_html,
    format_fee_list,
    format_operating_hours, 
    format_address, 
    format_amenities,
    format_parking_accessibility,
    format_fees,
    format_passes

)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


PARK_CODE_NAMES = {
    "chis": "Channel Islands National Park",
    "deva": "Death Valley National Park",
    "jotr": "Joshua Tree National Park",
    "kica": "Kings Canyon National Park",
    "lavo": "Lassen Volcanic National Park",
    "pinn": "Pinnacles National Park",
    "redw": "Redwood National and State Parks",
    "seki": "Sequoia National Park",
    "yose": "Yosemite National Park",
}

def pick_park_name(related_parks):
    """
    A record can list several related parks, and the one we queried is not
    always first. Return the first one that is in PARK_CODE_NAMES (our scope).
    Returns "" if none match; ingest_nps.py then falls back to the park it queried.
    """
    for park in related_parks or []:
        if park.get("parkCode") in PARK_CODE_NAMES:
            return park.get("fullName", "")
    return ""


def transform_park(item):
    description = clean_html(item.get("description", ""))
    directions = clean_html(item.get("directionsInfo", ""))
    weather = clean_html(item.get("weatherInfo", ""))
    fees = format_fee_list(item.get("entranceFees"))
    hours = format_operating_hours(item.get("operatingHours"))
    addresses = format_address(item.get("addresses"))
    activities = ", ".join(a.get("name", "") for a in item.get("activities", []) if a.get("name"))
    topics = ", ".join(t.get("name", "") for t in item.get("topics", []) if t.get("name"))

    content = join_parts(
        description,
        f"Activities: {activities}" if activities else "",
        f"Topics: {topics}" if topics else "",
        f"Directions: {directions}" if directions else "",
        f"Weather: {weather}" if weather else "",
        f"Entrance fees:\n{fees}" if fees else "",
        f"Operating hours:\n{hours}" if hours else "",
        f"Address:\n{addresses}" if addresses else "",
    )

    return {
        "park_name": item.get("fullName", ""),
        "page_type": "park",
        "source_id": item.get("id", ""),
        "title": item.get("fullName", item.get("name", "")),
        "content": content,
        "url": item.get("url", ""),
    }

def transform_article(item):
    related_parks = item.get("relatedParks", [])
    park_name = pick_park_name(related_parks)

    content = join_parts(
        clean_html(item.get("listingDescription", "")),
    )
    return {
        "park_name": park_name,
        "page_type": "article",
        "source_id": item.get("id", ""),
        "title": item.get("title", ""),
        "content": content,
        "url": item.get("url", ""),
    }

def transform_place(item):
    related_parks = item.get("relatedParks", [])
    park_name = pick_park_name(related_parks)
    amenities = ", ".join(item.get("amenities", []))

    content = join_parts(
        clean_html(item.get("listingDescription", "")),
        clean_html(item.get("bodyText", "")),
        amenities,
    )
    return {
        "park_name": park_name,
        "page_type": "place",
        "source_id": item.get("id", ""),
        "title": item.get("title", ""),
        "content": content,
        "url": item.get("url", ""),
    }

def transform_thing_to_do(item):
    related_parks = item.get("relatedParks", [])
    park_name = pick_park_name(related_parks)

    activities = ", ".join(a.get("name", "") for a in item.get("activities", []))
    topics = ", ".join(t.get("name", "") for t in item.get("topics", []))
    season = ", ".join(item.get("season", []))

    content = join_parts(
        clean_html(item.get("shortDescription", "")),
        clean_html(item.get("longDescription", "")),
        f"Difficulty: {item.get('activityDescription', '')}",
        f"Duration: {item.get('duration', '')}",
        clean_html(item.get("durationDescription", "")),
        f"Location: {item.get('location', '')}",
        clean_html(item.get("locationDescription", "")),
        f"Season: {season}",
        f"Activities: {activities}",
        f"Topics: {topics}",
        clean_html(item.get("accessibilityInformation", "")),
        clean_html(item.get("petsDescription", "")),
    )
    return {
        "park_name": park_name,
        "page_type": "thing_to_do",
        "source_id": item.get("id", ""),
        "title": item.get("title", ""),
        "content": content,
        "url": item.get("url", ""),
    }

def transform_campground(item):
    park_code = item.get("parkCode", "")
    park_name = PARK_CODE_NAMES.get(park_code, park_code)

    campsites = item.get("campsites") or {}
    total_sites = campsites.get("totalSites", "")
    amenities = format_amenities(item.get("amenities"))

    content = join_parts(
        clean_html(item.get("description", "")),
        clean_html(item.get("directionsOverview", "")),
        clean_html(item.get("weatherOverview", "")),
        clean_html(item.get("regulationsOverview", "")),
        f"Total sites: {total_sites}",
        amenities,
    )
    return {
        "park_name": park_name,
        "page_type": "campground",
        "source_id": item.get("id", ""),
        "title": item.get("name", ""),
        "content": content,
        "url": item.get("url", ""),
    }

def transform_visitor_centers(item):
    park_code = item.get("parkCode", "")
    park_name = PARK_CODE_NAMES.get(park_code, park_code)

    amenities = ", ".join(item.get("amenities", []))
    hours = format_operating_hours(item.get("operatingHours"))
    addresses = format_address(item.get("addresses"))

    phones = item.get("contacts", {}).get("phoneNumbers", [])
    phone = phones[0].get("phoneNumber", "") if phones else ""
    emails = item.get("contacts", {}).get("emailAddresses", [])
    email = emails[0].get("emailAddress", "") if emails else ""

    content = join_parts(
        clean_html(item.get("description", "")),
        clean_html(item.get("directionsInfo", "")),
        amenities,
        hours,
        addresses,
        f"Phone: {phone}" if phone else "",
        f"Email: {email}" if email else "",
    )
    return {
        "park_name": park_name,
        "page_type": "visitor_center",
        "source_id": item.get("id", ""),
        "title": item.get("name", ""),
        "content": content,
        "url": item.get("url", ""),
    }

def transform_tour(item):
    park = item.get("park", {})
    park_name = park.get("fullName", "")

    topics = ", ".join(t.get("name", "") for t in item.get("topics", []))
    activities = ", ".join(a.get("name", "") for a in item.get("activities", []))

    duration_min = item.get("durationMin", "")
    duration_max = item.get("durationMax", "")
    duration_unit = item.get("durationUnit", "")
    duration = f"Duration: {duration_min}-{duration_max} {duration_unit}" if duration_min else ""

    stops = []
    for stop in item.get("stops", []):
        stops.append(join_parts(
            stop.get("assetName", ""),
            stop.get("significance", ""),
            stop.get("directionsToNextStop", ""),
            sep=" | ",
        ))
    stops_text = "\n".join(s for s in stops if s)

    content = join_parts(
        clean_html(item.get("description", "")),
        f"Topics: {topics}" if topics else "",
        f"Activities: {activities}" if activities else "",
        duration,
        f"Stops:\n{stops_text}" if stops_text else "",
    )
    return {
        "park_name": park_name,
        "page_type": "tour",
        "source_id": item.get("id", ""),
        "title": item.get("title", ""),
        "content": content,
        "url": "",
    }

def transform_fees_passes(item):
    park_code = item.get("parkCode", "")
    park_name = PARK_CODE_NAMES.get(park_code, park_code)

    content = join_parts(
        clean_html(item.get("entranceFeeDescription", "")),
        clean_html(item.get("entrancePassDescription", "")),
        format_fees(item.get("fees")),
        format_passes(item.get("passes")),
        clean_html(item.get("timedEntryDescription", "")),
        clean_html(item.get("paidParkingDescription", "")),
        clean_html(item.get("customFeeDescription", "")),
    )
    return {
        "park_name": park_name,
        "page_type": "fees_passes",
        "source_id": f"{park_code}-fees",
        "title": f"{park_name} Fees and Passes",
        "content": content,
        "url": item.get("feesAtWorkUrl", ""),
    }

def transform_parking_lot(item):
    related_parks = item.get("relatedParks", [])
    park_name = pick_park_name(related_parks)

    content = join_parts(
        clean_html(item.get("description", "")),
        format_parking_accessibility(item.get("accessibility")),
        format_fee_list(item.get("fees")),
        format_operating_hours(item.get("operatingHours")),
    )
    return {
        "park_name": park_name,
        "page_type": "parking_lot",
        "source_id": item.get("id", ""),
        "title": item.get("name", ""),
        "content": content,
        "url": "",
    }
