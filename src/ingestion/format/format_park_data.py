import re

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

def clean_html(text):
    """Strip HTML tags and collapse whitespace. Safe on None/empty input."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return _WS_RE.sub(" ", text).strip()


def join_parts(*parts, sep="\n\n"):
    """Join non-empty string parts with a separator, skipping blanks."""
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return sep.join(cleaned)


def format_amenities(amenities):
    """
    /campgrounds shape: a dict (or a list containing one dict) with mixed
    value types (strings, lists, booleans). Not for /places or
    /visitorcenters, where amenities is already a flat list of strings.
    """
    if not amenities:
        return ""
    if isinstance(amenities, list):
        amenities = amenities[0]
    lines = []
    for key, value in amenities.items():
        if isinstance(value, list):
            value = ", ".join(value)
        elif isinstance(value, bool):
            value = "Yes" if value else "No"
        if value == "":
            continue
        lines.append(f"{key}: {value}")
    return ", ".join(lines)


def format_operating_hours(operating_hours):
    """
    standardHours is a DICT on /visitorcenters and /parkinglots, but a LIST
    containing one dict on /parks. Handles both.
    """
    lines = []
    for entry in operating_hours or []:
        name = entry.get("name", "")
        desc = clean_html(entry.get("description", ""))

        standard = entry.get("standardHours") or {}
        if isinstance(standard, list):
            standard = standard[0] if standard else {}

        days = ", ".join(f"{day}: {hrs}" for day, hrs in standard.items() if hrs)
        line = join_parts(name, desc, days, sep=" | ")
        if line:
            lines.append(line)
    return "\n".join(lines)


def format_address(addresses):
    """postalCode is sometimes an int, sometimes a string — hence str()."""
    lines = []
    for addr in addresses or []:
        parts = [addr.get("line1", ""), addr.get("city", ""),
                 addr.get("stateCode", ""), addr.get("postalCode", "")]
        line = ", ".join(str(p) for p in parts if p)
        if line:
            lines.append(f"{addr.get('type', '')}: {line}")
    return "\n".join(lines)


def format_fee_list(fees):
    """For fees shaped {cost, title, description} — /parkinglots and /parks."""
    lines = []
    for fee in fees or []:
        cost = fee.get("cost", "")
        lines.append(join_parts(
            fee.get("title", ""),
            f"${cost}" if cost else "",
            clean_html(fee.get("description", "")),
            sep=" | ",
        ))
    return "\n".join(line for line in lines if line)


def format_fees(fees):
    """
    For /feespasses, where fees use entranceFeeType instead of title.
    Different key from format_fee_list above — don't mix them up.
    """
    lines = []
    for fee in fees or []:
        cost = fee.get("cost", "")
        lines.append(join_parts(
            fee.get("entranceFeeType", ""),
            f"${cost}" if cost else "",
            clean_html(fee.get("description", "")),
            clean_html(fee.get("paymentDescription", "")),
            sep=" | ",
        ))
    return "\n".join(line for line in lines if line)


def format_passes(passes):
    """For /feespasses passes[] — {category, cost, description}."""
    lines = []
    for p in passes or []:
        cost = p.get("cost", "")
        lines.append(join_parts(
            p.get("category", ""),
            f"${cost}" if cost else "",
            clean_html(p.get("description", "")),
            sep=" | ",
        ))
    return "\n".join(line for line in lines if line)


def format_parking_accessibility(acc):
    """
    /parkinglots: accessibility is a DICT (on /campgrounds it's a list).
    Note NPS's own casing: numberofAdaSpaces (lowercase 'of') vs
    numberOfOversizeVehicleSpaces (capital 'Of').
    """
    acc = acc or {}
    total = acc.get("totalSpaces", "")
    ada = acc.get("numberofAdaSpaces", "")
    oversize = acc.get("numberOfOversizeVehicleSpaces", "")
    return join_parts(
        f"Total spaces: {total}" if total else "",
        f"ADA spaces: {ada}" if ada else "",
        f"Oversize vehicle spaces: {oversize}" if oversize else "",
        clean_html(acc.get("adaFacilitiesDescription", "")),
        sep=" | ",
    )