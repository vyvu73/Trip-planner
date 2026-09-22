"""
park_names.py

One park_code and one display name per park, shared by every chunker so
that documents_chunks can be filtered the same way for all sources:

    WHERE metadata->>'park_code' = 'yose'
"""


PARK_DISPLAY_NAMES = {
    "chis": "Channel Islands National Park",
    "deva": "Death Valley National Park",
    "jotr": "Joshua Tree National Park",
    "lavo": "Lassen Volcanic National Park",
    "pinn": "Pinnacles National Park",
    "redw": "Redwood National and State Parks",
    "seki": "Sequoia & Kings Canyon National Parks",
    "yose": "Yosemite National Park",
}

# Every spelling that appears in raw_data_nps (and the web data).
NAME_TO_CODE = {name: code for code, name in PARK_DISPLAY_NAMES.items()}
NAME_TO_CODE.update({
    "Sequoia National Park": "seki",
    "Kings Canyon National Park": "seki",   # NPS files Kings Canyon under seki
})


def normalize_park(park_name):
    """Return (park_code, display_name). Unknown names keep their own name."""
    code = NAME_TO_CODE.get((park_name or "").strip())

    if code is None:
        return None, park_name

    return code, PARK_DISPLAY_NAMES[code]
