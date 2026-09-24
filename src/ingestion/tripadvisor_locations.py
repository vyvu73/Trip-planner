"""
tripadvisor_locations.py

Hand-written list of Tripadvisor attractions to pull reviews from.

- park_name must be a spelling normalize_park accepts (park_names.py).
- location_id is resolved once by hand with resolve_tripadvisor_ids.py and
  pasted in here. Fetch runs never search; None means "skip with a warning".
"""


TRIPADVISOR_LOCATIONS = [
    # Channel Islands
    # Tripadvisor lists the islands, not spots on them (Scorpion Anchorage, Inspiration Point).
    {"park_name": "Channel Islands National Park", "location_name": "Santa Cruz Island", "location_id": "141863"},
    {"park_name": "Channel Islands National Park", "location_name": "Anacapa Island", "location_id": "108635"},
    {"park_name": "Channel Islands National Park", "location_name": "Santa Rosa Island", "location_id": "1058176"},
    {"park_name": "Channel Islands National Park", "location_name": "Robert J. Lagomarsino Visitor Center", "location_id": "3502192"},

    # Death Valley
    {"park_name": "Death Valley National Park", "location_name": "Badwater", "location_id": "116979"},
    {"park_name": "Death Valley National Park", "location_name": "Zabriskie Point", "location_id": "116978"},
    {"park_name": "Death Valley National Park", "location_name": "Dante's View", "location_id": "531585"},
    {"park_name": "Death Valley National Park", "location_name": "Artists Palette", "location_id": "532011"},
    {"park_name": "Death Valley National Park", "location_name": "Mesquite Flat Sand Dunes", "location_id": "2255762"},

    # Joshua Tree
    {"park_name": "Joshua Tree National Park", "location_name": "Hidden Valley", "location_id": None},
    {"park_name": "Joshua Tree National Park", "location_name": "Keys View", "location_id": "116981"},
    {"park_name": "Joshua Tree National Park", "location_name": "Cholla Cactus Garden", "location_id": "116982"},
    {"park_name": "Joshua Tree National Park", "location_name": "Skull Rock", "location_id": "1846385"},
    {"park_name": "Joshua Tree National Park", "location_name": "Barker Dam Trail", "location_id": "1114199"},

    # Lassen Volcanic
    {"park_name": "Lassen Volcanic National Park", "location_name": "Bumpass Hell", "location_id": "145160"},
    {"park_name": "Lassen Volcanic National Park", "location_name": "Lassen Volcanic National Park Hiking Trails", "location_id": "4355193"},
    {"park_name": "Lassen Volcanic National Park", "location_name": "Sulphur Works", "location_id": "7702497"},
    {"park_name": "Lassen Volcanic National Park", "location_name": "Manzanita Lake", "location_id": "145161"},

    # Pinnacles
    {"park_name": "Pinnacles National Park", "location_name": "Pinnacles National Park", "location_id": "143240"},
    {"park_name": "Pinnacles National Park", "location_name": "Bear Gulch Caves", "location_id": "142802"},
    {"park_name": "Pinnacles National Park", "location_name": "High Peaks Trail", "location_id": "23963595"},
    {"park_name": "Pinnacles National Park", "location_name": "Balconies Cliffs Trails", "location_id": "8514010"},

    # Redwood
    {"park_name": "Redwood National and State Parks", "location_name": "Fern Canyon", "location_id": "145139"},
    {"park_name": "Redwood National and State Parks", "location_name": "Lady Bird Johnson Grove", "location_id": "2655827"},
    {"park_name": "Redwood National and State Parks", "location_name": "Tall Trees Grove", "location_id": "116944"},
    {"park_name": "Redwood National and State Parks", "location_name": "Stout Grove", "location_id": "2227085"},

    # Sequoia & Kings Canyon
    {"park_name": "Sequoia & Kings Canyon National Parks", "location_name": "General Sherman Tree", "location_id": "146450"},
    {"park_name": "Sequoia & Kings Canyon National Parks", "location_name": "Moro Rock Trail", "location_id": "146443"},
    {"park_name": "Sequoia & Kings Canyon National Parks", "location_name": "Congress Trail", "location_id": "6650757"},
    {"park_name": "Sequoia & Kings Canyon National Parks", "location_name": "Crystal Cave", "location_id": None},
    {"park_name": "Sequoia & Kings Canyon National Parks", "location_name": "General Grant Tree Trail", "location_id": "207980"},
    {"park_name": "Sequoia & Kings Canyon National Parks", "location_name": "Zumwalt Meadow Trail", "location_id": "4102300"},

    # Yosemite
    {"park_name": "Yosemite National Park", "location_name": "Glacier Point", "location_id": "139187"},
    {"park_name": "Yosemite National Park", "location_name": "Yosemite Falls", "location_id": "483476"},
    {"park_name": "Yosemite National Park", "location_name": "Tunnel View", "location_id": "593212"},
    {"park_name": "Yosemite National Park", "location_name": "Mist Trail", "location_id": "483482"},
    {"park_name": "Yosemite National Park", "location_name": "Mariposa Grove of Giant Sequoias", "location_id": "592681"},
]
