# Tools the answer model can call: their schemas and a dispatcher
import json

from src.ingestion.park_names import PARK_DISPLAY_NAMES
from src.tools.route import get_route
from src.tools.transportation_cost import get_transportation_cost
from src.tools.weather import get_weather

PARK_CODES = list(PARK_DISPLAY_NAMES)   # Kings Canyon is under seki

# Responses API function schemas. strict mode requires every property to be
# listed in "required"; optional values are typed as nullable instead.
TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "Live National Weather Service forecast for the next few days at one "
            "representative spot in a supported park. Only useful when the trip "
            "is within about the next 7 days."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "park_code": {"type": "string", "enum": PARK_CODES},
            },
            "required": ["park_code"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_route",
        "description": (
            "Driving distance and time between two places in the US, plus the "
            "origin's state. Use for drive times to a park, between places in or "
            "near a park, and before estimating gas cost."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Where the drive starts: a US city or place such as 'San Jose, CA', or a park code.",
                },
                "destination": {
                    "type": "string",
                    "description": (
                        "A park code (e.g. 'yose') for the park's main visitor center, "
                        "or a specific place such as 'Glacier Point, Yosemite' or a gateway town."
                    ),
                },
            },
            "required": ["origin", "destination"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_transportation_cost",
        "description": (
            "Gas cost of a drive, one way and round trip, from current EIA gasoline "
            "prices. Call get_route first and pass its miles and origin state."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "miles": {"type": "number", "description": "One-way miles from get_route."},
                "origin_state": {"type": "string", "description": "Two-letter US state of the origin, from get_route."},
                "mpg": {
                    "type": ["number", "null"],
                    "description": "The user's car mpg if they gave it, otherwise null.",
                },
            },
            "required": ["miles", "origin_state", "mpg"],
            "additionalProperties": False,
        },
    },
]

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_route": get_route,
    "get_transportation_cost": get_transportation_cost,
}

# Shown in the chat while a tool runs.
TOOL_STATUS = {
    "get_weather": "Checking the forecast",
    "get_route": "Looking up the drive",
    "get_transportation_cost": "Estimating gas cost",
}


def run_tool(name, arguments_json):
    """Run one tool call. Always returns text for the model, never raises."""
    print(f"tool call: {name}({arguments_json})")
    try:
        arguments = json.loads(arguments_json)
        return TOOL_FUNCTIONS[name](**arguments)
    except Exception as error:
        print(f"run_tool {name} failed: {error}")
        return f"Tool {name} failed: {error}"
