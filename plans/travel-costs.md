# Travel costs (flight + driving tools)

Status: planned, not implemented.

## Goal

Answer "how much does it cost to get there?" for a user's home city. The
answer lists two separate costs, with no combined total:

- **Driving, one way (estimate):** gas cost from the user's city to the park.
- **Flight, round trip, per person:** cheapest fare to the park's airport.

This is the app's first use of LLM tool calling. Unlike alerts, the router
does not fetch anything; the answer model decides when to call the tools.

## Design decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Mechanism | OpenAI tool calling inside `stream_response`. The model decides when to call. |
| 2 | Tools | `estimate_driving_cost` and `search_flights`. |
| 3 | Scope | Travel cost only. No rental car, no entrance fee, no airport-to-park drive, no Channel Islands boat fare. |
| 4 | Presentation | Driving and flight listed separately, each labeled. Never summed. |
| 5 | Origin | A city the user gives. |
| 6 | Driving formula | `one_way_miles / mpg = gallons`, `gallons * price_per_gallon = cost`. Computed in Python, never by the model. Shown as an estimate with the working. |
| 7 | Distance | OpenRouteService road route, origin city to a fixed park visitor-center point. |
| 8 | Origin geocoding | OpenRouteService geocoder (same key). Also supplies the origin state for the gas price. |
| 9 | MPG | Default 25. Use the user's number if given. |
| 10 | Gas price | EIA weekly retail regular gasoline. Most specific match for the origin: city, then state, then PADD region, then U.S. average. The answer names the one used and its week. |
| 11 | Drive direction | One way. |
| 12 | Flight API | SerpApi, `engine=google_flights`. |
| 13 | Park airport | Fixed table (below). |
| 14 | Origin airport | The model passes the IATA code for the user's city as a tool argument. |
| 15 | Flight dates | Required. The model asks the user for travel dates before calling `search_flights`. |
| 16 | Flight shape | Round trip (`type=1`), 1 adult, economy, USD. Priced per person. |
| 17 | Flight output | Cheapest option (price, airlines, stops, total duration), `price_insights` (price level, typical range), and the Google Flights URL. |
| 18 | Tool loop | At most 3 tool rounds; after that the model must answer in text. Parallel calls in one round are allowed. |
| 19 | Tool errors | Caught and returned to the model as the tool output (`{"error": "..."}`). The model tells the user that cost is unavailable. The request does not fail. |
| 20 | Caching | None. Every call hits the API. Each flight question uses one SerpApi search (~100 free/month). |
| 21 | Timeouts | None (user's choice). A hung API call hangs that chat. |

## Park table

`PARK_TRAVEL` in `src/tools/park_travel.py`, keyed by the codes in
`PARK_DISPLAY_NAMES`. Coordinates checked against OpenStreetMap on
2026-09-24.

| Code | Visitor center | Lat, lon | Airport |
|---|---|---|---|
| chis | Robert J. Lagomarsino VC, Ventura Harbor | 34.2485, -119.2666 | LAX |
| deva | Furnace Creek VC | 36.4614, -116.8670 | LAS |
| jotr | Joshua Tree VC | 34.1339, -116.3156 | PSP |
| lavo | Kohm Yah-mah-nee VC | 40.4378, -121.5338 | RDD |
| pinn | Pinnacles VC (east side) | 36.4937, -121.1464 | SJC |
| redw | Thomas H. Kuchel VC, Orick | 41.2869, -124.0909 | ACV |
| seki | Foothills VC | 36.4909, -118.8252 | FAT |
| yose | Yosemite Valley VC | 37.7487, -119.5872 | FAT |

Channel Islands' driving cost ends at the Ventura Harbor dock; the answer
notes the boat is not included.

## Flow

```
user message
  └─► route_message (unchanged)
        ├─ out_of_scope / ambiguous ──► reply as today
        └─ in_scope
             ├─► retrieve_chunks(...)
             ├─► alerts_context(...)
             └─► stream_response(message, chunks, recent, alerts)
                   round 1..3:
                     responses.create(tools=TOOLS, stream=True)
                       ├─ text deltas ──► yield to UI
                       └─ function_call items ──► run_tool(name, args)
                             └─► next round with function_call_output items
                   after round 3: tool_choice="none" ──► text only
```

## Changes

### `src/tools/park_travel.py` (new)

- `PARK_TRAVEL = {"yose": {"airport": "FAT", "lat": ..., "lon": ...}, ...}`.

### `src/tools/driving_cost.py` (new)

- Env: `ORS_API_KEY`, `EIA_API_KEY`.
- `geocode(city) -> (lon, lat, state_abbrev)`: ORS `GET /geocode/search`
  with `text=city`, `size=1`. State from `features[0].properties.region_a`.
  Raises if nothing matches.
- `road_miles(origin, destination) -> float`: ORS
  `POST /v2/directions/driving-car` with `[[lon, lat], [lon, lat]]`;
  `routes[0].summary.distance` meters converted to miles.
- `gas_price(city, state) -> (price, label, week)`: EIA v2
  `GET /v2/petroleum/pri/gnd/data/` with `frequency=weekly`, `data[]=value`,
  `facets[product][]=EPMR` (regular), `facets[duoarea][]=<code>`, sorted by
  `period` desc, `length=1`. Uses `value` and `period` from `response.data[0]`.
  Tries `duoarea` codes in order (checked 2026-09-24):
  - City: `Y05LA` Los Angeles, `Y05SF` San Francisco, `Y48SE` Seattle,
    `YBOS` Boston, `YORD` Chicago, `YCLE` Cleveland, `YDEN` Denver,
    `Y44HO` Houston, `YMIA` Miami, `Y35NY` New York City.
  - State: `SCA`, `SCO`, `SFL`, `SMA`, `SMN`, `SNY`, `SOH`, `STX`, `SWA`.
  - Region: `STATE_TO_PADD` maps each state to `R10`..`R40`; PADD 5 states
    other than California use `R5XCA` (PADD 5 except California).
  - Fallback: `NUS`.
- `estimate_driving_cost(origin_city, park_code, mpg=25) -> dict`:
  `{origin, park, one_way_miles, mpg, gallons, price_per_gallon,
  price_source, price_week, estimated_cost}`, rounded for display.

### `src/tools/flights.py` (new)

- Env: `SERPAPI_API_KEY`.
- `search_flights(origin_airport, park_code, outbound_date, return_date) -> dict`:
  SerpApi `GET https://serpapi.com/search` with `engine=google_flights`,
  `departure_id`, `arrival_id=PARK_TRAVEL[code]["airport"]`, dates,
  `type=1`, `adults=1`, `currency=USD`, `hl=en`.
- Picks the cheapest of `best_flights` + `other_flights`. Returns
  `{origin_airport, destination_airport, outbound_date, return_date,
  price, airlines, stops, total_duration_minutes, price_level,
  typical_price_range, google_flights_url}`.
- No flights found: returns `{"error": "no flights found"}`.

### `src/tools/__init__.py` (new)

- `TOOLS`: the two Responses API function schemas (`type: "function"`,
  `strict: true`). `park_code` is an enum of the 8 codes. Dates are
  `YYYY-MM-DD`. `mpg` is a number.
- Tool descriptions say: use the nearest major airport's IATA code for
  the user's city; only call `search_flights` once the user has given
  travel dates.
- `run_tool(name, arguments_json) -> str`: dispatches, catches every
  exception, returns JSON (`{"error": str(error)}` on failure).

### `src/generation/generate_response.py`

- `stream_response` becomes a loop of at most `MAX_TOOL_ROUNDS = 3`:
  - Stream as today, yielding text deltas.
  - Collect `function_call` items from `response.output_item.done`, and the
    response id from `response.completed`.
  - No calls: done. Otherwise run each through `run_tool` and call
    `responses.create` again with `previous_response_id`, the
    `function_call_output` items, the same `instructions`, `tools` and
    `reasoning`.
  - The round after the cap uses `tool_choice="none"`.
- `generate_response` is unused and stays as it is (no tools).
- System prompt gains a "Travel costs" section:
  - Use the tools for cost-to-get-there questions; never estimate prices
    yourself.
  - Ask for travel dates before searching flights.
  - List "Driving, one way (estimate)" with the working (miles, MPG,
    gallons, price per gallon and its source/week) and "Flight, round trip,
    per person" with airline, stops, duration, price level, typical range
    and the Google Flights link. Never add them together.
  - If a tool returns an error, say that cost is unavailable right now.
  - Channel Islands: the boat to the islands is not included.

### `app.py`

- No change: `stream_answer` already consumes `stream_response`.

### `.env`

- Add `SERPAPI_API_KEY`, `EIA_API_KEY`, `ORS_API_KEY`.

## Tests

No test makes a live API call.

`tests/test_travel_tools.py` (mock `requests`):
- Driving math: miles / MPG * price, with a custom MPG.
- EIA fallback order: city, then state, then PADD, then U.S.
- SerpApi parsing: cheapest across `best_flights` and `other_flights`,
  `price_insights` fields, no-results error.
- `run_tool` turns an exception into `{"error": ...}`.

`tests/test_tool_loop.py` (mock the OpenAI client):
- Text-only reply streams as before.
- A function call is run and its output sent back with
  `previous_response_id`, then the text streams.
- Two calls in one round both run.
- The loop stops after 3 rounds with `tool_choice="none"`.

Checked by hand in the app: the model asks for dates before searching
flights, and never sums the two costs.

## Not doing

Caching, timeouts, rental cars, entrance fees, airport-to-park driving,
boat fares, multiple travelers, one-way flights, an in-chat "checking
prices" status.
