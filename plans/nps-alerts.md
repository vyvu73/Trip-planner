# Live NPS alerts

Status: planned, not implemented.

## Goal

Include current NPS alerts (closures, dangers, cautions) in answers so trip
plans don't send people to closed roads or trails. Alerts come live from the
NPS `/alerts` endpoint and are never stored in the database or embedded.

## Design decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Storage | Live at query time plus an in-process cache. Nothing is written to Postgres or the vector store. |
| 2 | Fetch scope | One request with `stateCode=CA` (`/alerts` only honors the first code in a comma-separated `parkCode` list), filtered to the 8 supported park codes and cached as a single entry. `kica` is left out: NPS files Kings Canyon under `seki`. `ingest_nps.py` is not changed. |
| 3 | Cache TTL (success) | 1 hour. The first request after expiry refetches. |
| 4 | Startup | `app.py` fetches once before `demo.launch()` to warm the cache. |
| 5 | Failure | Timeout, 5xx, rate limit, or missing key: log it, return `None`. No stale fallback. |
| 6 | Failure backoff | After a failure, don't call NPS again for 1 hour (`FAILURE_BACKOFF_SECONDS`, kept separate from the success TTL). Answers during that hour get the "check nps.gov" line immediately. |
| 7 | Timeout | 5s (ingestion's 30s is too long for chat). |
| 8 | Concurrency | A `threading.Lock` around fetch-and-store so simultaneous requests on an expired cache trigger one NPS call. |
| 9 | Which park | The router gets a new `park_codes: list[str]` field, resolved from the message and history. |
| 10 | No park named | An alerts or closures question with no park in the message or history routes to `ambiguous` and asks which park. |
| 11 | When alerts are included | Every in-scope answer where `park_codes` is non-empty, not only alert questions. |
| 12 | Using alerts in answers | New system-prompt rule: mention an alert only when it affects the plan or question, and cite its updated date. |
| 13 | Fields | `title`, `category`, `description`, `url`, plus the park name. |
| 14 | Date | `lastIndexedDate`, labeled "updated", never presented as a start or end date. Today's date goes in the prompt so the model can judge dates written in descriptions. |
| 15 | Order | Park Closure and Danger first, then Caution, then Information. |
| 16 | Prompt text | `alerts_context(park_codes)` in `fetch_alerts.py` builds the finished alerts text: "" when no park, "No active NPS alerts for <park>" when none, the conditions-page fallback when unavailable, else `format_alerts`. `stream_response` takes that string as `alerts`. |

## Flow

```
app start ──► get_alerts() warm-up ──► cache (ok for 1h / failed for 1h)

user message
  └─► route_message ──► RouteDecision(route, standalone_question, park_codes, ...)
        ├─ out_of_scope / ambiguous ──► reply as today
        └─ in_scope
             ├─► retrieve_chunks(...)
             ├─► get_alerts(park_codes) ──► cache hit, or fetch if expired
             └─► stream_response(message, chunks, alerts=..., recent=history)
```

## Changes

### `src/retrieval/fetch_alerts.py` (new)

- Reuses `NPS_BASE_URL`, `NPS_API_KEY` from `src/ingestion/ingest_nps.py`.
- `ALERT_PARK_CODES = [code for code in PARK_CODES if code != "kica"]` (8 codes; Kings Canyon alerts come under `seki`).
- Module-level cache: `{"fetched_at": float, "alerts": list | None}`. `alerts is None` means the last fetch failed.
- Constants: `CACHE_TTL_SECONDS = 3600`, `FAILURE_BACKOFF_SECONDS = 3600`, `TIMEOUT_SECONDS = 5`.
- `get_alerts(park_codes) -> list[dict] | None`
  - Under the lock: if the cache is empty or its entry has expired (TTL for success, backoff for failure), fetch.
  - Returns `None` if the cached state is a failure, otherwise the alerts for `park_codes`, sorted by category.
  - `get_alerts([])` warms the cache and returns `[]`.
- `format_alerts(alerts) -> str` renders each alert:
  ```
  [Yosemite] Park Closure: Tioga Road closed for season (updated 2026-09-20)
  <description>
  More: <url>
  ```

### `src/generation/memory.py`

- `RouteDecision` gets `park_codes: list[str]`.
- Router prompt: list each park with its code; fill `park_codes` from the message and history; an alerts or closures question with no resolvable park goes to `ambiguous` with a "Which park are you visiting?" style question.

### `src/generation/generate_response.py`

- `stream_response` takes an `alerts` argument.
- The prompt gets a "Current alerts (as of <today>)" section with `format_alerts(...)`, or, when `alerts is None`:
  "Live alerts are unavailable; tell the user to check nps.gov/<code>/planyourvisit/conditions.htm before going."
- When `park_codes` is empty, the section is left out.
- System prompt rule: mention an alert only when it affects the plan or question, and cite its updated date.

### `app.py`

- Call `get_alerts([])` once at startup, before `demo.launch()`.
- In `stream_answer`, after routing to `in_scope`: `alerts = get_alerts(decision.park_codes)` if `park_codes` is non-empty, and pass it to `stream_response`.

## Tests

### `tests/test_alerts.py` (new, `requests.get` mocked)

- A second call within the TTL uses the cache (one HTTP call).
- An expired cache fetches again.
- A failure returns `None`, and calls during the backoff make no HTTP call.
- After the backoff, the next call fetches again.
- Results are filtered to `park_codes` and sorted by category.
- `format_alerts` output includes the park name, category, title, updated date, and url.

### `tests/test_router.py`

- `park_codes` is resolved from the message ("Is Tioga Road open?" → `yose`) and from history on follow-ups.
- "Any closures this week?" with no park in history → `ambiguous`.

## Risks

- A single transient failure (including at startup) means a full hour without alerts. Shorten `FAILURE_BACKOFF_SECONDS` if that shows up in practice.
- The cache lives in process memory, so a restart clears it.
- `lastIndexedDate` is an index date, not a closure date, and real closure dates exist only in description text.
