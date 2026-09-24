# Tripadvisor reviews ingestion

Status: implemented (2026-09-23). Data, threshold, and manual checks pass
(manual #1 partial, see below).

## Goal

Add visitor reviews from Tripadvisor to `documents_chunks` so answers can
include first-hand experience (crowds, views, felt difficulty, tips) next to
the official NPS and web facts, without reviews being mistaken for current
conditions.

## Design decisions

| # | Decision | Choice |
|---|---|---|
| 1 | API | Tripadvisor free plan. Documented as at most 5 reviews per location; in practice it returns **at most 3**. Coverage comes from many locations per park, not depth per location. |
| 2 | Storage terms | Store review text in our DB anyway (personal project, not deployed publicly). All rows use `source_type = 'tripadvisor'` so they can be removed with one `DELETE`. |
| 3 | Location list | Hand-written `TRIPADVISOR_LOCATIONS`, trails, viewpoints, and a few whole-park listings mixed. Covers all 8 parks. |
| 4 | Location IDs | Resolved once by hand and written into the list as `{park_name, location_name, location_id}`. Fetch runs never call search. `location_id: None` → skip with a warning. |
| 5 | Chunk unit | One chunk per review. No splitting, even for the few reviews over `MAX_TOKENS = 500` (max 789): one person's opinion loses meaning when halved, and the embedding model accepts ~8k tokens. |
| 6 | Pipeline | Same as NPS and web: `raw_tripadvisor_review` → `chunk_tripadvisor.py` → `documents_chunks` → `load_chunk.py`. |
| 7 | Source labels | `format_context` labels every chunk by source. The system prompt says official sources win on facts; reviews are opinions from their travel date, shown as a short verbatim quote with a plain-text source (`— Tripadvisor review · <location>, traveled <date>`), no links. |
| 8 | Retrieval | `k = 7`: top 5 non-review chunks + up to 2 review chunks. |
| 9 | Review relevance | A review chunk is only included if its cosine distance is within `REVIEW_MAX_MARGIN = 0.15` of the **best non-review chunk's** distance. Unused review slots are backfilled with non-review chunks 6–7, so 7 chunks are always returned when available. (Replaces the planned fixed `REVIEW_MAX_DISTANCE = 0.5`; see Threshold tuning.) |
| 10 | Refresh | Re-runs insert with `ON CONFLICT (review_id) DO NOTHING`. With `MOST_RECENT` and ≤ 3 per location, the stored set grows slowly over time instead of being replaced. |

### Actual size

37 locations (35 with ids) → 90 reviews → 85 review chunks (5 under
`MIN_CHUNK_TOKENS`), against 3,469 NPS and 168 web chunks. Reserved slots
(decision 8) exist because reviews would otherwise rarely rank, or crowd out
facts when they do.

## Implementation

### 1. `src/ingestion/tripadvisor_locations.py`: location list

- `TRIPADVISOR_LOCATIONS`, each entry `{park_name, location_name, location_id}`.
- `park_name` is a spelling `normalize_park` accepts
  (`src/ingestion/park_names.py`).
- `location_name` uses Tripadvisor's own name where it differs from ours
  ("Badwater", "Dante's View", "Moro Rock Trail").
- Channel Islands: Tripadvisor lists the islands (Santa Cruz, Anacapa,
  Santa Rosa) and the Ventura visitor center, not spots on them like
  Scorpion Anchorage or Inspiration Point.
- Unresolved (`None`): Hidden Valley, Crystal Cave.

| Park | Locations with ids |
|---|---|
| Death Valley, Sequoia & Kings Canyon, Yosemite | 5 |
| Channel Islands, Joshua Tree, Lassen Volcanic, Pinnacles, Redwood | 4 |

### 2. `src/ingestion/resolve_tripadvisor_ids.py`: one-off resolver

- For each entry with `location_id` missing, calls `search_location` with the
  **plain location name** and prints the top 3: id, name, geo, address,
  review count. `*` marks candidates whose geo/address mentions the park.
- Search matches the start of the name, so extra words (e.g. adding the park
  name) return nothing; the park is judged from geo instead.
- Sleeps 1 s between searches (search rate limit).
- Prints only; the chosen id is pasted into the list by hand.

### 3. `raw_tripadvisor_review` table

Created by `ensure_table()` in `ingest_tripadvisor.py`
(`CREATE TABLE IF NOT EXISTS`, runs at the start of every fetch).

| Column | Type | Notes |
|---|---|---|
| `id` | serial PK | |
| `review_id` | text UNIQUE NOT NULL | Tripadvisor review id (int in the API); upsert key |
| `location_id` | text NOT NULL | |
| `location_name` | text | From our list, not the API |
| `park_name` | text | Normalized via `normalize_park` |
| `rating` | int | |
| `title` | text | |
| `content` | text | Review text |
| `trip_type` | text | e.g. `COUPLES`, `FAMILY` |
| `travel_date` | text | As returned (e.g. `2026-08`) |
| `published_date` | text | From the API's `publish_ts` |
| `url` | text | |
| `created_utc` | timestamptz | Ingest time, `DEFAULT now()` |

The older, empty `raw_data_tripadvisor` table is unused and can be dropped.

### 4. `src/ingestion/ingest_tripadvisor.py`: fetch

- Reads `TRIPADVISOR_LOCATIONS`; skips `location_id is None` with a warning.
- `fetch_reviews(location_id)`, `language=en`, `sort_by=MOST_RECENT`.
- `INSERT ... ON CONFLICT (review_id) DO NOTHING RETURNING id`; the returned
  rows count as "new".
- Commits per location; an HTTP error on one location is logged and skipped.
- Sleeps 1 s between locations.
- Prints per-location `fetched / new` and a total.
- Exploratory search loop and print helpers removed; `search_location` and
  `get_primary_name` kept for the resolver.

### 5. `src/ingestion/chunk_tripadvisor.py`: chunk

- One row in `documents_chunks` per review:
  - `source_type = 'tripadvisor'`, `source_table = 'raw_tripadvisor_review'`,
    `source_id = review_id`, `chunk_index = 0`
  - `park_name` = normalized display name, `title` = `location_name`,
    `url` = review URL
  - `content` = review title + blank line + review text
  - `metadata`: `park_code`, `page_type = 'review'`, `location_id`,
    `location_name`, `rating`, `trip_type`, `travel_date`
  - `token_count` via the same `tiktoken` encoder as the other chunkers
- Skips reviews under `MIN_CHUNK_TOKENS` (20).
- Upserts on `(source_table, source_id, chunk_index)`. `embedding` is reset to
  NULL only when `content` changed, so re-runs don't force re-embedding.

### 6. `src/vector_store/load_chunk.py`: embed header

No change. The header already carries `title` (location name), `park`, and
`page_type: review` (from metadata). Rating was deliberately left out.

### 7. `src/retrieval/retrieve_chunks.py`: split retrieval

- Selects `source_type` and `metadata` in addition to the previous columns.
- `run_vector_query(conn, vector, park_name, reviews, limit)` runs one
  nearest-neighbour query, reviews only or everything but reviews.
- `search_chunks(conn, query_vector, question, k=7, review_k=2)`:
  1. Non-review query: `source_type <> 'tripadvisor'`, same park filter,
     `LIMIT k`.
  2. Review query: `source_type = 'tripadvisor'`, same park filter,
     `LIMIT review_k`, then keep only reviews with
     `distance − best_fact_distance < REVIEW_MAX_MARGIN`.
  3. Result = first `k - len(reviews)` non-review chunks + reviews.
- `REVIEW_MAX_MARGIN = 0.15` as a module constant.

### 8. `app.py`

`retrieve_chunks(decision.standalone_question, k=10)` → `k=7`.

### 9. `src/generation/generate_response.py`: labels and rules

- `format_label` / `format_context` label by `source_type`:
  - `nps` → `[NPS official · <park> — <title>]`
  - `web_page` → `[Web article · <park> — <title>]`
  - `tripadvisor` → `[Visitor review · Tripadvisor · <title> · <rating>★ · traveled <travel_date> · <trip_type>]`
    (missing fields omitted)
- Added to `SYSTEM_PROMPT`:
  ```
  ## Source types
  - "NPS official" and "Web article" chunks are facts. Use them for hours,
    fees, permits, closures, road and trail status, and distances.
  - "Visitor review" chunks are one person's opinion from their travel date.
    Use them for experience: crowds, views, difficulty as felt, tips.
    Never present them as current conditions.
  - When you use a visitor review, quote the relevant part directly, then cite it:
      > "exact words from the review"
      > — Tripadvisor review · <location>, traveled <travel_date>
    Do not include links or URLs in the citation.
  - Copy the words exactly as they appear in the review. Keep quotes short
    (one or two sentences) and only quote the part that supports your point.
  - Never put paraphrased words in quotation marks.
  - Only quote reviews that relate to the answer; never use a review quote to
    support hours, fees, permits, closures, or current conditions.
  - If a review conflicts with an official source, follow the official
    source; you may mention the difference.
  ```
- The review URL is stored (`raw_tripadvisor_review.url`, `documents_chunks.url`)
  but deliberately not shown to the model or the user.

## Checks

### Data

- [x] Every entry in `TRIPADVISOR_LOCATIONS` has a `location_id` or an
  intentional `None`; all 8 parks present.
- [x] `SELECT park_name, count(*) FROM raw_tripadvisor_review GROUP BY 1`:
  every park has reviews (7–15 each).
- [x] Re-running `ingest_tripadvisor.py` inserts 0 duplicates
  (`fetched 90 / new 0`).
- [x] `SELECT count(*) FROM documents_chunks WHERE source_type = 'tripadvisor' AND embedding IS NULL`
  is 0 after `load_chunk.py`.

### Threshold tuning

Distances of the top review vs the best non-review chunk:

| # | Prompt | Reviews expected? | Best fact | Best review | Margin | Fixed 0.5 | Margin 0.15 |
|---|---|---|---|---|---|---|---|
| 1 | Is Bumpass Hell worth it? | Yes | 0.449 | 0.414 | −0.035 | ✅ | ✅ |
| 2 | How crowded is Glacier Point at sunset? | Yes | 0.437 | 0.504 | +0.067 | ❌ missed | ✅ |
| 3 | Tips for visiting Fern Canyon with kids | Yes | 0.367 | 0.470 | +0.103 | ✅ | ✅ |
| 4 | What's the entrance fee at Joshua Tree? | No | 0.244 | 0.485 | +0.241 | ❌ 1 leaked | ✅ |
| 5 | Do I need a reservation for Yosemite in summer? | No | 0.352 | 0.596 | +0.244 | ✅ | ✅ |

No fixed distance separates these (a "yes" at 0.504 sits above a "no" at
0.485). The margin does: factual questions have a near-exact fact match, so
reviews trail far behind. 0.15 sits between the largest "yes" margin (0.103)
and the smallest "no" margin (0.241), leaning strict. Held-out check:
"Is Badwater Basin crowded?" → 2 reviews ✅. Five prompts is a small sample;
add more before relying on the exact value.

### Manual

Run `uv run python app.py`:

| # | Prompt | Expected | Result |
|---|---|---|---|
| 1 | Is Bumpass Hell worth it in June? | NPS status first, then a quoted review | Partial: grounded in NPS ("rarely open in June… late July through October"), reviews retrieved but not quoted (both from Jul/Aug travel). Run before the quoting rule and under the fixed 0.5 threshold. |
| 2 | What's the entrance fee at Joshua Tree? | Facts only, no review content | ✅ Facts only. (Under 0.5 one review was in context but ignored; under the margin rule none is retrieved.) |
| 3 | Is Badwater Basin crowded? | Review content framed as visitor opinion | ✅ 2 runs: NPS facts (1–2 hr visit, no hiking after 10 AM in summer) plus a verbatim quote with plain-text source. One run quoted a weakly relevant line ("outstanding for flowers"). |
| 4 | Is parking easy at Glacier Point? (review: "it wasn't hard to find a parking spot") | Follows NPS, may mention the difference | ✅ 2 runs: leads with NPS (195 spaces, free, seasonal road, closures), review quoted verbatim as one visitor's experience. Weak contradiction test: NPS doesn't say parking is hard. |

All review quotes across these runs were verbatim and had no URL. Layout
varies (block quote vs inline in a bullet); both include quote and source.

## Completion criteria

- [x] All 8 parks have resolved locations and ingested reviews.
- [x] Review chunks are embedded.
- [x] `search_chunks` returns ≤ 2 reviews, only within the margin, and
      7 chunks total when available.
- [x] Factual prompts return no review chunks; experience prompts do.
- [x] Answers present reviews as opinions (verbatim quote + source) and
      never override NPS facts.

## Accepted trade-offs

- Storing review text goes against the free Content API terms; accepted
  for a personal project. Removal is one `DELETE` on `source_type` plus
  dropping `raw_tripadvisor_review`.
- ≤ 3 reviews per location means a small, possibly unrepresentative sample.
- `MOST_RECENT` sorting skews toward the current season at ingest time.
- The margin rule is tuned on 5 prompts: some experience questions may still
  miss reviews and some factual ones may include them. Router-based intent
  (`wants_experience`) is the next step if this proves a problem.
- Hand-written locations will miss places; coverage grows only by editing
  the list.
- Quotes are verbatim only because the prompt asks for it; nothing in code
  checks them. If invented quotes show up, add a post-generation check that
  each quoted span appears in a retrieved review chunk.
