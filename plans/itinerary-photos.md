# Itinerary photos

Status: planned, not implemented.

## Goal

When an answer is an itinerary, show a row of photos of the trails and
viewpoints it mentions (Mist Trail, Glacier Point, Fern Canyon...), falling
back to a photo of the park. Photos are picked by code from a hand-curated
list; the model never writes image URLs.

## Design decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Source | Hand-picked photos found online, one list in code. Prefer sources whose license allows reuse with credit: NPS (nps.gov, flickr.com/photos/nps), Wikimedia Commons, Unsplash. Record credit and source URL for every photo. |
| 2 | Storage | Download each photo into `static/photos/<park_code>/<slug>.jpg` (resized to ~800px wide) and serve it through Gradio. Not hotlinked: other sites move, block, or slow down images. |
| 3 | Photo list | `PHOTOS` in `src/ui/photos.py`, like `TRIPADVISOR_LOCATIONS`: `{park_code, name, aliases, file, alt, credit, source_url}`. One entry per trail/viewpoint, plus one `kind: "park"` entry per park as the fallback. Start from the 37 Tripadvisor locations. |
| 4 | Missing file | Entries whose file doesn't exist are skipped with a warning at startup. |
| 5 | When photos show | Only for itineraries: the final answer contains `**Day ` (required by the itinerary format in the system prompt). |
| 6 | Which photos | Entries for `decision.park_codes` whose `name` or an alias appears in the answer (case-insensitive, whole words), in order of first mention, max 6. If none match, the park photo(s). |
| 7 | Timing | After streaming finishes, yield once more with the photo section appended. Streaming itself is unchanged. |
| 8 | Rendering | Markdown in the same assistant message, so it persists in `BrowserState`: `**Photos**`, then the images on one line (`![alt](url "name")`), then a small credits line. CSS turns them into a row of equal-height thumbnails. |
| 9 | History | `format_turns` strips everything from `\n\n**Photos**` on, so image URLs never reach the router or the answer model. |

## Flow

```
stream_answer
  ├─ stream_response(...) ──► yield answer as it grows (unchanged)
  └─ after the last token:
       is_itinerary(answer)?  no ──► done
                              yes ──► pick_photos(answer, park_codes)
                                        └─► yield answer + photos_markdown(photos)
```

## Changes

### `src/ui/photos.py` (new)

- `PHOTOS` list (decision 3).
- `PHOTO_DIR = "static/photos"`, `MAX_PHOTOS = 6`, `PHOTOS_HEADER = "\n\n**Photos**\n"`.
- `available_photos()`: `PHOTOS` filtered to entries whose file exists; warns once for the rest.
- `is_itinerary(answer) -> bool`: `"**Day " in answer`.
- `pick_photos(answer, park_codes) -> list[dict]`: decision 6.
- `photos_markdown(photos) -> str`: `PHOTOS_HEADER` + image line + credits line, `""` if no photos.
- `strip_photos(text) -> str`: text before `PHOTOS_HEADER`.

### `app.py`

- `gr.set_static_paths([PHOTO_DIR])` before building the Blocks.
- In `stream_answer`, after the streaming loop: if `is_itinerary(answer)`, yield `answer + photos_markdown(pick_photos(answer, decision.park_codes))`.
- Photo errors are caught and logged; the answer is still shown without photos.

### `src/generation/memory.py`

- `format_turns` applies `strip_photos` to assistant turns.

### `src/ui/app.css`

- `#chat .prose img`: fixed height (~140px), `object-fit: cover`, rounded corners, small gap, wraps on narrow screens.
- Credits line: small, muted text.

### `static/photos/` (new)

- One folder per park code. Filenames match `file` in `PHOTOS`.

## Tests

### `tests/test_photos.py` (new)

- `is_itinerary`: true for a `**Day 1 — ...**` answer, false for a fact answer.
- `pick_photos`: matches names and aliases case-insensitively, keeps order of first mention, ignores other parks' photos, caps at `MAX_PHOTOS`, falls back to the park photo.
- Missing file is skipped.
- `photos_markdown` returns `""` for no photos, and includes the credits.
- `strip_photos` removes the section; `format_turns` output has no image URLs.

## Risks

- **Licensing.** Photos from random sites are usually copyrighted. Stick to the sources in decision 1 and keep the credit line; CC BY photos require it.
- **Static file URL.** Check the URL form Gradio 6.28 uses for `set_static_paths` files (`/gradio_api/file=static/photos/...`) and that the Chatbot's Markdown renders it.
- **Name matching.** Short or generic names ("Hidden Valley", "Lake") can false-match; use specific names and aliases, and only match within `park_codes`.
- **Itinerary check.** Relies on the model using the `**Day N — ...**` format; if it drifts, no photos show (safe failure).
