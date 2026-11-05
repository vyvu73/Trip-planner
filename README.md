# Trip Planner (Lupine)

Python/Gradio chatbot for California national parks, backed by PostgreSQL with pgvector.

## Local development

Use Python 3.13 and uv:

```bash
uv sync --locked
uv run python app.py
```

Set credentials in your local `.env` (never commit it). Without `DATABASE_URL`,
retrieval uses the existing local PostgreSQL database `trip_planner` on localhost,
port 5432, as user `postgres`.

## Deploy to Heroku

The root Procfile starts Gradio on `0.0.0.0` using Heroku's assigned `$PORT`.
Heroku installs production dependencies from `uv.lock`; `.python-version` selects Python 3.13.

1. Install the Heroku CLI in WSL, then run `heroku login` and
   `heroku create YOUR_UNIQUE_APP_NAME` from this repository.
2. Attach a Heroku Postgres plan that supports pgvector through the dashboard.
   Check current plan capabilities and pricing before provisioning. The primary
   database attachment sets `DATABASE_URL` automatically.
3. Migrate your existing schema and data, including `documents_chunks` and its
   embeddings, into the new database. Enable `CREATE EXTENSION IF NOT EXISTS vector;`
   if your migration does not already do so. Do not overwrite a populated database.
   Creating an empty database alone does not provide the retrieval corpus.
4. In Settings > Config Vars, configure the values below. Match `EMBED_MODEL` to
   the model and vector dimensions used by your existing embeddings.
5. Test with `heroku local web` using a local `.env` containing the hosted
   `DATABASE_URL`, then commit the deployment changes and run:

```bash
git push heroku HEAD:main
heroku ps:scale web=1
heroku open
heroku logs --tail
```

Select a suitable paid web dyno plan in Resources before scaling. App compute,
database hosting, and OpenAI API usage are separate costs. Test an in-scope park
question, streamed replies, alerts, and transportation tools after deployment.

| Config var | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for generation and embeddings |
| `EMBED_MODEL` | Required; match your stored embeddings |
| `DATABASE_URL` | Production PostgreSQL URL; supplied by attached Heroku Postgres |
| `NPS_API_KEY` | Live park alerts |
| `NPS_BASE_URL` | NPS API base URL, e.g. `https://developer.nps.gov/api/v1` |
| `ORS_API_KEY` | Route calculations |
| `EIA_API_KEY` | Gas prices |
| `BRAINTRUST_API_KEY` | Optional tracing to the project's configured Braintrust project |
| `OPENAI_MODEL` | Optional router model override |

Leave `KITARU_RECORD` unset unless you also configure a reachable production
Kitaru backend. Ingestion scripts still target the local database; this deployment
configures the web retrieval path and embedding loader to accept `DATABASE_URL`.
The data directory is excluded from the deployment bundle because retrieval reads
its corpus from PostgreSQL.

References: [Python builds](https://devcenter.heroku.com/articles/python-behavior-in-heroku),
[pgvector](https://devcenter.heroku.com/articles/pgvector-heroku-postgres),
[Git deployment](https://devcenter.heroku.com/articles/git).
