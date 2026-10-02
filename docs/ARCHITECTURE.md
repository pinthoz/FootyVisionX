# Architecture

FootyVision is a vertical slice of a scouting product: raw event data in, explained
recommendations out. The organising principle is a strict split of responsibilities —
**deterministic code computes every number, the LLM only explains numbers it was handed.**

```
StatsBomb Open Data ──► ETL (extract ▸ aggregate ▸ load) ──► PostgreSQL
                                                                 │
        ┌──────────────┬──────────────────┬─────────────────────┼──────────────────┐
        ▼              ▼                  ▼                      ▼                  ▼
  Similarity     Performance Score   XGBoost role clf      Value predictor    RAG vector
  (z + cosine)   (weighted pctl)     + SHAP                (LightGBM + SHAP)   store (embeds)
        └──────────────┴────────┬─────────┴──────────────────────┴──────────────────┘
                                ▼
                 Local LLM  (LM Studio / Ollama — chat + embeddings)
              reports ▸ NL → safe query ▸ conversational RAG assistant
                                ▼
                    FastAPI  ──►  web dashboard (radar / search / assistant)
```

## Layers

### 1. ETL — `src/footyvision/etl/`

| Module | Responsibility |
|---|---|
| `statsbomb.py` | Pulls competitions, matches and events through `statsbombpy` (cached to `data/`). |
| `aggregate.py` | Events → per-player-per-match rows → per-season totals and **per-90 rates**. |
| `load.py` | Upserts the aggregates into Postgres. |
| `transfermarkt.py` | Joins real market values (Kaggle dump) by fuzzy name match. |
| `sofifa.py` | Optional extra attributes via `soccerdata`. |

Players below `MIN_MINUTES` (default 600) are dropped from season aggregates — small
samples produce meaningless per-90 rates.

### 2. Storage — `src/footyvision/db/`

SQLAlchemy 2 models over seven tables: `competitions`, `seasons`, `teams`, `players`,
`matches`, `player_match_stats`, `player_season_stats`. Schema changes go through Alembic
(`migrations/`); `footyvision init-db` creates the schema from scratch.

### 3. ML — `src/footyvision/ml/`

| Module | Method | Output |
|---|---|---|
| `features.py` | Per-90 feature matrix, z-scored **within position group** so a full-back is never compared to a striker. | Shared feature frame. |
| `similarity.py` | Cosine similarity over the z-scored vectors. | "Stylistic replacements for X". |
| `scoring.py` | Position-weighted percentile composite, 0–100 — transparent by design, no black box. | Performance Score, rankings. |
| `talent.py` | XGBoost classifies position from playing style (~85% accuracy); SHAP explains each prediction. | Role fit + role-mismatch flags. |
| `value.py` | LightGBM regression on age + per-90 stats against real Transfermarkt values, explained with SHAP. | Value estimate vs. asking price. |

### 4. LLM — `src/footyvision/llm/`

`client.py` talks to any OpenAI-compatible endpoint (LM Studio, Ollama) for both chat and
embeddings; nothing leaves the machine and there are no paid API calls. `scouting.py`
builds a report prompt from a computed stat block — the model receives the numbers and
writes prose about them, so it cannot invent a figure.

### 5. Search — `src/footyvision/search/`

Natural-language search never generates SQL. The LLM's only job is to fill in a Pydantic
`PlayerQuery` (`query.py`); if validation fails, the request is rejected. The validated
object is then translated into a parameterised SQLAlchemy query by trusted code — an
injection-resistant boundary that also makes the feature testable without an LLM.

### 6. RAG — `src/footyvision/rag/`

`profiles.py` renders one natural-language profile per player from the database;
`store.py` embeds them into a numpy matrix persisted as `.npz` (for a few hundred players
a vector DB would be overkill — pgvector or FAISS is the swap-in at larger scale);
`assistant.py` retrieves the top-k profiles and answers strictly from that context,
citing the players it used.

Retrieval is **hybrid**. Players the question names are pinned lexically (accent-folded
token match), and the remaining slots are filled by embedding similarity — but the query
vector is then the *centroid of the pinned players' profiles*, not the question text.
Embedding the raw question lets proper nouns dominate the vector, which retrieves players
with similar-looking **names** instead of a similar playing **style**. Each profile also
carries the per-90 values and percentiles behind its phrases, so comparative questions
("who has the most xG?") are answerable from the context rather than guessed.

Documents and questions are embedded with the **task prefixes the model was trained on**
(`title: none | text: ` and `task: search result | query: ` for EmbeddingGemma). Retrieval
models are trained asymmetrically, and embedding both sides identically measurably hurts —
[`scripts/eval_embeddings.py`](../scripts/eval_embeddings.py) puts a number on it.

### 7. API & frontend

`api/main.py` mounts routers per capability (`health`, `players`, `similarity`, `talent`,
`reports`, `search`, `assistant`) with Pydantic response models in `schemas.py`; OpenAPI
docs at `/docs`. The dashboard in `frontend/web` (Next.js, App Router) consumes that API;
`frontend/radar_demo.html` is a dependency-free Plotly page for a quick look at radars.

## Design decisions worth knowing

- **Position-relative normalisation everywhere.** Raw per-90s make centre-backs look
  useless; every comparison is against positional peers.
- **The LLM is a narrator, not a calculator.** Every number in a report, ranking or answer
  was computed before the prompt was built.
- **Structured output over free text at trust boundaries.** NL search returns a validated
  schema, never a query string.
- **Honest evaluation.** The value model's held-out R² is ≈0.05 and the README says so;
  one season of public per-90 stats plus age genuinely does not predict market value.

## Running the pieces

| Command | Does |
|---|---|
| `footyvision init-db` | Create the schema. |
| `footyvision competitions` | List loadable StatsBomb competitions. |
| `footyvision load -c 11 -s 27` | Load La Liga 2015/16 end to end. |
| `footyvision aggregate` | Rebuild season aggregates. |
| `footyvision talent-report` | Train/evaluate the role classifier + SHAP. |
| `footyvision value-report` | Train/evaluate the market-value model. |
| `footyvision index` | Embed player profiles into the RAG store. |
| `uvicorn footyvision.api.main:app --reload` | Serve the API on :8000. |
