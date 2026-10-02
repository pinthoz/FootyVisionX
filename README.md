<div align="center">

<img src="docs/images/logo.svg" alt="FootyVision" width="110" height="110">

# FootyVision

**An AI football scouting platform — the models do the maths, the LLM explains them.**

[![CI](https://github.com/pinthoz/FootyVisionX/actions/workflows/ci.yml/badge.svg)](https://github.com/pinthoz/FootyVisionX/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

[Quick start](#quick-start) · [What it does](#what-it-does) · [Architecture](docs/ARCHITECTURE.md) · [Roadmap](docs/ROADMAP.md) · [Contributing](CONTRIBUTING.md)

</div>

---

FootyVision helps sporting directors, scouts and analysts answer questions like *"who are the
best stylistic replacements for this player?"*, *"which players fit this role?"*, *"is this
signing good value?"* and *"find me a ball-winning midfielder"*.

**Guiding principle:** traditional ML models do the maths; the **LLM explains and interprets**
(scouting reports, comparisons, natural-language search, a RAG assistant) — it never invents
numbers.

> Built end-to-end: StatsBomb event ETL → Postgres → similarity / scoring / value ML →
> local LLM + embeddings → FastAPI + a web dashboard. Runs fully **locally**, with no paid APIs.

## What it does

| Capability | How it works | Try it |
|---|---|---|
| **Similarity engine** | Per-90 features, z-scored **within position group**, cosine similarity | `GET /players/{id}/similar` |
| **Scouting radars** | Percentile-vs-peers on each metric | `GET /players/{id}/radar` |
| **Performance Score** | Transparent position-weighted percentile composite (0–100) | `GET /players/{id}/score`, `GET /rankings` |
| **Position classifiers** | XGBoost on per-90 style plus *where* the player acts (pitch-map zones from every event location): 4 groups (94% accuracy, 94% balanced), 10 roles (82% / 66%), and the exact position as a three-name shortlist (right 88% of the time; top pick 71%, up from 47% before the zones carried a side). Per-class recall is published, because accuracy hides that *Wing Back* is still rarely told apart from a full-back (recall 0.14) | `GET /talent/model-info` |
| **LLM scouting reports** | Report grounded in computed stats; the model can't invent numbers | `POST /players/{id}/report` |
| **Natural-language search** | LLM → validated Pydantic `PlayerQuery` (never raw SQL) → safe query | `POST /search` |
| **Market value range** | LightGBM on Transfermarkt 2015/16 values, served as a 5th–95th band with the share of held-out players it actually caught (72%) and the error of predicting the median for everyone | `GET /players/{id}/value`, `GET /value/bargains` |
| **Team strength** | Poisson attack/defence ratings per side, fitted per competition | `GET /teams/strength` |
| **Pitch maps** | Where each player acts and shoots, from every event location of 2,322 matches; adds how high and on which side a player plays, which took the exact-position model from 47% to 71% | `GET /players/{id}/pitch` |
| **Expected Threat (xT)** | Karun Singh's possession-value model: each completed pass and carry valued by how much it raised the chance of a goal. Tracks a team's goals in a match better than its pass count (r 0.37 vs 0.27), fitted and tested on disjoint matches ([`eval_expected_threat.py`](scripts/eval_expected_threat.py)) | `GET /players/{id}/pitch` |
| **Expected table** | Points each team's chances deserved, from every shot's xG. First-half xPts predict second-half points better than the real points do (r 0.72 vs 0.67, 143 team-seasons — [`eval_expected_table.py`](scripts/eval_expected_table.py)) | `GET /teams/expected-table` |
| **RAG assistant** | Player profiles embedded locally → retrieve + grounded answer, cites names. Hard filters (foot, age, nationality, role) apply before ranking, and "who makes the most X" returns the actual top of the filtered pool by X — the true leader in 42 of 42 bank questions, against 12 by similarity alone ([`eval_metric_leaders.py`](scripts/eval_metric_leaders.py)) | `POST /assistant` |

**Example** — *"La Liga forwards with xG per 90 over 0.5"* → Ronaldo, Benzema, Suárez, Messi.
*"médio defensivo que ganhe bolas e intercete"* → retrieves Busquets/Camacho and recommends
Camacho, citing his real intercepting/tackling profile.

## Architecture

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

Layer-by-layer detail and the reasoning behind each design decision:
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

## Tech stack

- **Backend:** FastAPI · SQLAlchemy 2 · Alembic · PostgreSQL
- **Data/ML:** pandas · scikit-learn · **XGBoost** · **LightGBM** · **SHAP** · rapidfuzz
- **LLM (local):** OpenAI-compatible endpoint — chat (Gemma/Qwen/Llama) + embeddings (`EmbeddingGemma-300M`)
- **Data sources:** StatsBomb Open Data (`statsbombpy`) · Transfermarkt values (Kaggle) · FBref/SoFIFA (`soccerdata`)
- **Frontend:** Next.js dashboard (`frontend/web`) · self-contained Plotly radar demo (`frontend/radar_demo.html`)
- **Infra:** Docker Compose · GitHub Actions

## Quick start

Prerequisites: Docker (for Postgres), Python 3.11+, and a local LLM server
(LM Studio or Ollama) for the LLM/RAG features.

```bash
# 1. Configuration
cp .env.example .env                  # the defaults work out of the box

# 2. Postgres
docker compose up -d db

# 3. Python env + package
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows;  macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

# 4. Schema + data (StatsBomb Open Data)
footyvision init-db
footyvision competitions              # list available competitions
footyvision load -c 11 -s 27          # La Liga 2015/16 (~380 matches); repeat per season

# 4b. Model outputs the API serves (needs the `train` extra, included in `dev`)
footyvision spatial                   # pitch maps, xT, expected table (~10 min: raw events)
footyvision precompute                # position profiles, shortlists, team ratings

# 5. API
uvicorn footyvision.api.main:app --reload    # -> http://localhost:8000/docs
```

**LLM features** (reports, NL search, assistant): start LM Studio, load a chat model + the
`embeddinggemma-300m` embedding model, Start Server (port 1234), set `LLM_MODEL` in `.env`, then:

```bash
footyvision index                     # embed player profiles into the RAG vector store
```

**Value model** (optional): needs Kaggle access to `davidcariboo/player-scores`; drop
`players.csv` + `player_valuations.csv` into `data/`, then `footyvision value-report`, which
writes `models/value/value_model.joblib` for the API to serve.

**Everything at once:**

- Docker: `docker compose up --build` (the API reaches the host LLM via `host.docker.internal`).
- Windows: `.\scripts\start.ps1` boots Postgres, LM Studio, the API and the dashboard.
- Dashboard only: `cd frontend/web && npm install && npm run dev`.

## API reference

**Players** `/players` · `/players/{id}` · `/players/{id}/seasons` · `/players/{id}/similar` ·
`/players/{id}/radar` · `/players/{id}/score` · `/players/{id}/value` · `/rankings` ·
`/metrics/{metric}/distribution`
**Models** `/talent/model-info` · `/value/model-info` · `/value/bargains` · `/teams/strength`
**Language** `POST /search` · `POST /search/structured` · `POST /assistant` ·
`POST /players/{id}/report` · `/players/{id}/report/context`
**Provenance** `/coverage` · `/eval/assistant` · `/health` · `/llm/health`

The three endpoints that call an LLM are rate-limited per client and reject questions over
500 characters or assistants asked for more than 12 players.
Interactive OpenAPI docs at `/docs`.

## Repository layout

```
src/footyvision/       # the Python package
  config.py            #   env-driven settings
  db/                  #   base.py, models.py            — SQLAlchemy schema
  etl/                 #   statsbomb, aggregate, load, transfermarkt, sofifa
  ml/                  #   features, similarity, scoring, talent, value
  llm/                 #   client.py (chat + embeddings), scouting.py
  search/              #   query.py (safe PlayerQuery), nl.py (NL → query)
  rag/                 #   profiles, store, assistant, service
  api/                 #   main.py, routers/, schemas.py
  cli.py               #   init-db · load · aggregate · enrich · spatial · precompute · value-report · index
frontend/              # radar_demo.html · web/ (Next.js dashboard)
migrations/            # Alembic
scripts/               # start.ps1 · eval_embeddings.py (retrieval benchmark)
tests/                 # 190 tests (unit + API), DB/network/LLM-free
docs/                  # ARCHITECTURE.md, ROADMAP.md
```

## Data reality (and why it matters)

This project deliberately reports what public/free data **can't** do, not just what it can:

- **StatsBomb Open Data** has no Portuguese league. The pool is 2,562 player-seasons from
  2,322 matches across 10 competitions: the five big men's leagues in 2015/16 and five
  women's leagues in 2023/24. Date of birth and preferred foot exist only for the men, and
  every age or foot filter says how many players it could not check.
- **FBref** only exposes advanced stats (xG, progression) for the Big-5 leagues, so a rich
  Primeira Liga engine isn't feasible from free sources.
- The **embedding model was chosen by measurement, not by leaderboard**
  ([`scripts/eval_embeddings.py`](scripts/eval_embeddings.py) scores retrieval on these 411
  profiles). The original setup — nomic-embed-text with no task prefixes — retrieved the
  right position for only **30% of Portuguese queries** against 90% of English ones.
  EmbeddingGemma-300M with its proper prefixes reaches **73% / 97%** (measured when the
  pool was La Liga alone). A cross-lingual gap remains, and the script reports it.
- The **value model** explains R² 0.43 of log(value) but only **0.22 in euros**, and its
  MAE (€4.29M) beats predicting the median for everyone (€5.11M) by about €0.8M. It is
  served as a range for that reason. A quarter of its labels used to be somebody else's:
  fuzzy matching gave Casemiro "Henrique"'s value and 76 women the values of men with
  similar names, until matching moved from string similarity to shared name tokens.
- The **RAGAS evaluation** (241 questions, one answer model, one local judge) scores
  faithfulness 0.81 and context precision 0.36. The low precision is concentrated where a
  question asks for a shortlist and the answer then picks one of the six players retrieved
  — the metric penalises exactly the comparison pool a scout wants. It also cannot tell
  whether those six were the right six: when "who makes the most X" was changed to return
  the actual leaders (from 12 of 42 to 42 of 42), RAGAS saw no significant change in
  context precision, and a borderline gain in faithfulness only once the 163 questions
  whose retrieval did not change were used to measure run-to-run noise.

## Development

```bash
pytest                             # 190 tests, no Postgres/network/LLM required
ruff check src tests scripts
ruff format --check src tests
```

### Deployment memory

The API imports no machine-learning library. It serves the classifiers', team model's and
value model's outputs from `models/talent/predictions.json` and
`models/value/value_model.joblib`, which is what fits it in a 512MB instance: 152MB warm and
264MB at peak across 24 concurrent requests, against 505MB when it loaded the models itself.
Production installs without the `train` extra, so it cannot refit by accident.

**After importing a season, run `footyvision spatial`, then `footyvision precompute` (and
`value-report`), and commit the artifacts.** The classifiers read the pitch maps, so the
order matters. Until then the API serves the previous predictions and reports
`predictions_stale: true` in `/talent/model-info`.

CI runs the same checks on Python 3.11 and 3.12, plus a production build of the dashboard.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the ground rules (tests stay offline, the LLM
never invents numbers, no raw SQL from user input).

## License

[MIT](LICENSE) © Ana Pinto.

Football data from [StatsBomb Open Data](https://github.com/statsbomb/open-data), used under
their terms; market values from the public Transfermarkt dataset on Kaggle. This project is
not affiliated with StatsBomb or Transfermarkt.
