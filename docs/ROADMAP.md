# FootyVision — Roadmap

The goal is a **vertical slice that looks like a professional scouting product**, not a
broad-but-shallow demo. Phases 1–3 alone already showcase data engineering, ML, LLMs and
full-stack work.

Guiding principle: **ML predicts, the LLM explains.**

---

## Phase 0 — Foundations ✅
- Repo scaffold, `pyproject`, Docker Compose, ruff/pytest.
- Postgres schema (competitions, seasons, teams, players, matches, per-match & per-season stats).
- ETL: StatsBomb Open Data → per-player-per-match aggregation (minutes, progressive passes/carries,
  defensive actions) → per-season totals and **per-90 rates**.
- FastAPI skeleton with health + player endpoints.

**Definition of done:** `footyvision load` populates the DB and `/players/{id}/seasons`
returns per-90 stats.

## Phase 1 — Similarity Engine ✅
- Feature matrix from `player_season_stats` (per-90 features) — [ml/features.py](../src/footyvision/ml/features.py).
- **Standardised per position group** (z-score) — a striker is never compared to a centre-back.
- Cosine similarity → "most similar players to X" — [ml/similarity.py](../src/footyvision/ml/similarity.py).
- Minutes threshold enforced (`MIN_MINUTES`) to avoid small-sample noise.
- API: `GET /players/{id}/similar` and `GET /players/{id}/radar` (percentiles within group).
- Frontend: interactive **radar** comparing two players — [frontend/radar_demo.html](../frontend/radar_demo.html).
- *Still open:* UMAP 2D "player map" (visualisation only), Euclidean option, position-group weighting.

## Phase 2 — LLM Scouting Reports ✅
- Local OpenAI-compatible client behind a thin `LLMClient` — [llm/client.py](../src/footyvision/llm/client.py).
- Grounded context (radar percentiles + similar players + strengths/weaknesses) →
  prompt that **forbids inventing stats** → report — [llm/scouting.py](../src/footyvision/llm/scouting.py).
- API: `POST /players/{id}/report`, `GET /players/{id}/report/context` (grounding, no LLM),
  `GET /llm/health`. Report button wired into the radar UI.
- Needs a local LLM running (LM Studio / Ollama); returns HTTP 503 with guidance if it is not.
- *Still open:* stream tokens, cache reports, let the user pick report language/length.

## Phase 3 — Natural-Language Search (safe text-to-SQL) ✅
- LLM produces a **validated `PlayerQuery`** (Pydantic whitelist) — never raw SQL — so no
  injection is possible: [search/query.py](../src/footyvision/search/query.py),
  [search/nl.py](../src/footyvision/search/nl.py).
- API: `POST /search` (natural language, via LLM) and `POST /search/structured` (direct
  query, no LLM — powers testing and power users).
- Validated live: *"La Liga forwards with xG/90 > 0.5"* → Ronaldo/Benzema/Suárez/Messi;
  *"midfielders with progressive passes/90 > 7"* → Kroos/Modrić/Banega.
- *Known gap:* no age/DOB in the dataset, so "under-23"-style filters can't be answered
  (the prompt tells the LLM to ignore them).

## Phase 4 — Performance Score + XGBoost/SHAP ✅
- No future/age/value labels exist, so instead of a fabricated "potential" target we ship:
- **Performance Score (0-100)** — transparent, position-weighted percentile composite,
  fully traceable to metric contributions: [ml/scoring.py](../src/footyvision/ml/scoring.py).
- **XGBoost position classifier** on a REAL label (position group from style), with honest
  held-out evaluation (~85% accuracy), **SHAP** feature importance, per-player *style
  profiles*, and *role-mismatch* detection (e.g. Dani Alves plays like a MID):
  [ml/talent.py](../src/footyvision/ml/talent.py).
- API: `GET /players/{id}/score`, `GET /rankings`, `GET /talent/model-info`.
  CLI: `footyvision talent-report` (accuracy + SHAP + mismatches).

## Phase 5 — Market Value Predictor ✅ (built on real data; honest low signal)
- **LightGBM** regression on log(value) + **age**, **SHAP** importance, fuzzy name matching,
  and a "bargains" view: [ml/value.py](../src/footyvision/ml/value.py). Data from **Kaggle**
  (davidcariboo/player-scores Transfermarkt values) via the API + [etl/transfermarkt.py](../src/footyvision/etl/transfermarkt.py).
  CLI: `footyvision value-report`. (A SoFIFA loader also exists but SoFIFA scraping is too slow here.)
- **Honest evaluation:** matched 245/411 players; held-out **R² ≈ 0.05, MAE ≈ €5M**. SHAP
  correctly ranks **age** as the dominant driver — but one season of public per-90 stats + age
  explains little market-value variance (value is driven by reputation, club, potential,
  marketing, contract — none in our features). Cross-source entity resolution (StatsBomb full
  Spanish names ↔ short Transfermarkt names) adds label noise. A working pipeline with a
  truthful "this doesn't predict well" result — not an overfit vanity metric.
- **Career Simulator** — infeasible: needs longitudinal labelled careers (not available free).

## Phase 5b — RAG Scouting Assistant ✅
- Player profiles embedded with the local **`nomic-embed-text`** model into a numpy vector
  store ([rag/store.py](../src/footyvision/rag/store.py), [rag/profiles.py](../src/footyvision/rag/profiles.py));
  a question is embedded, the nearest profiles retrieved, and the LLM answers **grounded in
  them** (cites names, no invention): [rag/assistant.py](../src/footyvision/rag/assistant.py).
- API: `POST /assistant`. CLI: `footyvision index` (build/persist the vector store).
- Validated: *"médio defensivo que ganhe bolas e intercete"* → retrieves Busquets/Camacho/
  Celso Borges and recommends Camacho with his real intercepting/tackling traits.
- Retrieval is hybrid: named players are pinned lexically and their profile centroid —
  not the question text — is the vector query, because proper nouns otherwise dominate the
  embedding and pull in similar-*sounding* names. Profiles carry per-90 values and
  percentiles so comparative questions are answerable.
- The embedding model was picked by measuring, not by leaderboard — `scripts/eval_embeddings.py`
  scores position accuracy and trait strength over PT/EN query pairs on these 411 profiles:

  | config | position@6 | trait pct | EN | PT |
  |---|---|---|---|---|
  | nomic, no prefixes (original) | 0.60 | 59.7 | 0.90 | 0.30 |
  | nomic + prefixes | 0.73 | 60.9 | 0.93 | 0.53 |
  | EmbeddingGemma, no prefixes | 0.68 | 69.9 | 0.83 | 0.53 |
  | **EmbeddingGemma + prefixes** | **0.85** | **78.0** | **0.97** | **0.73** |

- Honest note: a cross-lingual gap survives (0.97 EN vs 0.73 PT) because the profiles are
  written in English. "extremo driblador que passa pelos defesas" is still the worst case —
  its English twin scores 1.00, the Portuguese 0.17. Translating the profiles, or indexing
  them in both languages, is the open thread.

---

## What's next

Phases 0–5b are shipped. These are the open threads, roughly in the order they'd add the
most value:

**Product**
- **Age/DOB in the dataset** — the single biggest unlock. Without it, "under-23" filters
  are impossible (Phase 3) and the value model is missing its strongest feature at
  inference time (Phase 5).
- **Stream report tokens** and cache generated reports; let the user pick language/length.
- **UMAP 2D "player map"** — the Phase 1 stretch goal, visualisation only.
- **Compare view in the dashboard** — two radars side by side, driven by the similarity
  endpoint rather than the standalone demo page.

**Data**
- **A second full season** to compare like-for-like across years; StatsBomb Open Data has
  only La Liga 2015/16 complete, so this needs a different source.
- **Better entity resolution** for the Transfermarkt join (245/411 matched today) — the
  label noise is a real part of the Phase 5 R².
- **A larger embedding model** for the RAG store: retrieval is good for defensive and
  dribbling queries, noisier for creative/finishing ones.

**Engineering**
- **Alembic migrations** — the scaffolding exists and an initial revision is checked in;
  `init-db` still uses `create_all` for a fast first run.
- **Auth and rate limiting** before this is ever exposed beyond localhost (see
  [SECURITY.md](../SECURITY.md)).
- **Coverage of the ETL path** — the aggregation maths is unit-tested, the StatsBomb
  extraction is not (it needs network fixtures).

## Cross-cutting quality (do throughout)
- Tests for each new computation; keep aggregation logic DB/network-free and unit-tested.
- Document every heuristic (e.g. what "progressive" means) so results are explainable.
