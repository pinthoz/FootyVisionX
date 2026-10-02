# Contributing to FootyVision

Thanks for taking a look. Issues, questions and pull requests are all welcome.

## Getting set up

```bash
git clone https://github.com/<your-user>/FootyVisionX.git
cd FootyVision

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows;  macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env              # adjust if your Postgres/LLM ports differ
docker compose up -d db
footyvision init-db
```

The frontend lives in `frontend/web`:

```bash
cd frontend/web && npm install && npm run dev
```

## Before you open a pull request

```bash
ruff check src tests
ruff format --check src tests
pytest
```

CI runs exactly these on Python 3.11 and 3.12, plus `next build` for the dashboard.

## Ground rules

- **Tests stay offline.** The suite must run with no database, no network and no LLM
  server. Use fixtures and fakes — see `tests/test_rag.py` for the pattern.
- **The LLM never invents numbers.** Anything numeric shown to a user comes from the ETL
  or an ML model; the LLM only explains values it was handed. Keep prompts grounded.
- **No raw SQL from user input.** Natural-language search goes through the validated
  `PlayerQuery` model in `src/footyvision/search/query.py`.
- **Be honest about the data.** If a model scores poorly, report the real number — see the
  "Data reality" section of the README. Inflated metrics are worse than modest ones.
- **Never commit** `.env`, credentials, datasets or model artefacts. `data/` is gitignored.

## Style

- Python 3.11+, `from __future__ import annotations`, type hints on public functions.
- Line length 100, formatted and linted with ruff (config in `pyproject.toml`).
- Commit messages in the imperative mood: `Add value model SHAP export`.

## Project layout

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the pieces fit together, and
[`docs/ROADMAP.md`](docs/ROADMAP.md) for what is planned.
