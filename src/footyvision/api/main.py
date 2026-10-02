from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from footyvision import __version__
from footyvision.api import docs
from footyvision.api.routers import (
    assistant,
    coverage,
    evaluation,
    health,
    metrics,
    pitch,
    players,
    reports,
    search,
    similarity,
    talent,
    teams,
    value,
    warm,
)
from footyvision.config import get_settings

app = FastAPI(
    # The generated Swagger page is replaced by a themed one in api/docs.py; the schema
    # behind it is untouched, so the reference stays generated rather than written.
    docs_url=None,
    title="FootyVision API",
    version=__version__,
    description="AI football scouting platform — similarity, talent scoring and LLM reports.",
)

# Browsers may only call this API from the configured frontends. This is hygiene, not a
# security boundary: it constrains other *sites*, not scripts. The rate limiter in
# api/limits.py is what actually protects the LLM budget.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_origin_regex=get_settings().allowed_origin_regex,
    # The dashboard sends no cookies and no auth headers, so there is nothing to share.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(coverage.router)
app.include_router(players.router)
app.include_router(pitch.router)
app.include_router(similarity.router)
app.include_router(reports.router)
app.include_router(search.router)
app.include_router(talent.router)
app.include_router(metrics.router)
app.include_router(assistant.router)
app.include_router(evaluation.router)
app.include_router(teams.router)
app.include_router(value.router)
app.include_router(warm.router)

# After the routers, because the page is built from the schema they define.
docs.install(app)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {"name": "FootyVision API", "version": __version__, "docs": "/docs"}
