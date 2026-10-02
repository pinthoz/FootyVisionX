"""Lazily build/load and cache the player vector store for the assistant.

The index lives in Postgres. A file on disk is still read as a fallback for a local
checkout that has one, but nothing writes to disk in production: on a host with an
ephemeral filesystem the file would vanish on every restart and the first request after
each cold start would re-embed the whole squad inline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from footyvision.config import get_settings
from footyvision.llm.client import LLMClient
from footyvision.ml.features import load_feature_frame
from footyvision.rag.profiles import build_profiles
from footyvision.rag.store import VectorStore

STORE_PATH = Path("data/player_vectors.npz")

logger = logging.getLogger(__name__)

_STORE: VectorStore | None = None


def expected_embed_model(client: LLMClient | None = None) -> str:
    """The model this process would embed a *query* with, given the current config."""
    settings = get_settings()
    if settings.finetuned_embed_path:
        return f"finetuned:{settings.finetuned_embed_path}"
    client = client or LLMClient()
    return client.embed_model


def build_store(session: Session, client: LLMClient | None = None) -> VectorStore:
    """Embed every player profile and persist the index to the database."""
    client = client or LLMClient()
    frame = load_feature_frame(session, get_settings().min_minutes)
    docs = build_profiles(frame)
    store = VectorStore.build(docs, client)
    # Recorded after the call, because the local->cloud fallback decides at runtime.
    store.save_db(session, client.last_embed_model)
    return store


def store_is_stale(session: Session, store: VectorStore) -> int:
    """How many players are in the pool but not in the index (0 when in step).

    Loading a season does not touch the index, and a stale index fails silently: the
    assistant answers confidently from whichever players it holds, never saying it has
    not heard of the rest. That is worse than an error, so the gap is measured.
    """
    pool = load_feature_frame(session, get_settings().min_minutes)
    return max(len(pool) - len(store), 0)


def get_store(session: Session, rebuild: bool = False) -> VectorStore:
    """Return the cached store; read it from the database, then disk, then build it."""
    global _STORE
    if _STORE is not None and not rebuild:
        return _STORE

    if not rebuild:
        stored = VectorStore.load_db(session)
        if stored is not None:
            _STORE, built_with = stored
            _warn_if_mismatched(built_with)
            _warn_if_stale(session, _STORE)
            return attach_roles(session, _STORE)
        if STORE_PATH.exists():
            logger.info("no index in the database; falling back to %s", STORE_PATH)
            _STORE = VectorStore.load(STORE_PATH)
            _warn_if_stale(session, _STORE)
            return attach_roles(session, _STORE)

    _STORE = build_store(session)
    return attach_roles(session, _STORE)


def attach_roles(session: Session, store: VectorStore) -> VectorStore:
    """Give every indexed player the playing role of their most-played season.

    Derived here, from the stats table, rather than stored with the vectors. The role is a
    pure function of the primary position the database already holds, so storing it too
    would have meant a migration and re-embedding every profile for a column that can be
    computed in one query — and would go stale the same way the position already cannot.
    A player with no recorded position gets none, which the filter treats as failing a
    role requirement rather than passing it.
    """
    from sqlalchemy import select

    from footyvision.db.models import PlayerSeasonStats
    from footyvision.ml.features import position_role

    rows = session.execute(
        select(
            PlayerSeasonStats.player_id,
            PlayerSeasonStats.primary_position,
            PlayerSeasonStats.minutes,
        )
    ).all()
    best: dict[int, tuple[float, str | None]] = {}
    for player_id, position, minutes in rows:
        if player_id not in best or (minutes or 0) > best[player_id][0]:
            best[player_id] = (minutes or 0, position)
    roles = {pid: position_role(pos) for pid, (_, pos) in best.items() if pos}
    store.position_role = np.asarray([roles.get(int(pid)) for pid in store.ids], dtype=object)
    return store


def _warn_if_mismatched(built_with: str) -> None:
    """Vectors are only comparable to a query encoded by the same model.

    Building the index with one model and querying with another returns confident
    nonsense rather than an error, which is exactly the kind of failure worth shouting
    about — an index built by EmbeddingGemma and queried by a cloud model shares no
    vector space with it at all.
    """
    expected = expected_embed_model()
    if built_with != expected:
        logger.warning(
            "RAG index was built with %r but this process embeds queries with %r. "
            "The vectors are not comparable — rebuild with `footyvision index`.",
            built_with,
            expected,
        )


def _warn_if_stale(session: Session, store: VectorStore) -> None:
    missing = store_is_stale(session, store)
    if missing:
        logger.warning(
            "RAG index holds %d profiles but the pool has %d — %d players are invisible "
            "to the assistant. Run `footyvision index` to rebuild.",
            len(store),
            len(store) + missing,
            missing,
        )
