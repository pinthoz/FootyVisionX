"""Database schema for the FootyVision scouting platform.

IDs that originate from StatsBomb (competition_id, season_id, team_id, player_id,
match_id) are reused as primary keys so the ETL is idempotent (upserts by natural key).
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import (
    Date,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from footyvision.db.base import Base

# Counting metrics aggregated from StatsBomb events. Kept as a tuple so the ETL and
# similarity layers can iterate the canonical metric list without hardcoding it.
METRIC_COLUMNS: tuple[str, ...] = (
    "goals",
    "assists",
    "shots",
    "xg",
    "passes",
    "passes_completed",
    "progressive_passes",
    "dribbles",
    "dribbles_completed",
    "carries",
    "progressive_carries",
    "tackles",
    "interceptions",
    "blocks",
    "clearances",
    "ball_recoveries",
    "pressures",
)


class Competition(Base):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(primary_key=True)  # StatsBomb competition_id
    name: Mapped[str] = mapped_column(String(120))
    country: Mapped[str | None] = mapped_column(String(120))
    # "male" | "female", straight from the StatsBomb competition list. Percentiles and the
    # similarity pool are partitioned on it: the two games have different baselines, and a
    # rate ranked across both describes neither.
    gender: Mapped[str | None] = mapped_column(String(10))

    seasons: Mapped[list[Season]] = relationship(back_populates="competition")


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("competition_id", "sb_season_id", name="uq_comp_season"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)
    sb_season_id: Mapped[int] = mapped_column(Integer)  # StatsBomb season_id
    name: Mapped[str] = mapped_column(String(60))  # e.g. "2015/2016"

    competition: Mapped[Competition] = relationship(back_populates="seasons")


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)  # StatsBomb team_id
    name: Mapped[str] = mapped_column(String(120))


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)  # StatsBomb player_id
    name: Mapped[str] = mapped_column(String(120), index=True)
    country: Mapped[str | None] = mapped_column(String(120))
    # The short form the player is actually known by ("Joselu" for "José Luis Sanmartín
    # Mato"). The lineup feed carries it, and it is the form other providers use, which
    # makes it a far better key for cross-source matching than the full legal name.
    nickname: Mapped[str | None] = mapped_column(String(120), index=True)
    # Absent from the open data entirely; backfilled from the Transfermarkt export.
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    # "right" | "left" | "both". The only attribute here that carries a *side*, which is
    # why the exact-position classifier takes it and the side-agnostic ones do not.
    foot: Mapped[str | None] = mapped_column(String(10))
    # A scouting attribute, not a model feature: ablation showed it adds nothing to
    # position prediction (0.489 -> 0.491, inside the noise).
    height_cm: Mapped[float | None] = mapped_column(Float)


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)  # StatsBomb match_id
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)
    sb_season_id: Mapped[int] = mapped_column(Integer, index=True)
    match_date: Mapped[date | None] = mapped_column(Date)
    home_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))


class PlayerVector(Base):
    """The RAG index, stored in the database rather than on disk.

    On a platform with an ephemeral filesystem — Render's free tier, for one — a `.npz`
    written at runtime does not survive a restart, so every cold start would re-embed the
    whole squad inside the first HTTP request that needed it. Postgres already outlives
    the container, so the index lives there.

    `embed_model` is the important column: vectors are only comparable to a query encoded
    by the *same* model, and this instance can embed via a fine-tuned local model, LM
    Studio or a cloud provider. Recording which one produced the index lets a mismatch be
    detected rather than silently returning confident nonsense.
    """

    __tablename__ = "player_vectors"

    # A surrogate key, not the player id: a player who changed league mid-season holds one
    # profile per competition, and both belong in the index.
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    text: Mapped[str] = mapped_column(Text)
    # float32 little-endian, straight from numpy. A float array column would be portable
    # but roughly four times the size and slower to reassemble.
    vector: Mapped[bytes] = mapped_column(LargeBinary)
    dim: Mapped[int] = mapped_column(Integer)
    embed_model: Mapped[str] = mapped_column(String(200), index=True)
    # Attributes the retriever filters on before ranking.
    foot: Mapped[str | None] = mapped_column(String(10))
    age: Mapped[float | None] = mapped_column(Float)
    position_group: Mapped[str | None] = mapped_column(String(20))
    nationality: Mapped[str | None] = mapped_column(String(120))
    # The seventeen per-90 values as a JSON object. The prose profile only names a
    # player's four best and one worst metric, so a question about anything else -- "which
    # midfielders make the most tackles" -- reached an assistant whose context genuinely
    # did not contain the number, and it correctly answered that it did not have the data
    # we had all along. Carried here so the asked-for metric can be added to the context.
    metrics: Mapped[str | None] = mapped_column(Text)


class PlayerMatchStats(Base):
    __tablename__ = "player_match_stats"
    __table_args__ = (UniqueConstraint("match_id", "player_id", name="uq_match_player"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    position: Mapped[str | None] = mapped_column(String(40))
    minutes: Mapped[float] = mapped_column(Float, default=0.0)

    goals: Mapped[float] = mapped_column(Float, default=0.0)
    assists: Mapped[float] = mapped_column(Float, default=0.0)
    shots: Mapped[float] = mapped_column(Float, default=0.0)
    xg: Mapped[float] = mapped_column(Float, default=0.0)
    passes: Mapped[float] = mapped_column(Float, default=0.0)
    passes_completed: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_passes: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles_completed: Mapped[float] = mapped_column(Float, default=0.0)
    carries: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_carries: Mapped[float] = mapped_column(Float, default=0.0)
    tackles: Mapped[float] = mapped_column(Float, default=0.0)
    interceptions: Mapped[float] = mapped_column(Float, default=0.0)
    blocks: Mapped[float] = mapped_column(Float, default=0.0)
    clearances: Mapped[float] = mapped_column(Float, default=0.0)
    ball_recoveries: Mapped[float] = mapped_column(Float, default=0.0)
    pressures: Mapped[float] = mapped_column(Float, default=0.0)


class PlayerSeasonStats(Base):
    """Per player+season totals plus per-90 rates — the table the ML/similarity layer reads."""

    __tablename__ = "player_season_stats"
    __table_args__ = (
        UniqueConstraint("player_id", "competition_id", "sb_season_id", name="uq_player_season"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)
    sb_season_id: Mapped[int] = mapped_column(Integer, index=True)
    primary_position: Mapped[str | None] = mapped_column(String(40))
    matches_played: Mapped[int] = mapped_column(Integer, default=0)
    minutes: Mapped[float] = mapped_column(Float, default=0.0)

    # Totals
    goals: Mapped[float] = mapped_column(Float, default=0.0)
    assists: Mapped[float] = mapped_column(Float, default=0.0)
    shots: Mapped[float] = mapped_column(Float, default=0.0)
    xg: Mapped[float] = mapped_column(Float, default=0.0)
    passes: Mapped[float] = mapped_column(Float, default=0.0)
    passes_completed: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_passes: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles_completed: Mapped[float] = mapped_column(Float, default=0.0)
    carries: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_carries: Mapped[float] = mapped_column(Float, default=0.0)
    tackles: Mapped[float] = mapped_column(Float, default=0.0)
    interceptions: Mapped[float] = mapped_column(Float, default=0.0)
    blocks: Mapped[float] = mapped_column(Float, default=0.0)
    clearances: Mapped[float] = mapped_column(Float, default=0.0)
    ball_recoveries: Mapped[float] = mapped_column(Float, default=0.0)
    pressures: Mapped[float] = mapped_column(Float, default=0.0)

    # Per-90 rates (the comparable features)
    goals_per90: Mapped[float] = mapped_column(Float, default=0.0)
    assists_per90: Mapped[float] = mapped_column(Float, default=0.0)
    shots_per90: Mapped[float] = mapped_column(Float, default=0.0)
    xg_per90: Mapped[float] = mapped_column(Float, default=0.0)
    passes_per90: Mapped[float] = mapped_column(Float, default=0.0)
    passes_completed_per90: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_passes_per90: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles_per90: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles_completed_per90: Mapped[float] = mapped_column(Float, default=0.0)
    carries_per90: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_carries_per90: Mapped[float] = mapped_column(Float, default=0.0)
    tackles_per90: Mapped[float] = mapped_column(Float, default=0.0)
    interceptions_per90: Mapped[float] = mapped_column(Float, default=0.0)
    blocks_per90: Mapped[float] = mapped_column(Float, default=0.0)
    clearances_per90: Mapped[float] = mapped_column(Float, default=0.0)
    ball_recoveries_per90: Mapped[float] = mapped_column(Float, default=0.0)
    pressures_per90: Mapped[float] = mapped_column(Float, default=0.0)
