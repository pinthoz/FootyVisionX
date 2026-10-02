"""Grounded scouting reports: the LLM explains numbers we computed, never invents them.

Pipeline: feature frame -> radar percentiles + similar players -> a compact factual
context -> a prompt that forbids inventing stats -> the local LLM writes the prose.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from footyvision.llm.client import LLMClient
from footyvision.ml.features import load_feature_frame
from footyvision.ml.similarity import find_similar, radar_percentiles

# Human-readable metric labels for the report context.
READABLE: dict[str, str] = {
    "goals_per90": "goals/90",
    "assists_per90": "assists/90",
    "shots_per90": "shots/90",
    "xg_per90": "xG/90",
    "passes_per90": "passes/90",
    "passes_completed_per90": "completed passes/90",
    "progressive_passes_per90": "progressive passes/90",
    "dribbles_per90": "dribbles/90",
    "dribbles_completed_per90": "completed dribbles/90",
    "carries_per90": "carries/90",
    "progressive_carries_per90": "progressive carries/90",
    "tackles_per90": "tackles/90",
    "interceptions_per90": "interceptions/90",
    "blocks_per90": "blocks/90",
    "clearances_per90": "clearances/90",
    "ball_recoveries_per90": "ball recoveries/90",
    "pressures_per90": "pressures/90",
}


def assemble_context(
    target: pd.Series,
    group: str,
    metrics: dict[str, dict[str, float]],
    similar: list[dict[str, Any]],
    n_strengths: int = 5,
    n_weaknesses: int = 4,
) -> dict[str, Any]:
    """Turn computed features into a compact, JSON-serialisable scouting context."""
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["percentile"], reverse=True)

    def fmt(items: list[tuple[str, dict[str, float]]]) -> list[dict[str, Any]]:
        return [
            {
                "metric": READABLE.get(k, k),
                "value": round(v["value"], 2),
                "percentile": v["percentile"],
            }
            for k, v in items
        ]

    return {
        "name": target["name"],
        "position": target.get("primary_position"),
        "position_group": group,
        "minutes": int(target["minutes"]),
        "matches_played": int(target.get("matches_played", 0)),
        "strengths": fmt(ranked[:n_strengths]),
        "weaknesses": fmt(ranked[-n_weaknesses:][::-1]),
        "similar_players": similar,
    }


def build_prompt(context: dict[str, Any]) -> tuple[str, str]:
    """Return (system, user) messages. Percentiles are vs peers in the same position."""
    system = (
        "You are a professional football scout. You write concise, objective scouting "
        "reports in English. Base the report EXCLUSIVELY on the data provided — never "
        "invent numbers, clubs, ages or statistics that are not in the context. "
        "Percentiles are relative to players in the same position. Structure the report "
        "with the sections: Summary, Strengths, Weaknesses, Tactical Fit, "
        "Development Potential, Risk."
    )

    def lines(items: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"  - {i['metric']}: {i['value']} (percentile {i['percentile']})" for i in items
        )

    sims = (
        ", ".join(f"{s['name']} ({s['similarity']:.2f})" for s in context["similar_players"])
        or "n/a"
    )

    user = (
        f"Player: {context['name']}\n"
        f"Position: {context['position']} (group {context['position_group']})\n"
        f"Minutes: {context['minutes']} across {context['matches_played']} matches\n\n"
        f"Strengths (highest percentiles):\n{lines(context['strengths'])}\n\n"
        f"Weaknesses (lowest percentiles):\n{lines(context['weaknesses'])}\n\n"
        f"Most similar players (cosine similarity): {sims}\n\n"
        "Write the scouting report."
    )
    return system, user


def build_report_context(
    session: Session,
    player_id: int,
    min_minutes: float | None = None,
    competition_id: int | None = None,
    season_id: int | None = None,
    top_similar: int = 5,
) -> dict[str, Any] | None:
    """Load data and assemble the factual context, or None if the player is absent."""
    frame = load_feature_frame(session, min_minutes, competition_id, season_id)
    radar = radar_percentiles(frame, player_id)
    if radar is None:
        return None
    target, group, metrics = radar

    sim = find_similar(frame, player_id, top_n=top_similar)
    similar: list[dict[str, Any]] = []
    if sim is not None:
        _, results = sim
        similar = [
            {"name": r["name"], "similarity": round(float(r["similarity"]), 2)}
            for _, r in results.iterrows()
        ]
    return assemble_context(target, group, metrics, similar)


def generate_report(
    session: Session,
    player_id: int,
    client: LLMClient | None = None,
    **context_kwargs: Any,
) -> dict[str, Any] | None:
    """Full pipeline: context + LLM prose. Returns {context, report} or None if absent."""
    context = build_report_context(session, player_id, **context_kwargs)
    if context is None:
        return None
    client = client or LLMClient()
    system, user = build_prompt(context)
    # Generous budget: a 6-section report plus a reasoning model's hidden thinking.
    report = client.chat(system, user, max_tokens=3000)
    return {"context": context, "report": report}
