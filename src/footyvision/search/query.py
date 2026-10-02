"""Safe structured player search.

The LLM (or a client) produces a `PlayerQuery`; Pydantic validation is the security
boundary — only whitelisted fields and operators are accepted, so no arbitrary SQL or
attribute access is ever possible. Execution runs over the in-memory feature frame.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from footyvision.ml.features import PER90_FEATURES, load_feature_frame

# Fields a query is allowed to filter or sort on. `age` is the player's age at the
# midpoint of the season the row describes, not today — a 2015/16 row is a 2015/16
# player. Rows without a date of birth carry NaN and are excluded by any age condition.
SEARCHABLE_FIELDS: frozenset[str] = frozenset(PER90_FEATURES) | {
    "minutes",
    "matches_played",
    "age",
    "height_cm",
}

Operator = Literal["gt", "gte", "lt", "lte", "eq"]
_OPS = {
    "gt": lambda s, v: s > v,
    "gte": lambda s, v: s >= v,
    "lt": lambda s, v: s < v,
    "lte": lambda s, v: s <= v,
    "eq": lambda s, v: s == v,
}


class Condition(BaseModel):
    field: str
    op: Operator
    value: float

    @field_validator("field")
    @classmethod
    def _known_field(cls, v: str) -> str:
        if v not in SEARCHABLE_FIELDS:
            raise ValueError(f"Unknown field '{v}'. Allowed: {sorted(SEARCHABLE_FIELDS)}")
        return v


class PlayerQuery(BaseModel):
    position_group: Literal["GK", "DEF", "MID", "FWD"] | None = None
    gender: Literal["male", "female"] | None = None
    # Categorical, so it is an equality filter rather than a numeric condition.
    foot: Literal["left", "right", "both"] | None = None
    competition: str | None = Field(None, description="Substring matched against league name.")
    # The player's own country, not the league's. Matched as a substring so "Venezuela"
    # finds "Venezuela (Bolivarian Republic)" and the caller need not know how the source
    # spells it. Unlike foot and age it is recorded for every player in the pool.
    nationality: str | None = Field(
        None, description="Substring matched against the player's country."
    )
    min_minutes: float | None = None
    conditions: list[Condition] = Field(default_factory=list)
    order_by: str | None = None
    order_desc: bool = True
    limit: int = Field(20, ge=1, le=100)

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data: Any) -> Any:
        # LLMs often emit explicit nulls (e.g. "limit": null); drop them so field
        # defaults apply instead of failing validation on a non-optional field.
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data

    @field_validator("order_by")
    @classmethod
    def _known_order(cls, v: str | None) -> str | None:
        if v is not None and v not in SEARCHABLE_FIELDS:
            raise ValueError(f"Unknown order_by '{v}'. Allowed: {sorted(SEARCHABLE_FIELDS)}")
        return v


def execute_query(session: Session, query: PlayerQuery) -> list[dict[str, Any]]:
    """Run a validated PlayerQuery against the feature frame; return result rows."""
    frame = load_feature_frame(session, query.min_minutes)
    if frame.empty:
        return []

    if query.competition:
        frame = frame[frame["competition"].str.contains(query.competition, case=False, na=False)]
    if query.position_group:
        frame = frame[frame["position_group"] == query.position_group]
    if query.gender and "gender" in frame.columns:
        frame = frame[frame["gender"].astype(str).str.lower() == query.gender.lower()]
    if query.foot:
        frame = frame[frame["foot"] == query.foot]
    if query.nationality:
        frame = frame[frame["nationality"].str.contains(query.nationality, case=False, na=False)]
    for cond in query.conditions:
        frame = frame[_OPS[cond.op](frame[cond.field], cond.value)]

    if query.order_by:
        frame = frame.sort_values(query.order_by, ascending=not query.order_desc)
    frame = frame.head(query.limit)

    # Return identity columns plus every field referenced by the query, so the caller
    # can see exactly why each row matched.
    referenced = {c.field for c in query.conditions}
    if query.order_by:
        referenced.add(query.order_by)
    referenced.update({"minutes", "matches_played"})

    rows: list[dict[str, Any]] = []
    for _, r in frame.iterrows():
        rows.append(
            {
                "player_id": int(r["player_id"]),
                "name": r["name"],
                "competition": r["competition"],
                "primary_position": r["primary_position"],
                "position_group": r["position_group"],
                "gender": r.get("gender"),
                "nationality": r.get("nationality"),
                "stats": {f: round(float(r[f]), 3) for f in sorted(referenced)},
            }
        )
    return rows
