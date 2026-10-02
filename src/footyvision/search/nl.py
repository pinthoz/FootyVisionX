"""Natural-language → structured PlayerQuery via the LLM.

The LLM only ever emits JSON for the PlayerQuery schema; Pydantic validation rejects
anything outside the whitelist. No SQL is ever produced by the model.
"""

from __future__ import annotations

import json
import re

from footyvision.llm.client import LLMClient
from footyvision.search.query import SEARCHABLE_FIELDS, PlayerQuery

_FEMALE_RE = re.compile(
    r"\b(female|women|woman|mulher|mulheres|jogadora|jogadoras|feminin[oa]s?)\b", re.IGNORECASE
)
_MALE_RE = re.compile(r"\b(male|men|man|homem|homens|masculin[oa]s?)\b", re.IGNORECASE)


class NLParseError(ValueError):
    """The LLM output could not be parsed into a valid PlayerQuery."""


def _fields_help() -> str:
    return ", ".join(sorted(SEARCHABLE_FIELDS))


def build_nl_prompt(text: str) -> tuple[str, str]:
    system = (
        "You translate a football scout's request into a JSON query object. "
        "Output ONLY a JSON object, no prose, no code fences.\n"
        "Schema:\n"
        "{\n"
        '  "position_group": one of "GK"|"DEF"|"MID"|"FWD" or null,\n'
        '  "gender": one of "female"|"male" or null,\n'
        '  "foot": one of "left"|"right"|"both" or null,\n'
        '  "competition": league-name substring or null,\n'
        '  "nationality": player country-name substring or null,\n'
        '  "min_minutes": number or null,\n'
        '  "conditions": [{"field": <field>, "op": "gt"|"gte"|"lt"|"lte"|"eq", "value": number}],\n'
        '  "order_by": <field> or null,\n'
        '  "order_desc": boolean,\n'
        '  "limit": integer (1-100)\n'
        "}\n"
        f"Allowed <field> values (per-90 unless stated): {_fields_help()}.\n"
        "'gender' filters by competition gender: 'female' for women's football / female players "
        "(e.g. 'female', 'women', 'mulher', 'mulheres', 'jogadora', 'feminino'), "
        "or 'male' for male players "
        "(e.g. 'male', 'men', 'homem', 'homens', 'masculino'). If not specified, set null.\n"
        "Preferred foot is the top-level 'foot' key, never a condition: "
        '\'left-footed wingers\' sets "foot": "left". Height in centimetres is the '
        "'height_cm' field, so 'taller than 190cm' is a condition on it.\n"
        "'nationality' is the player's own country and 'competition' is the league, which "
        "are not the same thing: 'Brazilians in La Liga' sets both. Give nationality as the "
        'country name in English ("Brazil", not "Brazilian"), because it is matched as a '
        "substring of the stored country.\n"
        "Age IS available as the 'age' field, in years at the middle of the season: "
        '\'under 23\' becomes {"field": "age", "op": "lt", "value": 23} and '
        "'over 30' uses op 'gt'. Map metric names to the closest allowed field "
        "(e.g. 'xG per 90' -> 'xg_per90', 'progressive passes' -> 'progressive_passes_per90', "
        "'shots' -> 'shots_per90')."
    )
    user = f"Request: {text}\nJSON:"
    return system, user


def _extract_json(raw: str) -> str:
    """Pull the first JSON object out of the model output (tolerates fences/prose)."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise NLParseError(f"No JSON object found in LLM output: {raw[:200]!r}")
    return raw[start : end + 1]


def parse_nl(text: str, client: LLMClient | None = None) -> PlayerQuery:
    """Ask the LLM to structure `text`, then validate into a PlayerQuery."""
    client = client or LLMClient()
    system, user = build_nl_prompt(text)
    # Headroom for reasoning models that "think" before emitting the JSON.
    raw = client.chat(system, user, temperature=0.1, max_tokens=1500)
    payload = _extract_json(raw)
    try:
        q = PlayerQuery.model_validate(json.loads(payload))
    except (json.JSONDecodeError, ValueError) as exc:
        raise NLParseError(f"Invalid query from LLM: {exc}. Raw: {payload[:200]!r}") from exc

    # Deterministic safeguard: if user explicitly requested female or male players,
    # ensure query.gender is accurately set regardless of LLM quirks.
    if _FEMALE_RE.search(text) and not _MALE_RE.search(text):
        q.gender = "female"
    elif _MALE_RE.search(text) and not _FEMALE_RE.search(text):
        q.gender = "male"

    return q
