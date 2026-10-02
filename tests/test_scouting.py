"""Unit tests for report grounding — no DB and no live LLM required."""

from __future__ import annotations

import pandas as pd

from footyvision.llm import scouting
from footyvision.llm.scouting import assemble_context, build_prompt, generate_report


def _target() -> pd.Series:
    return pd.Series(
        {
            "name": "Test Player",
            "primary_position": "Center Forward",
            "minutes": 2400,
            "matches_played": 30,
        }
    )


def _metrics() -> dict[str, dict[str, float]]:
    return {
        "goals_per90": {"value": 0.8, "percentile": 95.0},
        "xg_per90": {"value": 0.7, "percentile": 90.0},
        "tackles_per90": {"value": 0.3, "percentile": 10.0},
        "interceptions_per90": {"value": 0.2, "percentile": 5.0},
    }


def test_assemble_context_orders_strengths_and_weaknesses():
    ctx = assemble_context(
        _target(),
        "FWD",
        _metrics(),
        similar=[{"name": "Peer", "similarity": 0.9}],
        n_strengths=2,
        n_weaknesses=2,
    )
    assert ctx["name"] == "Test Player"
    assert ctx["position_group"] == "FWD"
    # Highest percentile first among strengths, lowest among weaknesses.
    assert ctx["strengths"][0]["metric"] == "goals/90"
    assert ctx["weaknesses"][0]["metric"] == "interceptions/90"
    assert ctx["similar_players"][0]["name"] == "Peer"


def test_build_prompt_grounds_and_forbids_invention():
    ctx = assemble_context(_target(), "FWD", _metrics(), similar=[])
    system, user = build_prompt(ctx)
    assert "never invent" in system.lower()
    assert "Test Player" in user
    assert "goals/90" in user


class _FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, system: str, user: str, **_) -> str:
        self.calls.append((system, user))
        return "Summary: benchmark player.\nRisk: low."


def test_generate_report_uses_context_and_client(monkeypatch):
    fixed = assemble_context(_target(), "FWD", _metrics(), similar=[])
    monkeypatch.setattr(scouting, "build_report_context", lambda *a, **k: fixed)

    client = _FakeClient()
    result = generate_report(session=None, player_id=1, client=client)
    assert result["context"]["name"] == "Test Player"
    assert "Summary" in result["report"]
    # The prompt actually sent to the LLM was grounded in our numbers.
    assert "goals/90" in client.calls[0][1]


def test_generate_report_returns_none_when_player_absent(monkeypatch):
    monkeypatch.setattr(scouting, "build_report_context", lambda *a, **k: None)
    assert generate_report(session=None, player_id=999, client=_FakeClient()) is None
