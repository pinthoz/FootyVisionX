"use client";

import { useState } from "react";
import PlayerAvatar from "./PlayerAvatar";
import JugglingBoot from "./JugglingBoot";
import { LLM_UNAVAILABLE, Player, SearchRow, api } from "../lib/api";
export default function NLSearch({
  onPick,
}: {
  onPick: (p: Player, target?: "A" | "B") => void;
}) {
  const [q, setQ] = useState("");
  const [rows, setRows] = useState<SearchRow[]>([]);
  const [info, setInfo] = useState("");
  const [busy, setBusy] = useState(false);

  async function run() {
    if (!q.trim()) return;
    setBusy(true);
    setInfo("Parsing natural language constraints…");
    setRows([]);
    const { status, data } = await api.nlSearch(q.trim());
    setBusy(false);
    if (status === 503) return setInfo(LLM_UNAVAILABLE);
    if (status === 422) return setInfo("Could not interpret into structured filters.");
    if (!data) return setInfo(`Error code ${status}.`);
    const conds = [
      data.interpreted.gender ? `Gender: ${data.interpreted.gender}` : null,
      data.interpreted.position_group ? `Pos: ${data.interpreted.position_group}` : null,
      data.interpreted.competition ? `Comp: ${data.interpreted.competition}` : null,
      ...(data.interpreted.conditions ?? []).map(
        (c: { field: string; op: string; value: number }) =>
          `${c.field} ${c.op} ${c.value}`
      ),
    ]
      .filter(Boolean)
      .join(" · ");
    setInfo(`${data.count} matches found · ${conds || "all filters passed"}`);
    setRows(data.results);
  }

  return (
    <div className="nl-search-container">
      <div className="ai-input-group">
        <input
          type="text"
          placeholder="e.g. Female player that has more than 0.8 pct shots, or La Liga wingers with xG over 0.4"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <button
          className="action-btn primary-btn"
          onClick={run}
          disabled={busy}
        >
          Run Query
        </button>
      </div>

      {busy && (
        <div className="ai-thinking-state filter-state">
          <JugglingBoot scale={0.72} />
          <div className="ai-thinking-text">
            <div className="ai-thinking-title">
              Filtering Player Pool
              <span className="busy-dots">
                <span />
                <span />
                <span />
              </span>
            </div>
            <div className="ai-thinking-subtitle">
              Parsing natural language constraints into SQL filters & ranking candidate profiles…
            </div>
          </div>
          <div className="ai-scanline-bar filter-bar" />
        </div>
      )}

      {info && !busy && <div className="nl-search-info">{info}</div>}

      <div className="nl-results-grid">
        {rows.map((r) => (
          <div key={r.player_id} className="nl-result-card">
            <div className="nl-card-top">
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <PlayerAvatar name={r.name} size="sm" themeColor="var(--accent)" />
                <span className="nl-player-name">{r.name}</span>
                {r.gender && (
                  <span className={`player-gender-tag ${r.gender}`}>
                    {r.gender === "female" ? "♀ Female" : "♂ Male"}
                  </span>
                )}
              </div>
              <span className="pos-badge">{r.primary_position ?? r.position_group}</span>
            </div>
            <div className="nl-card-comp">{r.competition ?? "Domestic League"}</div>
            <div className="nl-card-actions">
              <button
                className="micro-btn a-btn"
                onClick={() =>
                  onPick({ id: r.player_id, name: r.name, country: null, gender: r.gender }, "A")
                }
              >
                Set to Slot A
              </button>
              <button
                className="micro-btn b-btn"
                onClick={() =>
                  onPick({ id: r.player_id, name: r.name, country: null, gender: r.gender }, "B")
                }
              >
                Set to Slot B
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
