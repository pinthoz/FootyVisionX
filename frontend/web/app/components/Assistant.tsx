"use client";

import { useState } from "react";
import PlayerAvatar from "./PlayerAvatar";
import Markdown from "./Markdown";
import JugglingBoot from "./JugglingBoot";
import { AssistantResult, LLM_UNAVAILABLE, Player, api } from "../lib/api";
export default function Assistant({
  onPick,
}: {
  onPick: (p: Player, target?: "A" | "B") => void;
}) {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  // The model ran out of tokens rather than finishing. The text is trimmed to its last
  // full sentence server-side, so nothing looks broken — which is exactly why it has to
  // be said out loud instead of left to look complete.
  const [truncated, setTruncated] = useState(false);
  const [sources, setSources] = useState<AssistantResult["sources"]>([]);
  // Players a filter removed for want of the attribute rather than for failing it.
  // Rendered here rather than asked of the model: a claim about the search is
  // grounded in no profile, and the model paraphrased it into things that were false.
  const [omitted, setOmitted] = useState<Record<string, number> | null>(null);
  // The metric the sources are ordered by, when the question asked who leads in one.
  const [rankedBy, setRankedBy] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const promptSuggestions = [
    "Find an aggressive defensive midfielder who leads in ball recoveries and progressive passes",
    "Identify a creative playmaker with high xG and take-ons",
    "Who are the best pressing wingers with elite workrate?",
  ];

  async function ask(questionToAsk?: string) {
    const query = questionToAsk ?? q;
    if (!query.trim()) return;
    setBusy(true);
    setAnswer("");
    setTruncated(false);
    setSources([]);
    setOmitted(null);
    setRankedBy(null);
    const { status, data } = await api.assistant(query.trim());
    setBusy(false);
    if (status === 503) return setAnswer(LLM_UNAVAILABLE);
    if (!data) return setAnswer("Scouting request failed.");
    setAnswer(data.answer);
    setTruncated(Boolean(data.truncated));
    setSources(data.sources);
    setOmitted(data.not_considered ?? null);
    setRankedBy(data.ranked_by ?? null);
  }

  return (
    <div className="ai-chat-container">
      <div className="ai-input-group">
        <input
          type="text"
          placeholder="e.g. A box-to-box midfielder with high tackles and progressive carries"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
        />
        <button
          className="action-btn primary-btn"
          onClick={() => ask()}
          disabled={busy}
        >
          Ask
        </button>
      </div>

      <div className="prompt-suggestions">
        <span className="suggestion-label">Try asking:</span>
        {promptSuggestions.map((s, idx) => (
          <button
            key={idx}
            className="suggestion-chip"
            onClick={() => {
              setQ(s);
              ask(s);
            }}
          >
            {s}
          </button>
        ))}
      </div>

      {busy && (
        <div className="ai-thinking-state">
          <JugglingBoot scale={0.72} />
          <div className="ai-thinking-text">
            <div className="ai-thinking-title">
              Scout AI is analyzing
              <span className="busy-dots">
                <span />
                <span />
                <span />
              </span>
            </div>
            <div className="ai-thinking-subtitle">
              Retrieving player vectors, grounding tactical stats & formulating answer…
            </div>
          </div>
          <div className="ai-scanline-bar" />
        </div>
      )}

      {answer ? (
        <div className="prose ai-response-box">
          <Markdown text={answer} />
          {truncated && (
            <p className="chartnote">
              The model reached its token limit before finishing, so this answer stops
              early. Asking for fewer players, or for one thing at a time, gets a complete
              one.
            </p>
          )}
        </div>
      ) : !busy ? (
        <div className="ai-idle-hint">
          Describe the player you want, or ask how two compare. Every answer names its sources.
        </div>
      ) : null}

      {omitted && Object.keys(omitted).length > 0 && (
        <div className="ai-omitted">
          Not searched:{" "}
          {Object.entries(omitted)
            .map(([field, n]) => `${n.toLocaleString()} players with no recorded ${field}`)
            .join(", ")}
          . Date of birth and preferred foot come from a men&apos;s football source, so
          those requirements cannot be checked against the women&apos;s competitions.
        </div>
      )}

      {sources.length > 0 && (
        <div className="sources-card">
          <div className="sources-title">
            {rankedBy
              ? `Top ${sources.length} by ${rankedBy} per 90, highest first (click to compare):`
              : "Retrieved Players (Click to compare):"}
          </div>
          <div className="sources-grid">
            {sources.map((s) => (
              <div key={s.player_id} className="source-chip">
                <PlayerAvatar name={s.name} size="sm" themeColor="var(--accent)" />
                <span className="source-name">{s.name}</span>
                <div className="source-actions">
                  <button
                    className="micro-btn a-btn"
                    title="Load as Player A"
                    onClick={() => onPick({ id: s.player_id, name: s.name, country: null }, "A")}
                  >
                    + A
                  </button>
                  <button
                    className="micro-btn b-btn"
                    title="Load as Player B"
                    onClick={() => onPick({ id: s.player_id, name: s.name, country: null }, "B")}
                  >
                    + B
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
