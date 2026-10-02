"use client";

import { useState } from "react";
import Markdown from "./Markdown";
import SoccerBall from "./SoccerBall";
import JugglingBoot from "./JugglingBoot";
import { LLM_UNAVAILABLE, Player, api } from "../lib/api";
export default function Report({
  playerA,
  playerB,
}: {
  playerA: Player | null;
  playerB: Player | null;
}) {
  const [text, setText] = useState(
    "Pick Player A or Player B, then generate a written scouting dossier."
  );
  const [selectedForReport, setSelectedForReport] = useState<"A" | "B">("A");
  const [busy, setBusy] = useState(false);

  const targetPlayer = selectedForReport === "A" ? playerA : playerB;

  async function generate() {
    if (!targetPlayer) return;
    setBusy(true);
    setText(`Writing the report for ${targetPlayer.name}…`);
    const { status, data } = await api.report(targetPlayer.id);
    setBusy(false);
    if (status === 503) return setText(LLM_UNAVAILABLE);
    if (!data) return setText("Report generation failed.");
    setText(data.report);
  }

  return (
    <div className="scouting-report-container">
      <div className="report-header-row">
        <div className="report-target-selector">
          <span>Generate Dossier for:</span>
          <button
            className={`chip ${selectedForReport === "A" ? "active" : ""}`}
            onClick={() => setSelectedForReport("A")}
            disabled={!playerA}
          >
            {playerA ? `Player A (${playerA.name})` : "Player A (Empty)"}
          </button>
          <button
            className={`chip ${selectedForReport === "B" ? "active" : ""}`}
            onClick={() => setSelectedForReport("B")}
            disabled={!playerB}
          >
            {playerB ? `Player B (${playerB.name})` : "Player B (Empty)"}
          </button>
        </div>

        <button
          className="action-btn primary-btn"
          onClick={generate}
          disabled={!targetPlayer || busy}
        >
          {`Generate report${targetPlayer ? ` for ${targetPlayer.name}` : ""}`}
        </button>
      </div>

      {busy ? (
        <div className="report-generating-card">
          <div className="report-gen-header">
            <div className="report-badge-pulsing">
              <SoccerBall size={12} mode="spin" />
              AI SCOUT ENGINE ACTIVE
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <JugglingBoot scale={0.55} />
              <span className="report-gen-target">
                Synthesizing tactical dossier for <strong>{targetPlayer?.name}</strong>
                <span className="busy-dots">
                  <span />
                  <span />
                  <span />
                </span>
              </span>
            </div>
          </div>

          <div className="tactical-pass-lane">
            <div className="tactical-node active">
              <span>⚡ Scout Engine</span>
            </div>
            <div className="tactical-pass-track" />
            <div className="tactical-rolling-ball">
              <SoccerBall size={16} mode="spin" glow={true} />
            </div>
            <div className="tactical-node active">
              <span>🎯 {targetPlayer?.name || "Player Dossier"}</span>
            </div>
          </div>

          <div className="report-skeleton-lines">
            <div className="skeleton-line long" />
            <div className="skeleton-line medium" />
            <div className="skeleton-line short" />
            <div className="skeleton-spacer" />
            <div className="skeleton-line medium" />
            <div className="skeleton-line long" />
            <div className="skeleton-line short" />
          </div>
          <div className="ai-scanline-bar" />
        </div>
      ) : (
        <div className="prose report-document-box">
          <Markdown text={text} />
        </div>
      )}
    </div>
  );
}
