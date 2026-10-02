"use client";

import { useState } from "react";
import { ALL_METRICS, MetricCategory, Radar as RadarData } from "../lib/api";

type StatDuelProps = {
  a: { name: string; radar: RadarData; color?: string } | null;
  b: { name: string; radar: RadarData; color?: string } | null;
};

export default function StatDuel({ a, b }: StatDuelProps) {
  const [filter, setFilter] = useState<MetricCategory>("all");
  const colorA = a?.color || "var(--a)";
  const colorB = b?.color || "var(--b)";

  const metrics = ALL_METRICS.filter(
    (m) => filter === "all" || m.category === filter
  );

  // Compute stat duel score if both players exist
  let aWins = 0;
  let bWins = 0;
  let ties = 0;

  if (a && b) {
    ALL_METRICS.forEach((m) => {
      const aPct = a.radar.metrics[m.key]?.percentile ?? 0;
      const bPct = b.radar.metrics[m.key]?.percentile ?? 0;
      if (Math.abs(aPct - bPct) < 2) {
        ties++;
      } else if (aPct > bPct) {
        aWins++;
      } else {
        bWins++;
      }
    });
  }

  return (
    <div className="stat-duel-container">
      <div className="stat-duel-header">
        <div className="stat-duel-title-group">
          <h3 className="section-title">Head-to-Head Stat Duel</h3>
          <p className="section-sub">
            Direct percentile and per-90 comparison across key skill dimensions
          </p>
        </div>

        <div className="filters">
          {[
            { id: "all", label: "All (17)" },
            { id: "attack", label: "Attacking" },
            { id: "passing", label: "Playmaking" },
            { id: "defense", label: "Defending" },
          ].map((cat) => (
            <button
              key={cat.id}
              className={`chip ${filter === cat.id ? "active" : ""}`}
              onClick={() => setFilter(cat.id as MetricCategory)}
            >
              {cat.label}
            </button>
          ))}
        </div>
      </div>

      {a && b && (
        <div className="duel-scoreboard">
          <div className="duel-score-player player-a-side">
            <span className="duel-name" style={{ color: colorA }}>{a.name}</span>
            <span className="duel-badge-win" style={{ borderColor: colorA, color: colorA }}>
              {aWins} advantages
            </span>
          </div>
          <div className="duel-vs-badge">
            <span>VS</span>
            {ties > 0 && <span className="duel-ties">{ties} even</span>}
          </div>
          <div className="duel-score-player player-b-side">
            <span className="duel-badge-win" style={{ borderColor: colorB, color: colorB }}>
              {bWins} advantages
            </span>
            <span className="duel-name" style={{ color: colorB }}>{b.name}</span>
          </div>
        </div>
      )}

      <div className="duel-table">
        {metrics.map((m) => {
          const aMetric = a ? a.radar.metrics[m.key] : null;
          const bMetric = b ? b.radar.metrics[m.key] : null;

          const aVal = aMetric?.value ?? 0;
          const aPct = aMetric?.percentile ?? 0;
          const bVal = bMetric?.value ?? 0;
          const bPct = bMetric?.percentile ?? 0;

          const hasBoth = Boolean(a && b);
          const aLeads = hasBoth && aPct > bPct + 1.5;
          const bLeads = hasBoth && bPct > aPct + 1.5;

          return (
            <div key={m.key} className="duel-row" title={m.description}>
              {/* Player A side */}
              <div className={`duel-cell duel-a ${aLeads ? "winner" : ""}`}>
                {a ? (
                  <>
                    <span className="duel-pct-badge" style={{ backgroundColor: aLeads ? "rgba(184, 142, 45, 0.2)" : "rgba(255,255,255,0.04)", color: colorA }}>
                      {Math.round(aPct)}%
                    </span>
                    <span className="duel-val">{aVal.toFixed(2)}</span>
                    <div className="duel-bar-wrapper a-bar">
                      <div
                        className="duel-bar-fill"
                        style={{
                          width: `${Math.max(4, aPct)}%`,
                          background: `linear-gradient(270deg, ${colorA}, rgba(184, 142, 45, 0.4))`,
                        }}
                      />
                    </div>
                  </>
                ) : (
                  <span className="duel-empty">—</span>
                )}
              </div>

              {/* Center Metric Label */}
              <div className="duel-metric-center">
                <span className="duel-metric-name">{m.label}</span>
                <span className={`duel-category-dot cat-${m.category}`} />
              </div>

              {/* Player B side */}
              <div className={`duel-cell duel-b ${bLeads ? "winner" : ""}`}>
                {b ? (
                  <>
                    <div className="duel-bar-wrapper b-bar">
                      <div
                        className="duel-bar-fill"
                        style={{
                          width: `${Math.max(4, bPct)}%`,
                          background: `linear-gradient(90deg, ${colorB}, rgba(224, 96, 126, 0.4))`,
                        }}
                      />
                    </div>
                    <span className="duel-val">{bVal.toFixed(2)}</span>
                    <span className="duel-pct-badge" style={{ backgroundColor: bLeads ? "rgba(224, 96, 126, 0.2)" : "rgba(255,255,255,0.04)", color: colorB }}>
                      {Math.round(bPct)}%
                    </span>
                  </>
                ) : (
                  <span className="duel-empty">—</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div className="chartnote">
        Values represent per-90 rates · Percentiles (%) are calibrated against same-position peers.
      </div>
    </div>
  );
}
