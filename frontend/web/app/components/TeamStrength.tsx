"use client";

import { ScatterPlot, ThemeProvider } from "@withqwerty/campos-react";
import { useEffect, useMemo, useState } from "react";
import { PITCH_CHART_THEME } from "../lib/chartTheme";
import { TeamRating, TeamStrength as TeamStrengthData, api } from "../lib/api";
import ExpectedTable from "./ExpectedTable";

/** Team attack and defence, the one dimension a player-only dashboard cannot supply.

    A striker's 0.6 xG per 90 reads differently depending on the side around him, and until
    the match model there was nothing here that could say whether his attack was the best in
    the league or the worst. Two Poisson coefficients per team is the whole answer, and a
    quadrant is the natural way to read two coefficients at once. */
type Point = { id: string; name: string; x: number; y: number };

const ALL = "All competitions";

export default function TeamStrength() {
  const [data, setData] = useState<TeamStrengthData | null>(null);
  const [failed, setFailed] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [competition, setCompetition] = useState(ALL);
  // Strength describes how good a side is; the expected table asks whether its results
  // matched its chances. Same teams, two questions, one modal.
  const [view, setView] = useState<"strength" | "table">("strength");

  useEffect(() => {
    let stale = false;
    api
      .teamStrength()
      .then((d) => !stale && setData(d))
      .catch(() => !stale && setFailed(true));
    return () => {
      stale = true;
    };
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setIsOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen]);

  const competitions = useMemo(
    () => [ALL, ...new Set((data?.teams ?? []).map((t) => t.competition ?? "—"))],
    [data]
  );

  const shown = useMemo(
    () =>
      (data?.teams ?? []).filter(
        (t) => competition === ALL || (t.competition ?? "—") === competition
      ),
    [data, competition]
  );

  // Defence is negative when it is good, which is the sign that catches people out. It is
  // flipped here and named for what it means, so both axes read the same way: further out
  // is better. The underlying coefficient is untouched — only its presentation.
  const points: Point[] = shown.map((t) => ({
    id: String(t.team_id),
    name: t.name,
    x: t.attack,
    y: -t.defence,
  }));

  // Labelling 160 teams is a wall of text. Within one league every side fits; across all of
  // them only the corners are legible, and the corners are the interesting part anyway.
  const labelled = useMemo(() => {
    if (competition !== ALL) return shown.map((t) => String(t.team_id));
    const best = [...shown].sort((a, b) => b.attack - a.attack).slice(0, 4);
    const meanest = [...shown].sort((a, b) => a.defence - b.defence).slice(0, 4);
    const worst = [...shown].sort((a, b) => a.attack - b.attack).slice(0, 2);
    return [...new Set([...best, ...meanest, ...worst].map((t) => String(t.team_id)))];
  }, [shown, competition]);

  const highlighted = new Set(labelled);

  if (failed) return null;

  return (
    <>
      <button
        className="coverage-trigger"
        onClick={() => setIsOpen(true)}
        title="Attack and defence strength, per team"
      >
        <span className="coverage-trigger-label">Teams:</span>
        {data ? (
          <span className="coverage-trigger-value">{data.teams.length} rated</span>
        ) : (
          <span className="coverage-trigger-value dim">…</span>
        )}
      </button>

      {isOpen && data && (
        <div className="modal-backdrop" onClick={() => setIsOpen(false)}>
          <div className="coverage-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="picker-modal-header">
              <div>
                <div className="coverage-header-pill">Poisson attack / defence</div>
                <h3 className="picker-title">Team Strength</h3>
              </div>
              <button className="picker-close-btn" onClick={() => setIsOpen(false)}>
                ✕
              </button>
            </div>

            <div className="coverage-modal-body">
              <div className="strength-filter">
                <button
                  className={`chip ${view === "strength" ? "active" : ""}`}
                  onClick={() => setView("strength")}
                >
                  Attack / defence
                </button>
                <button
                  className={`chip ${view === "table" ? "active" : ""}`}
                  onClick={() => setView("table")}
                >
                  Expected table
                </button>
              </div>

              {view === "table" && <ExpectedTable />}

              {view === "strength" && (
              <>
              <div className="strength-filter">
                {competitions.map((name) => (
                  <button
                    key={name}
                    className={`chip ${competition === name ? "active" : ""}`}
                    onClick={() => setCompetition(name)}
                  >
                    {name}
                  </button>
                ))}
              </div>

              <div className="chart-frame strength-chart">
                <ThemeProvider value={PITCH_CHART_THEME}>
                  <ScatterPlot<Point>
                    points={points}
                    idKey="id"
                    xKey="x"
                    yKey="y"
                    labelKey="name"
                    xLabel="Attacking strength →"
                    yLabel="Defensive solidity →"
                    labelStrategy="manual"
                    labelIds={labelled}
                    markers={{
                      fill: ({ point }) =>
                        point && highlighted.has(point.id)
                          ? "var(--accent)"
                          : "rgba(233, 240, 230, 0.34)",
                      radius: ({ point }) => (point && highlighted.has(point.id) ? 5 : 3),
                    }}
                    labelStyle={{ fill: "var(--text)", fontSize: 9 }}
                  />
                </ThemeProvider>
              </div>

              <p className="why-note">
                Every team gets one coefficient for scoring and one for conceding, fitted so
                that goals follow a Poisson distribution — the classical football model, and
                the same one that turns two numbers back into a scoreline. Both are on a log
                scale where <strong>0 is an average side</strong>, so a team at +0.7 scores
                roughly twice as often as one at 0.
              </p>

              <p className="why-note">
                Top right is strong at both ends. The measured home advantage across these{" "}
                {data.matches.toLocaleString()} matches is a factor of{" "}
                <strong>{Math.exp(data.home_advantage).toFixed(2)}</strong> on a side&apos;s
                goal rate — worth about a quarter of a goal a game, which is most of why
                predicting a home win is such a hard baseline to beat.
              </p>

              <StrengthTable teams={shown} />
              </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/** The extremes as numbers, because a scatter shows shape and hides magnitude. */
function StrengthTable({ teams }: { teams: TeamRating[] }) {
  const byAttack = [...teams].sort((a, b) => b.attack - a.attack);
  const rows = teams.length > 12 ? [...byAttack.slice(0, 5), ...byAttack.slice(-3)] : byAttack;

  return (
    <table className="coverage-table">
      <thead>
        <tr>
          <th>Team</th>
          <th className="num">Attack</th>
          <th className="num">Defence</th>
          <th className="num">Matches</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((t) => (
          <tr key={t.team_id}>
            <td>
              <span className="coverage-comp">{t.name}</span>
              {t.competition && <span className="coverage-country">{t.competition}</span>}
            </td>
            <td className="num">{t.attack >= 0 ? "+" : ""}{t.attack.toFixed(2)}</td>
            <td className="num">{t.defence >= 0 ? "+" : ""}{t.defence.toFixed(2)}</td>
            <td className="num">{t.matches}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
