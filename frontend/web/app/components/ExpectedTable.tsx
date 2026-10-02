"use client";

import { useEffect, useMemo, useState } from "react";
import { ExpectedTable as ExpectedTableData, api } from "../lib/api";

/** The league table the chances deserved, next to the real one.

    Expected points come from every shot's xG: each match's chances of a win, a draw and a
    loss follow from both sides' shots, and summing them gives the points a team's chances
    were worth. The gap to the real points is finishing, goalkeeping and luck. Measured on
    143 team-seasons, expected points in the first half of a season predict the second half
    better than the real points do — which is the reason to look at this table at all. */
export default function ExpectedTable() {
  // One fetch, kept with a flag for whether it has arrived, so nothing is reset inside the
  // effect body.
  const [result, setResult] = useState<{ data: ExpectedTableData | null } | null>(null);
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(() => {
    let stale = false;
    api
      .expectedTable()
      .then((d) => !stale && setResult({ data: d }))
      .catch(() => !stale && setResult({ data: null }));
    return () => {
      stale = true;
    };
  }, []);

  const seasons = useMemo(() => result?.data?.seasons ?? [], [result]);
  const names = seasons.map((s) => s.competition ?? `Competition ${s.competition_id}`);
  const current =
    picked ?? (names.includes("La Liga") ? "La Liga" : (names[0] ?? null));
  const season = seasons.find(
    (s) => (s.competition ?? `Competition ${s.competition_id}`) === current
  );

  if (!result) return <div className="chartnote">Loading the expected table…</div>;
  if (!result.data || !season)
    return <div className="chartnote">No expected table published.</div>;

  return (
    <>
      <div className="strength-filter">
        {names.map((name) => (
          <button
            key={name}
            className={`chip ${current === name ? "active" : ""}`}
            onClick={() => setPicked(name)}
          >
            {name}
          </button>
        ))}
      </div>

      <table className="coverage-table">
        <thead>
          <tr>
            <th className="num">#</th>
            <th>Team</th>
            <th className="num">P</th>
            <th className="num">Pts</th>
            <th className="num" title="Expected points, from every shot's xG">
              xPts
            </th>
            <th className="num" title="Points minus expected points">
              ±
            </th>
            <th className="num" title="xG for, penalties included">
              xG for
            </th>
            <th className="num" title="xG against">
              xG ag.
            </th>
          </tr>
        </thead>
        <tbody>
          {season.teams.map((t, i) => (
            <tr key={t.team_id}>
              <td className="num">{i + 1}</td>
              <td>
                <span className="coverage-comp">{t.name}</span>
              </td>
              <td className="num">{t.played}</td>
              <td className="num">
                <strong>{t.points}</strong>
              </td>
              <td className="num">{t.xpts.toFixed(1)}</td>
              <td
                className="num"
                style={{
                  color:
                    t.luck > 3 ? "var(--accent)" : t.luck < -3 ? "var(--b)" : undefined,
                }}
              >
                {t.luck >= 0 ? "+" : ""}
                {t.luck.toFixed(1)}
              </td>
              <td className="num">{t.xgf.toFixed(1)}</td>
              <td className="num">{t.xga.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="why-note">
        <strong>xPts</strong> is what each team&apos;s chances were worth: every shot counts
        as a chance of scoring equal to its xG, and both sides&apos; shots together give each
        match&apos;s probabilities of a win, a draw and a loss. <strong>±</strong> is the gap
        to the real points, which is finishing, goalkeeping and luck. It is worth reading
        because it is the part that does not last: over 143 team-seasons, expected points in
        the first half of a season predicted the second half better than the points actually
        won.
      </p>
      <p className="why-note">
        Two simplifications pull xPts slightly towards the middle: shots in the same move are
        treated as independent, and own goals are missing from the real scorelines, as they
        are everywhere else in this dashboard.
      </p>
    </>
  );
}
