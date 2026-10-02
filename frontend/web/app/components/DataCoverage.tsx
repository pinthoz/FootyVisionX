"use client";

import { useEffect, useState } from "react";
import { CatalogueEntry, Coverage, api } from "../lib/api";

/** Header pill reporting what the database holds, with the full picture behind a click.

    The open dataset mixes complete league seasons with fragments carrying the same
    competition name, so each season is measured against a full double round-robin
    rather than by match count alone — 34 matches and 380 matches must not look alike. */
export default function DataCoverage() {
  const [data, setData] = useState<Coverage | null>(null);
  const [failed, setFailed] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    let stale = false;
    api
      .coverage()
      .then((d) => !stale && setData(d))
      .catch(() => !stale && setFailed(true));
    return () => {
      stale = true;
    };
  }, []);

  // Escape closes the modal, matching the rest of the dashboard's overlays.
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setIsOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen]);

  const complete = data?.seasons.filter((s) => s.complete) ?? [];
  const fragments = data?.seasons.filter((s) => !s.complete) ?? [];

  const pick = (kind: CatalogueEntry["kind"], gender?: CatalogueEntry["gender"]) =>
    (data?.catalogue ?? []).filter(
      (e) =>
        !e.loaded && e.complete && e.kind === kind && (gender ? e.gender === gender : true)
    );

  const mensLeagues = pick("league", "men");
  const womensLeagues = pick("league", "women");
  const tournaments = pick("tournament");
  const junk = (data?.catalogue ?? []).filter((e) => !e.complete && !e.loaded);

  return (
    <>
      <button
        className="coverage-trigger"
        onClick={() => setIsOpen(true)}
        title="Which competitions are loaded, and what else is available"
      >
        <span className="coverage-trigger-label">Dataset:</span>
        {data ? (
          <span className="coverage-trigger-value">
            {data.matches.toLocaleString()} matches · {data.seasons.length}{" "}
            {data.seasons.length === 1 ? "season" : "seasons"}
          </span>
        ) : (
          <span className="coverage-trigger-value dim">{failed ? "unavailable" : "…"}</span>
        )}
      </button>

      {isOpen && (
        <div className="modal-backdrop" onClick={() => setIsOpen(false)}>
          <div className="coverage-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="picker-modal-header">
              <div>
                <div className="coverage-header-pill">StatsBomb Open Data</div>
                <h3 className="picker-title">Dataset Coverage</h3>
              </div>
              <button className="picker-close-btn" onClick={() => setIsOpen(false)}>
                ✕
              </button>
            </div>

            <div className="coverage-modal-body">
              {failed && (
                <div className="chartnote">
                  Could not reach the API. Start the backend to see which competitions are
                  loaded.
                </div>
              )}

              {data && (
                <>
                  <div className="coverage-totals">
                    <span>
                      <strong>{data.matches.toLocaleString()}</strong> matches
                    </span>
                    <span>
                      <strong>{data.players.toLocaleString()}</strong> player seasons
                    </span>
                    <span>
                      <strong>{complete.length}</strong> full{" "}
                      {complete.length === 1 ? "season" : "seasons"}
                    </span>
                  </div>

                  <CoverageGroup title="In the database" count={data.seasons.length}>
                    <table className="coverage-table">
                      <thead>
                        <tr>
                          <th>Competition</th>
                          <th>Season</th>
                          <th className="num">Matches</th>
                          <th className="num">Teams</th>
                          <th className="num">Players</th>
                          <th>Season coverage</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.seasons.map((s) => (
                          <tr
                            key={`${s.competition_id}-${s.season_id}`}
                            className={s.complete ? "" : "is-fragment"}
                          >
                            <td>
                              <span className="coverage-comp">{s.competition}</span>
                              {/* StatsBomb publishes some seasons as a single club's
                                  matches. Naming the club is the difference between
                                  "a tenth of the Bundesliga" and "all of Leverkusen's". */}
                              {s.focus_team && (
                                <span className="coverage-country">{s.focus_team} only</span>
                              )}
                              {s.country && <span className="coverage-country">{s.country}</span>}
                            </td>
                            <td className="coverage-season">{s.season}</td>
                            <td className="num">{s.matches}</td>
                            <td className="num">{s.teams}</td>
                            <td className="num">{s.players}</td>
                            <td>
                              <div className="coverage-cell">
                                <div className="coverage-bar">
                                  <span
                                    className={s.complete ? "full" : "partial"}
                                    style={{ width: `${Math.round(s.coverage * 100)}%` }}
                                  />
                                </div>
                                <span className="coverage-pct">
                                  {Math.round(s.coverage * 100)}%
                                </span>
                                <span
                                  className={`coverage-pill ${s.complete ? "full" : "partial"}`}
                                >
                                  {s.complete ? "Full" : "Fragment"}
                                </span>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </CoverageGroup>

                  {fragments.length > 0 && (
                    <div className="chartnote">
                      {fragments.length === 1
                        ? "One loaded season is"
                        : `${fragments.length} loaded seasons are`}{" "}
                      only a slice of the real competition, but{" "}
                      {fragments.reduce((n, s) => n + s.players, 0)} player seasons from{" "}
                      {fragments.length === 1 ? "it" : "them"} carry the same weight in
                      percentiles and similarity as everyone else.
                    </div>
                  )}

                  {mensLeagues.length > 0 && (
                    <CoverageGroup
                      title="Complete men's leagues, not loaded"
                      count={mensLeagues.length}
                      note="The only complete men's domestic seasons in the whole open dataset — and all in 2015/16, so percentiles stay comparable."
                    >
                      <CatalogueTable rows={mensLeagues} />
                    </CoverageGroup>
                  )}

                  {womensLeagues.length > 0 && (
                    <CoverageGroup
                      title="Complete women's leagues, not loaded"
                      count={womensLeagues.length}
                      note="The only recent complete league seasons in the open data. They belong in their own percentile pool — men's and women's cannot be mixed."
                    >
                      <CatalogueTable rows={womensLeagues} />
                    </CoverageGroup>
                  )}

                  {tournaments.length > 0 && (
                    <CoverageGroup
                      title="Complete tournaments"
                      count={tournaments.length}
                      note="Complete, but tournament-shaped: national teams and around seven matches per player at most, so the 600-minute floor would rule out nearly everyone."
                    >
                      <CatalogueTable rows={tournaments} />
                    </CoverageGroup>
                  )}

                  {junk.length > 0 && (
                    <CoverageGroup
                      title="Fragments — do not load"
                      count={junk.length}
                      note="These carry a league name and a season but only a slice of the matches. Every La Liga season outside 2015/16 is just Barcelona's own games."
                    >
                      <CatalogueTable rows={junk} />
                    </CoverageGroup>
                  )}

                  <div className="chartnote">
                    Nothing below the first table is in this instance yet — the dashboard
                    only ever reads what is already in Postgres, and never fetches a season
                    on demand. Load one from the command line with{" "}
                    <code>footyvision load -c &lt;cid&gt; -s &lt;sid&gt;</code> and reopen this
                    panel; each row carries its own command on hover.
                    Coverage is matches held over a full double round-robin of the teams
                    present. Portugal is absent from this dataset entirely, and the free
                    alternatives do not cover it either. Catalogue checked against the live
                    competition list on {data.catalogue_verified}.
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function CoverageGroup({
  title,
  count,
  note,
  children,
}: {
  title: string;
  count: number;
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="coverage-group">
      <div className="coverage-group-head">
        <span className="coverage-group-title">{title}</span>
        <span className="coverage-group-count">{count}</span>
      </div>
      {note && <div className="coverage-group-note">{note}</div>}
      <div className="coverage-table-scroll">{children}</div>
    </div>
  );
}

function CatalogueTable({ rows }: { rows: CatalogueEntry[] }) {
  return (
    <table className="coverage-table">
      <thead>
        <tr>
          <th>Competition</th>
          <th>Season</th>
          <th>cid / sid</th>
          <th className="num">Matches</th>
          <th className="num">Teams</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((e) => (
          <tr
            key={`${e.competition_id}-${e.season_id}`}
            className={e.complete ? "" : "is-fragment"}
            title={`footyvision load -c ${e.competition_id} -s ${e.season_id}`}
          >
            <td>
              <span className="coverage-comp">{e.competition}</span>
              <span className="coverage-country">{e.country}</span>
            </td>
            <td className="coverage-season">{e.season}</td>
            <td className="coverage-ids">
              {e.competition_id} / {e.season_id}
            </td>
            <td className="num">{e.matches}</td>
            <td className="num">{e.teams}</td>
            <td>
              <span className={`coverage-pill ${e.complete ? "ready" : "partial"}`}>
                {e.complete ? "Not loaded" : "Fragment"}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
