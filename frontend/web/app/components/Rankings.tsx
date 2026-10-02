"use client";

import { useEffect, useState } from "react";
import Bars, { Bar } from "./Bars";
import { Player, RankingRow, api } from "../lib/api";
/** League leaderboard, with the selected players highlighted. */
export default function Rankings({
  selectedIdA,
  selectedIdB,
  activeSlot,
  onPick,
}: {
  selectedIdA?: number;
  selectedIdB?: number;
  activeSlot: "A" | "B";
  onPick: (p: Player, target?: "A" | "B") => void;
}) {
  const [group, setGroup] = useState<string>("");
  const [gender, setGender] = useState<"all" | "female" | "male">("all");
  const [rows, setRows] = useState<RankingRow[]>([]);

  useEffect(() => {
    let stale = false;
    api
      .rankings(group || undefined, 10, gender)
      .then((r) => !stale && setRows(r))
      .catch(() => !stale && setRows([]));
    return () => {
      stale = true;
    };
  }, [group, gender]);

  const bars: Bar[] = rows.map((r, i) => {
    const isA = r.player_id === selectedIdA;
    const isB = r.player_id === selectedIdB;
    return {
      label: r.name,
      value: r.performance_score,
      display: r.performance_score.toFixed(1),
      highlight: isA || isB,
      customColor: isA ? "var(--a)" : isB ? "var(--b)" : undefined,
      rank: i + 1,
      detail: `${r.primary_position ?? r.position_group} · Performance Score ${r.performance_score.toFixed(1)}`,
    };
  });

  return (
    <div className="leaderboard-inner">
      <div className="panel-header" style={{ alignItems: "center", gap: 12, flexWrap: "wrap", justifyContent: "space-between" }}>
        <div>
          <label>Performance Leaderboard</label>
          <div className="leaderboard-sub">Overall top rated by peers</div>
        </div>

        {/* United Gender Bar (in the middle between title and position chips) */}
        <div className="leaderboard-gender-bar">
          {(["all", "female", "male"] as const).map((g) => (
            <button
              key={g}
              type="button"
              className={`leaderboard-gender-btn ${g} ${gender === g ? "active" : ""}`}
              onClick={() => setGender(g)}
            >
              {g === "all" ? "All" : g === "female" ? "Female" : "Male"}
            </button>
          ))}
        </div>

        {/* Position Group Filter Chips (on the right) */}
        <div className="pos-chips">
          {["", "GK", "DEF", "MID", "FWD"].map((g) => (
            <button
              key={g || "all"}
              type="button"
              className={`chip ${group === g ? "active" : ""}`}
              onClick={() => setGroup(g)}
            >
              {g || "All"}
            </button>
          ))}
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="muted" style={{ marginTop: 8 }}>
          No ranking data.
        </div>
      ) : (
        <Bars
          bars={bars}
          max={100}
          color="var(--accent)"
          onPick={(i) =>
            onPick(
              { id: rows[i].player_id, name: rows[i].name, country: null, gender: rows[i].gender },
              activeSlot
            )
          }
          onPickA={(i) =>
            onPick(
              { id: rows[i].player_id, name: rows[i].name, country: null, gender: rows[i].gender },
              "A"
            )
          }
          onPickB={(i) =>
            onPick(
              { id: rows[i].player_id, name: rows[i].name, country: null, gender: rows[i].gender },
              "B"
            )
          }
        />
      )}
      <div className="chartnote">
        Click to set Slot {activeSlot} · Hover for (+A / +B).
      </div>
    </div>
  );
}
