"use client";

import { Heatmap, ShotMap, ThemeProvider, type ShotMapProps } from "@withqwerty/campos-react";
import { useEffect, useMemo, useState } from "react";
import { PITCH_CHART_THEME } from "../lib/chartTheme";
import { api, type PitchMap } from "../lib/api";

// Where a player acts, and where they shoot from. Everything else on this page is a count
// per 90 minutes, and a count has no location: this is the view that shows a wing-back
// playing thirty metres higher than a full-back, or a "right winger" living on the left.
//
// The data is StatsBomb's, oriented to the player's own attack: x from 0 to 120 towards the
// opponent's goal, y from 0 to 80 with the player's left at 0. Campos puts the attacker's
// left at y = 100, so y is flipped on the way in — without it every map is a mirror image.

type Slot = { playerId: number; name: string; color: string };
// Read off the chart's own props rather than imported from campos-schema, which is only a
// transitive dependency here.
type Shot = ShotMapProps["shots"][number];

const toCamposX = (x: number) => (x / 120) * 100;
const toCamposY = (y: number) => 100 - (y / 80) * 100;

export default function PitchMaps({ players }: { players: Slot[] }) {
  return (
    <div className="pitch-maps">
      <div className="chartnote chart-centred">
        Drawn from each player&apos;s own point of view: attacking to the right, their left
        flank at the top.
      </div>
      <div className="pitch-maps-grid">
        {players.map((p) => (
          <PlayerPitch key={p.playerId} slot={p} />
        ))}
      </div>
    </div>
  );
}

function PlayerPitch({ slot }: { slot: Slot }) {
  // Kept with the player it was fetched for, so switching player shows "loading" by the
  // mismatch rather than by clearing state inside the effect.
  const [result, setResult] = useState<{ id: number; map: PitchMap | null } | null>(null);
  const current = result?.id === slot.playerId ? result : null;
  const map = current?.map ?? null;

  useEffect(() => {
    let stale = false;
    api
      .pitch(slot.playerId)
      .then((m) => !stale && setResult({ id: slot.playerId, map: m }))
      .catch(() => !stale && setResult({ id: slot.playerId, map: null }));
    return () => {
      stale = true;
    };
  }, [slot.playerId]);

  // One point per recorded action at the centre of its cell, with the same bins as the
  // published grid, so the heatmap redraws exactly the counts that were stored.
  const events = useMemo(() => {
    if (!map) return [];
    const out: { x: number; y: number }[] = [];
    map.cells.forEach((count, i) => {
      const x = (((i % map.cols) + 0.5) / map.cols) * 100;
      const y = 100 - ((Math.floor(i / map.cols) + 0.5) / map.rows) * 100;
      for (let n = 0; n < count; n++) out.push({ x, y });
    });
    return out;
  }, [map]);

  // Penalties are left off: one spot kick is worth ~0.76 xG and a player who took a few
  // would read as a far better chance-getter than open play says.
  const { shots, penalties } = useMemo(() => {
    if (!map) return { shots: [] as Shot[], penalties: 0 };
    const open = map.shots.filter((s) => !s.penalty);
    return {
      penalties: map.shots.length - open.length,
      shots: open.map<Shot>((s, i) => ({
        kind: "shot",
        id: `${map.player_id}-${i}`,
        matchId: "",
        teamId: "",
        playerId: String(map.player_id),
        playerName: slot.name,
        minute: 0,
        addedMinute: null,
        second: 0,
        period: 1,
        x: toCamposX(s.x),
        y: toCamposY(s.y),
        xg: s.xg,
        outcome: s.outcome,
        bodyPart: s.body_part,
        isOwnGoal: false,
        isPenalty: false,
        context: null,
        provider: "statsbomb",
        providerEventId: String(i),
      })),
    };
  }, [map, slot.name]);

  const z = map?.zones ?? {};
  const side =
    z.zone_left != null && z.zone_right != null
      ? z.zone_left > z.zone_right + 0.1
        ? "mostly on the left"
        : z.zone_right > z.zone_left + 0.1
          ? "mostly on the right"
          : "across both flanks"
      : null;

  return (
    <div className="pitch-player">
      <div className="pitch-player-name" style={{ color: slot.color }}>
        {slot.name}
      </div>
      {!current ? (
        <div className="chartnote">Loading the pitch map…</div>
      ) : !map ? (
        <div className="chartnote">No pitch map for this player.</div>
      ) : (
        <ThemeProvider value={PITCH_CHART_THEME}>
          <div className="chart-frame">
            <Heatmap
              events={events}
              gridX={map.cols}
              gridY={map.rows}
              attackingDirection="right"
              metricLabel="Actions"
              maxWidth={460}
            />
          </div>
          <div className="chartnote chart-centred">
            {map.actions.toLocaleString()} actions · {Math.round((z.zone_final_third ?? 0) * 100)}%
            in the final third{side ? ` · ${side}` : ""}
          </div>
          {/* Expected Threat: how much the completed passes and carries raised the chance of
              a goal, valued by where the ball went rather than how often the ball was passed. Shown
              against peers, as the radar is, because conversion differs by league. */}
          {map.xt_per90 != null && (
            <div
              className="chartnote chart-centred"
              title="Expected Threat added by completed open-play passes and carries, per 90 minutes. It values each move by how much closer it brought the ball to a goal."
            >
              <strong style={{ color: slot.color }}>{map.xt_per90.toFixed(2)} xT</strong> added
              per 90
              {map.xt_percentile != null && (
                <> · {Math.round(map.xt_percentile)}th percentile among peers</>
              )}
            </div>
          )}
          {shots.length > 0 ? (
            <div className="chart-frame">
              <ShotMap shots={shots} preset="statsbomb" crop="half" maxWidth={360} />
            </div>
          ) : (
            <div className="chartnote chart-centred">No open-play shots recorded.</div>
          )}
          {penalties > 0 && (
            <div className="chartnote chart-centred">
              {penalties} penalt{penalties === 1 ? "y" : "ies"} not shown.
            </div>
          )}
        </ThemeProvider>
      )}
    </div>
  );
}
