"use client";

import { ScatterPlot, ThemeProvider } from "@withqwerty/campos-react";
import { useEffect, useMemo, useState } from "react";
import { PITCH_CHART_THEME } from "../lib/chartTheme";
import { RADAR_AXES, api } from "../lib/api";

// Two metrics at once separates things a single ranking conflates: volume from quality.
// A striker high on shots but low on xG is taking bad ones; the radar cannot say that,
// because it shows each metric on its own axis.

type Row = { id: string; name: string; x: number; y: number };
type Marker = { playerId: number; name: string; color: string };
/** A selected player this chart cannot show, and the group they belong to instead. */
type Omitted = { name: string; group: string };

export default function MetricScatter({
  positionGroup,
  markers,
  omitted,
}: {
  positionGroup: string;
  markers: Marker[];
  omitted?: Omitted | null;
}) {
  const [xMetric, setXMetric] = useState("shots_per90");
  const [yMetric, setYMetric] = useState("xg_per90");
  // Each response is kept with the request it answered, so switching axes shows "loading"
  // by mismatch rather than by clearing state at the top of the effect.
  const request = `${xMetric}|${yMetric}|${positionGroup}`;
  const [result, setResult] = useState<{
    request: string;
    rows: Row[] | null;
    failed: boolean;
  } | null>(null);
  const current = result?.request === request ? result : null;
  const rows = current?.rows ?? null;
  const failed = current?.failed ?? false;

  useEffect(() => {
    let stale = false;
    Promise.all([
      api.distribution(xMetric, positionGroup),
      api.distribution(yMetric, positionGroup),
    ])
      .then(([xs, ys]) => {
        if (stale) return;
        // Joined on the player-season, not the player: eleven players hold one row in
        // each of two leagues, and joining on the player id silently merges them.
        const yById = new Map(ys.values.map((v) => [v.id, v.value]));
        setResult({
          request,
          failed: false,
          rows: xs.values
            .filter((v) => yById.has(v.id))
            .map((v) => ({
              id: v.id,
              // Just the name: the label is drawn inside the plot and a competition
              // suffix runs it off the edge. The id already keeps the two seasons apart.
              name: v.name,
              x: v.value,
              y: yById.get(v.id) as number,
            })),
        });
      })
      .catch(() => !stale && setResult({ request, rows: null, failed: true }));
    return () => {
      stale = true;
    };
  }, [request, xMetric, yMetric, positionGroup]);

  const labelOf = (metric: string) => RADAR_AXES.find(([key]) => key === metric)?.[1] ?? metric;

  // Markers are keyed by player, but a point is a player-season, so a marked player who
  // changed league appears twice and both are highlighted.
  const markerColour = useMemo(() => {
    const byPlayer = new Map(markers.map((m) => [String(m.playerId), m.color]));
    const out = new Map<string, string>();
    for (const row of rows ?? []) {
      const colour = byPlayer.get(row.id.split("-")[0]);
      if (colour) out.set(row.id, colour);
    }
    return out;
  }, [rows, markers]);

  // How much of one metric the other already tells you. Two axes that move together are a
  // thicker version of one axis, and the diagonal cloud is what says so.
  const correlation = useMemo(() => {
    if (!rows || rows.length < 3) return null;
    const n = rows.length;
    const mx = rows.reduce((s, r) => s + r.x, 0) / n;
    const my = rows.reduce((s, r) => s + r.y, 0) / n;
    let top = 0;
    let dx = 0;
    let dy = 0;
    for (const r of rows) {
      top += (r.x - mx) * (r.y - my);
      dx += (r.x - mx) ** 2;
      dy += (r.y - my) ** 2;
    }
    return dx && dy ? top / Math.sqrt(dx * dy) : null;
  }, [rows]);

  return (
    <div>
      <label>
        {labelOf(yMetric)} against {labelOf(xMetric)}, all {positionGroup}s
      </label>

      {failed && <div className="chartnote">Could not load the metrics.</div>}

      {/* Not an error. A percentile is a rank within a position group, so a forward has no
          place on a field of midfielders — but a marker that simply never appears reads as
          a broken chart, which is how this was reported. */}
      {omitted && (
        <div className="chartnote">
          {omitted.name} is not plotted here — this field is {positionGroup}s only, and{" "}
          {omitted.group}s are ranked against their own.
        </div>
      )}

      <div className="scatter-layout">
        <div className="axis-pickers">
          <AxisPicker axis="Y axis" value={yMetric} onChange={setYMetric} disabled={xMetric} />
          <AxisPicker axis="X axis" value={xMetric} onChange={setXMetric} disabled={yMetric} />
        </div>

        {rows && (
          <div className="chart-frame scatter-frame">
            <ThemeProvider value={PITCH_CHART_THEME}>
              <ScatterPlot<Row>
                points={rows}
                idKey="id"
                xKey="x"
                yKey="y"
                labelKey="name"
                xLabel={`${labelOf(xMetric)} per 90`}
                yLabel={`${labelOf(yMetric)} per 90`}
                labelStrategy="manual"
                labelIds={[...markerColour.keys()]}
                // The two medians cut the field into quadrants, which is what turns "a dot
                // somewhere in a cloud" into "above average at both".
                guides={[
                  { axis: "x", value: "median", label: "median",
                    stroke: "rgba(233,240,230,0.2)", strokeDasharray: "3 3" },
                  { axis: "y", value: "median", label: "median",
                    stroke: "rgba(233,240,230,0.2)", strokeDasharray: "3 3" },
                ]}
                markers={{
                  fill: ({ point }) =>
                    (point && markerColour.get(point.id)) ?? "rgba(233, 240, 230, 0.30)",
                  radius: ({ point }) => (point && markerColour.has(point.id) ? 6 : 2.6),
                  stroke: ({ point }) =>
                    point && markerColour.has(point.id) ? "var(--panel)" : "transparent",
                  strokeWidth: ({ point }) => (point && markerColour.has(point.id) ? 2 : 0),
                }}
                labelStyle={{
                  // The label carries the same identity as its marker, so it wears the
                  // same colour rather than the generic text token.
                  fill: ({ label }) => markerColour.get(label.id) ?? "var(--text)",
                  fontSize: 9,
                }}
                />
              </ThemeProvider>
          </div>
        )}
      </div>

      <div className="chartnote chart-centred narrow">
        {rows ? (
          <>
            {rows.length} {positionGroup} player-seasons, split at the median of each axis.
            Up and to the right is more of both.
            {correlation !== null && (
              <>
                {" "}These two move together at <strong>r = {correlation.toFixed(2)}</strong>
                {Math.abs(correlation) > 0.75
                  ? " — close enough that one is mostly a restatement of the other."
                  : Math.abs(correlation) < 0.25
                    ? " — near enough to independent that the pairing is informative."
                    : "."}
              </>
            )}
          </>
        ) : (
          "Loading the field…"
        )}
      </div>
    </div>
  );
}

/** One vertical column of metric chips per axis, standing beside the plot.

    The metric already on the other axis is disabled rather than merely allowed: plotting a
    metric against itself draws the line y = x and says nothing. */
function AxisPicker({
  axis,
  value,
  onChange,
  disabled,
}: {
  axis: string;
  value: string;
  onChange: (metric: string) => void;
  disabled: string;
}) {
  return (
    <div className="axis-picker">
      <span className="scatter-axis-tag">{axis}</span>
      {RADAR_AXES.map(([key, label]) => (
        <button
          key={key}
          className={`chip ${value === key ? "active" : ""}`}
          disabled={key === disabled}
          onClick={() => onChange(key)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
