"use client";

import { Beeswarm, ThemeProvider } from "@withqwerty/campos-react";
import { useEffect, useMemo, useState } from "react";
import { PITCH_CHART_THEME } from "../lib/chartTheme";
import { RADAR_AXES, api, type Distribution } from "../lib/api";

// A percentile says where a player ranks. It does not say what he is ranked against:
// the 96th percentile can be out on his own or packed in with twenty others. This plots
// every player in the position group so that shape is visible, and puts the number back
// beside the dot — a swarm shows shape and hides magnitude.

type Marker = { playerId: number; name: string; color: string };
/** A selected player this chart cannot show, and the group they belong to instead. */
type Omitted = { name: string; group: string };

export default function MetricDistribution({
  positionGroup,
  markers,
  omitted,
}: {
  positionGroup: string;
  markers: Marker[];
  omitted?: Omitted | null;
}) {
  const [metric, setMetric] = useState("xg_per90");
  // A response is kept with the request it answered; one for an earlier metric or group
  // simply does not count, which is what clearing state at the top of the effect used to do.
  const request = `${metric}|${positionGroup}`;
  const [result, setResult] = useState<{
    request: string;
    data: Distribution | null;
    failed: boolean;
  } | null>(null);
  const current = result?.request === request ? result : null;
  const data = current?.data ?? null;
  const failed = current?.failed ?? false;

  useEffect(() => {
    let stale = false;
    api
      .distribution(metric, positionGroup)
      .then((d) => !stale && setResult({ request, data: d, failed: false }))
      .catch(() => !stale && setResult({ request, data: null, failed: true }));
    return () => {
      stale = true;
    };
  }, [request, metric, positionGroup]);

  const label = RADAR_AXES.find(([key]) => key === metric)?.[1] ?? metric;
  const marked = new Map(markers.map((m) => [m.playerId, m]));

  // The field's own shape, so the chart can show where the middle is rather than leaving
  // the reader to guess it from dot density.
  const summary = useMemo(() => {
    if (!data?.values.length) return null;
    const sorted = [...data.values].map((v) => v.value).sort((a, b) => a - b);
    const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
    return { median: at(0.5), upper: at(0.75), top: at(0.9), sorted };
  }, [data]);

  /** Where a player sits, as the number a scout would actually quote. */
  const standing = (playerId: number) => {
    if (!summary || !data) return null;
    const point = data.values.find((v) => v.player_id === playerId);
    if (!point) return null;
    const below = summary.sorted.filter((v) => v < point.value).length;
    return { value: point.value, percentile: Math.round((below / summary.sorted.length) * 100) };
  };

  return (
    <div>
      <label>Where he sits in the {positionGroup} field</label>
      <div className="filters chart-centred">
        {RADAR_AXES.map(([key, axisLabel]) => (
          <button
            key={key}
            className={`chip ${metric === key ? "active" : ""}`}
            onClick={() => setMetric(key)}
          >
            {axisLabel}
          </button>
        ))}
      </div>

      {failed && <div className="chartnote">Could not load the distribution.</div>}

      {/* Not an error. A percentile is a rank within a position group, so a forward has no
          place on a field of midfielders — but a marker that simply never appears reads as
          a broken chart, which is how this was reported. */}
      {omitted && (
        <div className="chartnote">
          {omitted.name} is not plotted here — this field is {positionGroup}s only, and{" "}
          {omitted.group}s are ranked against their own.
        </div>
      )}

      {data && summary && (
        <div className="chart-frame swarm-frame">
          <ThemeProvider value={PITCH_CHART_THEME}>
            <Beeswarm
              groups={[
                {
                  id: positionGroup,
                  // The axis names the metric and the note below carries the sample size;
                  // a group label here only repeats them, and its gutter pushed the swarm
                  // out of the frame.
                  label: "",
                  values: data.values.map((v) => {
                    const mark = marked.get(v.player_id);
                    return {
                      // Not the player id: eleven players hold two player-seasons, and
                      // keying on the player drops one of them and warns about it.
                      id: v.id,
                      value: v.value,
                      label: v.competition ? `${v.name} · ${v.competition}` : v.name,
                      ...(mark
                        ? {
                            highlight: {
                              label: mark.name.split(" ").slice(-1)[0],
                              color: mark.color,
                              radius: 7,
                              stroke: "var(--panel)",
                              strokeWidth: 2,
                            },
                          }
                        : {}),
                    };
                  }),
                },
              ]}
              metric={{
                label: `${label} per 90`,
                format: (v: number) => v.toFixed(2),
              }}
              // Three marks turn an undifferentiated cloud into a scale you can read a
              // position off: the middle of the field, the top quarter, the top tenth.
              referenceLines={[
                { value: summary.median, label: "median", color: "rgba(233,240,230,0.35)",
                  dash: "3 3" },
                { value: summary.upper, label: "top 25%", color: "rgba(233,240,230,0.22)",
                  dash: "2 4" },
                { value: summary.top, label: "top 10%", color: "rgba(184,142,45,0.45)",
                  dash: "2 4" },
              ]}
              populationColor={{ mode: "uniform", color: "rgba(233, 240, 230, 0.42)" }}
              // Lower opacity lets dense clusters accumulate into darker zones, which is
              // the whole point of a beeswarm and was lost at full strength.
              populationOpacity={0.4}
              dotRadius={2.9}
              // Tall enough that the three reference labels sit above the swarm
              // rather than across it: they are drawn at the top of each line, so
              // height is the only thing that buys them clear space.
              layout={{ viewBoxHeight: 260, groupLabelSize: 0, outerPadding: 20 }}
              // The library paints highlight labels with currentColor and exposes no
              // per-highlight text style, so two players cannot be told apart by colour
              // inside the chart. The key below does that job instead.
              labelStrategy="none"
              maxWidth={600}
            />
          </ThemeProvider>
        </div>
      )}

      <div className="chart-key chart-centred">
        {markers.map((m) => {
          const where = standing(m.playerId);
          return (
            <span key={m.playerId} className="chart-key-item" style={{ color: m.color }}>
              <span className="dot" style={{ background: m.color }} />
              {m.name}
              {where && (
                <span className="chart-key-stat">
                  {where.value.toFixed(2)} · {ordinal(where.percentile)} pct
                </span>
              )}
            </span>
          );
        })}
      </div>

      <div className="chartnote chart-centred">
        {data && summary
          ? `Each dot is one of ${data.count} ${positionGroup} player-seasons. ` +
            `The median is ${summary.median.toFixed(2)} and the top tenth starts at ` +
            `${summary.top.toFixed(2)}. Hover a dot for the player behind it.`
          : "Loading the field…"}
      </div>
    </div>
  );
}

function ordinal(n: number): string {
  const suffix = n % 100 >= 11 && n % 100 <= 13
    ? "th"
    : { 1: "st", 2: "nd", 3: "rd" }[n % 10] ?? "th";
  return `${n}${suffix}`;
}
